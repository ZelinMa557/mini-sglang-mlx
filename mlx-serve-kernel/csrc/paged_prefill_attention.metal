#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "mlx/backend/metal/kernels/utils.h"
using namespace metal;

// ============================================================================
// Paged Prefill (Extend) Attention
//
// Grid: (batch, num_q_heads, ceil(max_q_len / BLOCK_M))
//
// NOTE: Shared memory and simdgroup matrices always use half (float16).
// When T = bfloat16_t, data is converted on load/store boundaries.
// This is because Metal simdgroup_matrix only supports half, not bfloat16.
// ============================================================================

template <
    typename T,
    short DK,        // head key dimension
    short DV,        // head value dimension
    short BLOCK_M,   // query tokens per threadgroup (8)
    short BLOCK_N,   // KV tokens per inner loop iteration (32)
    short NSG>       // number of SIMD groups (4)
[[kernel]] void paged_prefill_attention(
    const device T* Q                    [[buffer(0)]],
    device T* O                          [[buffer(1)]],
    const device T* K_cache              [[buffer(2)]],
    const device T* V_cache              [[buffer(3)]],
    const device int32_t* qo_indptr      [[buffer(4)]],
    const device int32_t* kv_indptr      [[buffer(5)]],
    const device int32_t* kv_indices     [[buffer(6)]],
    const device int32_t* prefix_lens    [[buffer(7)]],
    constant float& sm_scale             [[buffer(8)]],
    constant int& num_q_heads            [[buffer(9)]],
    constant int& num_kv_heads           [[buffer(10)]],
    constant int& window_size            [[buffer(11)]],
    threadgroup half* shmem              [[threadgroup(0)]],
    uint3 tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]) {

  const int cur_seq = tgpig.x;
  const int cur_head = tgpig.y;
  const int cur_block_m = tgpig.z;

  const int kv_group_num = num_q_heads / num_kv_heads;
  const int cur_kv_head = cur_head / kv_group_num;

  // ---- Sequence lengths ----
  const int q_start = qo_indptr[cur_seq];
  const int q_end = qo_indptr[cur_seq + 1];
  const int cur_seq_q_len = q_end - q_start;

  const int kv_start = kv_indptr[cur_seq];
  const int kv_end = kv_indptr[cur_seq + 1];
  const int cur_seq_kv_len = kv_end - kv_start;

  const int cur_prefix_len = prefix_lens[cur_seq];

  const int q_block_start = cur_block_m * BLOCK_M;
  if (q_block_start >= cur_seq_q_len) return;
  const int valid_m = min((int)BLOCK_M, cur_seq_q_len - q_block_start);

  // ---- Shared memory layout (always half) ----
  threadgroup half* sq = shmem;
  threadgroup half* sk = sq + BLOCK_M * DK;
  threadgroup half* sv = sk + BLOCK_N * DK;
  threadgroup float* ss = (threadgroup float*)(sv + BLOCK_N * DV);
  threadgroup float* so = ss + BLOCK_M * BLOCK_N;
  threadgroup float* s_emax = so + BLOCK_M * DV;
  threadgroup float* s_esum = s_emax + BLOCK_M;

  constexpr int NW = 32;
  const int tid = sgitg * NW + tiisg;
  const int total_threads = NSG * NW;

  // ---- Load Q block into shared memory (convert T -> half) ----
  const int q_stride_head = DK;
  const int q_stride_token = num_q_heads * DK;

  for (int m = 0; m < BLOCK_M; m++) {
    for (int d = tid; d < DK; d += total_threads) {
      if (m < valid_m) {
        int global_token = q_start + q_block_start + m;
        sq[m * DK + d] = static_cast<half>(Q[global_token * q_stride_token + cur_head * q_stride_head + d]);
      } else {
        sq[m * DK + d] = 0;
      }
    }
  }

  // Initialize output accumulator and softmax state
  for (int i = tid; i < BLOCK_M * DV; i += total_threads) {
    so[i] = 0.0f;
  }
  if (tid < BLOCK_M) {
    s_emax[tid] = -HUGE_VALF;
    s_esum[tid] = 0.0f;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Cache strides
  const int kv_cache_stride_k = num_kv_heads * DK;
  const int kv_cache_stride_v = num_kv_heads * DV;

  // ---- Main loop over KV blocks ----
  for (int kv_block_start = 0; kv_block_start < cur_seq_kv_len; kv_block_start += BLOCK_N) {
    const int kv_block_end = min(kv_block_start + BLOCK_N, cur_seq_kv_len);
    const int valid_n = kv_block_end - kv_block_start;

    // ---- Load K block (convert T -> half) ----
    for (int n = tid; n < BLOCK_N * DK; n += total_threads) {
      const int token_local = n / DK;
      const int d = n % DK;
      if (token_local < valid_n) {
        int page_idx = kv_indices[kv_start + kv_block_start + token_local];
        sk[token_local * DK + d] = static_cast<half>(K_cache[page_idx * kv_cache_stride_k + cur_kv_head * DK + d]);
      } else {
        sk[token_local * DK + d] = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- QK^T ----
    {
      constexpr int N_TILES = BLOCK_N / 8;
      constexpr int TILES_PER_SG = (N_TILES + NSG - 1) / NSG;

      for (int tile_idx = 0; tile_idx < TILES_PER_SG; tile_idx++) {
        int n_tile = sgitg * TILES_PER_SG + tile_idx;
        if (n_tile >= N_TILES) break;

        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

        constexpr int DK8 = DK / 8;
        for (int dk = 0; dk < DK8; dk++) {
          simdgroup_half8x8 mq;
          simdgroup_half8x8 mk;
          simdgroup_load(mq, sq + dk * 8, DK);
          simdgroup_load(mk, sk + dk * 8 + n_tile * 8 * DK, DK, 0, true);
          simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }

        simdgroup_store(mqk, ss + n_tile * 8, BLOCK_N, 0, false);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Apply scale, causal mask, sliding window mask ----
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += total_threads) {
      int m = i / BLOCK_N;
      int n = i % BLOCK_N;

      float val = ss[m * BLOCK_N + n];
      val *= sm_scale;

      bool is_valid = (m < valid_m) && (n < valid_n);

      if (is_valid) {
        int kv_pos = kv_block_start + n;

        // Causal mask: only for extend region
        if (kv_pos >= cur_prefix_len) {
          int k_extend_offset = kv_pos - cur_prefix_len;
          int q_offset = q_block_start + m;
          if (q_offset < k_extend_offset) {
            is_valid = false;
          }
        }

        // Sliding window mask
        if (is_valid && window_size > 0) {
          if (kv_pos != 0) {
            int q_abs = cur_prefix_len + q_block_start + m;
            if (kv_pos + window_size <= q_abs) {
              is_valid = false;
            }
          }
        }
      }

      ss[m * BLOCK_N + n] = is_valid ? val : -HUGE_VALF;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Online softmax ----
    for (int m = sgitg; m < BLOCK_M; m += NSG) {
      float row_max = -HUGE_VALF;
      for (int n = tiisg; n < BLOCK_N; n += NW) {
        row_max = max(row_max, ss[m * BLOCK_N + n]);
      }
      row_max = simd_max(row_max);

      float old_max = s_emax[m];
      float new_max = max(old_max, row_max);
      float rescale = exp(old_max - new_max);

      float row_sum = 0.0f;
      for (int n = tiisg; n < BLOCK_N; n += NW) {
        float p = exp(ss[m * BLOCK_N + n] - new_max);
        ss[m * BLOCK_N + n] = p;
        row_sum += p;
      }
      row_sum = simd_sum(row_sum);

      for (int d = tiisg; d < DV; d += NW) {
        so[m * DV + d] *= rescale;
      }

      if (tiisg == 0) {
        s_esum[m] = s_esum[m] * rescale + row_sum;
        s_emax[m] = new_max;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Load V block (convert T -> half) ----
    for (int n = tid; n < BLOCK_N * DV; n += total_threads) {
      const int token_local = n / DV;
      const int d = n % DV;
      if (token_local < valid_n) {
        int page_idx = kv_indices[kv_start + kv_block_start + token_local];
        sv[token_local * DV + d] = static_cast<half>(V_cache[page_idx * kv_cache_stride_v + cur_kv_head * DV + d]);
      } else {
        sv[token_local * DV + d] = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- P @ V ----
    // Convert P (float in ss) to half in sp (reuse sk scratch)
    threadgroup half* sp = sk;
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += total_threads) {
      sp[i] = static_cast<half>(ss[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      constexpr int DV_TILES = DV / 8;
      constexpr int TILES_PER_SG = (DV_TILES + NSG - 1) / NSG;

      for (int tile_idx = 0; tile_idx < TILES_PER_SG; tile_idx++) {
        int dv_tile = sgitg * TILES_PER_SG + tile_idx;
        if (dv_tile >= DV_TILES) break;

        simdgroup_float8x8 mo;
        simdgroup_load(mo, so + dv_tile * 8, DV);

        constexpr int BN8 = BLOCK_N / 8;
        for (int bn = 0; bn < BN8; bn++) {
          simdgroup_half8x8 mp;
          simdgroup_half8x8 mv;
          simdgroup_load(mp, sp + bn * 8, BLOCK_N);
          simdgroup_load(mv, sv + bn * 8 * DV + dv_tile * 8, DV);
          simdgroup_multiply_accumulate(mo, mp, mv, mo);
        }

        simdgroup_store(mo, so + dv_tile * 8, DV);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- Write output (convert float -> T) ----
  const int o_stride_token = num_q_heads * DV;
  const int o_stride_head = DV;

  for (int m = 0; m < valid_m; m++) {
    int global_token = q_start + q_block_start + m;
    float esum = s_esum[m];
    float inv_sum = (esum > 0.0f) ? (1.0f / esum) : 0.0f;

    for (int d = tid; d < DV; d += total_threads) {
      O[global_token * o_stride_token + cur_head * o_stride_head + d] =
          static_cast<T>(so[m * DV + d] * inv_sum);
    }
  }
}


// ============================================================================
// Template Instantiations
// ============================================================================

#define instantiate_prefill(type_name, type, dk, dv) \
  instantiate_kernel("paged_prefill_attention_" #type_name "_dk" #dk "_dv" #dv, \
    paged_prefill_attention, type, dk, dv, 8, 32, 4)

// float16
instantiate_prefill(float16, half, 64, 64)
instantiate_prefill(float16, half, 128, 128)
instantiate_prefill(float16, half, 512, 512)

// bfloat16
instantiate_prefill(bfloat16, bfloat16_t, 64, 64)
instantiate_prefill(bfloat16, bfloat16_t, 128, 128)
instantiate_prefill(bfloat16, bfloat16_t, 512, 512)
