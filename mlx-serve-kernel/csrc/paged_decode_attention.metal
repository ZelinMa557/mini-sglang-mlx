#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "mlx/backend/metal/kernels/utils.h"
using namespace metal;

// ============================================================================
// Stage 1: Grouped Paged Decode Attention
//
// Grid: (batch, ceil(num_q_heads / BLOCK_H), max_kv_splits)
//
// Shared memory Q/K/V tiles use native type T (half or bfloat16_t).
// simdgroup_matrix<T, 8, 8> is used for matmul -- works for both types.
// P (softmax probabilities) is stored as half since it's in [0,1].
// Accumulator (so) and softmax state are float32.
// ============================================================================

template <
    typename T,
    short DK,        // head key dimension
    short DV,        // head value dimension
    short BLOCK_H,   // query heads per threadgroup (8)
    short BLOCK_N,   // KV tokens per inner loop iteration (32)
    short NSG>       // number of SIMD groups (4)
[[kernel]] void paged_decode_attention_stage1(
    const device T* Q                 [[buffer(0)]],
    const device T* K_cache           [[buffer(1)]],
    const device T* V_cache           [[buffer(2)]],
    const device int32_t* kv_indptr   [[buffer(3)]],
    const device int32_t* kv_indices  [[buffer(4)]],
    const device int32_t* num_kv_splits_buf [[buffer(5)]],
    device float* Att_Out             [[buffer(6)]],
    device float* Att_Lse             [[buffer(7)]],
    constant float& sm_scale          [[buffer(8)]],
    constant int& num_q_heads         [[buffer(9)]],
    constant int& num_kv_heads        [[buffer(10)]],
    constant int& max_kv_splits       [[buffer(11)]],
    constant int& window_size         [[buffer(12)]],
    threadgroup char* shmem_raw       [[threadgroup(0)]],
    uint3 tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]) {

  // Type alias for simdgroup matrix matching T
  using T8x8 = simdgroup_matrix<T, 8, 8>;

  const int cur_batch = tgpig.x;
  const int cur_head_group = tgpig.y;
  const int split_kv_id = tgpig.z;

  const int kv_group_num = num_q_heads / num_kv_heads;
  const int valid_block_h = min((int)BLOCK_H, kv_group_num);

  const int cur_q_head_start = cur_head_group * valid_block_h;
  const int cur_kv_head = cur_q_head_start / kv_group_num;

  if (cur_q_head_start >= num_q_heads) return;

  const int actual_heads = min(valid_block_h, num_q_heads - cur_q_head_start);

  // ---- KV sequence range for this split ----
  const int kv_start = kv_indptr[cur_batch];
  const int cur_batch_seq_len = kv_indptr[cur_batch + 1] - kv_start;
  if (cur_batch_seq_len <= 0) return;

  const int kv_splits = num_kv_splits_buf[cur_batch];

  constexpr int MIN_BLOCK_KV = 32;
  const int kv_len_per_split = ((cur_batch_seq_len + kv_splits - 1) / kv_splits + MIN_BLOCK_KV - 1) / MIN_BLOCK_KV * MIN_BLOCK_KV;
  const int split_kv_start = kv_len_per_split * split_kv_id;
  const int split_kv_end = min(split_kv_start + kv_len_per_split, cur_batch_seq_len);

  if (split_kv_start >= split_kv_end) return;

  // ---- Sliding window boundaries ----
  const int win_start = (window_size > 0) ? max(1, cur_batch_seq_len - window_size) : 0;

  // ---- Shared memory layout ----
  // sq:  BLOCK_H * DK elements of T
  // sk:  BLOCK_N * DK elements of T
  // sv:  BLOCK_N * DV elements of T
  // sp:  BLOCK_H * BLOCK_N elements of half (P matrix for S@V)
  // ss:  BLOCK_H * BLOCK_N floats (QK^T scores)
  // so:  BLOCK_H * DV floats (output accumulator)
  // s_emax: BLOCK_H floats
  // s_esum: BLOCK_H floats
  threadgroup T* sq = (threadgroup T*)shmem_raw;
  threadgroup T* sk = sq + BLOCK_H * DK;
  threadgroup T* sv = sk + BLOCK_N * DK;
  threadgroup half* sp = (threadgroup half*)(sv + BLOCK_N * DV);
  threadgroup float* ss = (threadgroup float*)(sp + BLOCK_H * BLOCK_N);
  threadgroup float* so = ss + BLOCK_H * BLOCK_N;
  threadgroup float* s_emax = so + BLOCK_H * DV;
  threadgroup float* s_esum = s_emax + BLOCK_H;

  constexpr int NW = 32;
  const int tid = sgitg * NW + tiisg;
  const int total_threads = NSG * NW;

  // ---- Load Q into shared memory ----
  const int q_stride_batch = num_q_heads * DK;
  for (int h = 0; h < BLOCK_H; h++) {
    int global_head = cur_q_head_start + h;
    for (int d = tid; d < DK; d += total_threads) {
      sq[h * DK + d] = (h < actual_heads)
          ? Q[cur_batch * q_stride_batch + global_head * DK + d]
          : T(0);
    }
  }

  // Initialize accumulators
  for (int i = tid; i < BLOCK_H * DV; i += total_threads) {
    so[i] = 0.0f;
  }
  if (tid < BLOCK_H) {
    s_emax[tid] = -HUGE_VALF;
    s_esum[tid] = 0.0f;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int kv_cache_stride = num_kv_heads * DK;
  const int kv_cache_stride_v = num_kv_heads * DV;

  // ---- Main loop over KV blocks ----
  for (int block_start = split_kv_start; block_start < split_kv_end; block_start += BLOCK_N) {
    const int block_end = min(block_start + BLOCK_N, split_kv_end);
    const int valid_n = block_end - block_start;

    // ---- Load K block ----
    for (int n = tid; n < BLOCK_N * DK; n += total_threads) {
      const int token_local = n / DK;
      const int d = n % DK;
      if (token_local < valid_n) {
        int page_idx = kv_indices[kv_start + block_start + token_local];
        sk[token_local * DK + d] = K_cache[page_idx * kv_cache_stride + cur_kv_head * DK + d];
      } else {
        sk[token_local * DK + d] = T(0);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- QK^T using simdgroup_matrix<T> ----
    {
      constexpr int N_TILES = BLOCK_N / 8;
      constexpr int TILES_PER_SG = (N_TILES + NSG - 1) / NSG;

      for (int tile_idx = 0; tile_idx < TILES_PER_SG; tile_idx++) {
        int n_tile = sgitg * TILES_PER_SG + tile_idx;
        if (n_tile >= N_TILES) break;

        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

        constexpr int DK8 = DK / 8;
        for (int dk = 0; dk < DK8; dk++) {
          T8x8 mq;
          T8x8 mk;
          simdgroup_load(mq, sq + dk * 8, DK);
          simdgroup_load(mk, sk + dk * 8 + n_tile * 8 * DK, DK, 0, true);
          simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }

        simdgroup_store(mqk, ss + n_tile * 8, BLOCK_N, 0, false);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Scale + sliding window mask ----
    for (int i = tid; i < BLOCK_H * BLOCK_N; i += total_threads) {
      int h = i / BLOCK_N;
      int n = i % BLOCK_N;

      float val = ss[h * BLOCK_N + n] * sm_scale;

      int token_pos = block_start + n;
      bool is_valid = (n < valid_n) && (h < actual_heads);

      if (is_valid && window_size > 0) {
        is_valid = (token_pos >= win_start) || (token_pos == 0);
      }

      ss[h * BLOCK_N + n] = is_valid ? val : -HUGE_VALF;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Online softmax ----
    for (int h = sgitg; h < BLOCK_H; h += NSG) {
      float row_max = -HUGE_VALF;
      for (int n = tiisg; n < BLOCK_N; n += NW) {
        row_max = max(row_max, ss[h * BLOCK_N + n]);
      }
      row_max = simd_max(row_max);

      float old_max = s_emax[h];
      float new_max = max(old_max, row_max);
      float rescale = exp(old_max - new_max);

      float row_sum = 0.0f;
      for (int n = tiisg; n < BLOCK_N; n += NW) {
        float p = exp(ss[h * BLOCK_N + n] - new_max);
        ss[h * BLOCK_N + n] = p;
        row_sum += p;
      }
      row_sum = simd_sum(row_sum);

      for (int d = tiisg; d < DV; d += NW) {
        so[h * DV + d] *= rescale;
      }

      if (tiisg == 0) {
        s_esum[h] = s_esum[h] * rescale + row_sum;
        s_emax[h] = new_max;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Load V block ----
    for (int n = tid; n < BLOCK_N * DV; n += total_threads) {
      const int token_local = n / DV;
      const int d = n % DV;
      if (token_local < valid_n) {
        int page_idx = kv_indices[kv_start + block_start + token_local];
        sv[token_local * DV + d] = V_cache[page_idx * kv_cache_stride_v + cur_kv_head * DV + d];
      } else {
        sv[token_local * DV + d] = T(0);
      }
    }

    // Convert P (float ss) -> half sp for matmul
    // P values are in [0,1], half precision is sufficient
    for (int i = tid; i < BLOCK_H * BLOCK_N; i += total_threads) {
      sp[i] = static_cast<half>(ss[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- P @ V using simdgroup matmul ----
    // P is half, V is T. We use half8x8 for P and T8x8 for V.
    // simdgroup_multiply_accumulate(float8x8, half8x8, T8x8, float8x8) works
    // because the accumulator is float32.
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
          T8x8 mv;
          simdgroup_load(mp, sp + bn * 8, BLOCK_N);
          simdgroup_load(mv, sv + bn * 8 * DV + dv_tile * 8, DV);
          simdgroup_multiply_accumulate(mo, mp, mv, mo);
        }

        simdgroup_store(mo, so + dv_tile * 8, DV);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- Store partial results ----
  const int out_stride_batch = num_q_heads * max_kv_splits * DV;
  const int out_stride_head = max_kv_splits * DV;
  const int lse_stride_batch = num_q_heads * max_kv_splits;
  const int lse_stride_head = max_kv_splits;

  for (int h = 0; h < actual_heads; h++) {
    int global_head = cur_q_head_start + h;
    float esum = s_esum[h];
    float inv_sum = (esum > 0.0f) ? (1.0f / esum) : 0.0f;

    int out_offset = cur_batch * out_stride_batch + global_head * out_stride_head + split_kv_id * DV;
    for (int d = tid; d < DV; d += total_threads) {
      Att_Out[out_offset + d] = so[h * DV + d] * inv_sum;
    }

    if (tid == 0) {
      int lse_offset = cur_batch * lse_stride_batch + global_head * lse_stride_head + split_kv_id;
      Att_Lse[lse_offset] = s_emax[h] + log(esum);
    }
  }
}


// ============================================================================
// Stage 2: Cross-Split Reduction
// ============================================================================

template <typename T, short DV>
[[kernel]] void paged_decode_attention_stage2(
    const device float* Att_Out       [[buffer(0)]],
    const device float* Att_Lse       [[buffer(1)]],
    device T* O                       [[buffer(2)]],
    const device int32_t* kv_indptr   [[buffer(3)]],
    const device int32_t* num_kv_splits_buf [[buffer(4)]],
    constant int& num_q_heads         [[buffer(5)]],
    constant int& max_kv_splits       [[buffer(6)]],
    uint2 tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]) {

  const int cur_batch = tgpig.x;
  const int cur_head = tgpig.y;

  const int cur_batch_seq_len = kv_indptr[cur_batch + 1] - kv_indptr[cur_batch];
  if (cur_batch_seq_len <= 0) return;

  const int kv_splits = num_kv_splits_buf[cur_batch];

  constexpr int MIN_BLOCK_KV = 32;
  const int kv_len_per_split = ((cur_batch_seq_len + kv_splits - 1) / kv_splits + MIN_BLOCK_KV - 1) / MIN_BLOCK_KV * MIN_BLOCK_KV;

  const int out_stride_batch = num_q_heads * max_kv_splits * DV;
  const int out_stride_head = max_kv_splits * DV;
  const int lse_stride_batch = num_q_heads * max_kv_splits;
  const int lse_stride_head = max_kv_splits;

  constexpr int NW = 32;

  float e_max = -HUGE_VALF;
  float e_sum = 0.0f;

  constexpr int ELEMS_PER_THREAD = (DV + NW - 1) / NW;
  float acc[ELEMS_PER_THREAD];
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    acc[i] = 0.0f;
  }

  for (int split = 0; split < max_kv_splits; split++) {
    int split_start = kv_len_per_split * split;
    int split_end = min(split_start + kv_len_per_split, cur_batch_seq_len);
    if (split_start >= split_end) break;

    int lse_offset = cur_batch * lse_stride_batch + cur_head * lse_stride_head + split;
    float lse_val = Att_Lse[lse_offset];

    float new_max = max(e_max, lse_val);
    float old_scale = exp(e_max - new_max);
    float new_scale = exp(lse_val - new_max);

    int out_offset = cur_batch * out_stride_batch + cur_head * out_stride_head + split * DV;
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      int d = tiisg + i * NW;
      if (d < DV) {
        acc[i] = acc[i] * old_scale + new_scale * Att_Out[out_offset + d];
      }
    }

    e_sum = e_sum * old_scale + new_scale;
    e_max = new_max;
  }

  int o_offset = cur_batch * num_q_heads * DV + cur_head * DV;
  float inv_sum = (e_sum > 0.0f) ? (1.0f / e_sum) : 0.0f;

  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    int d = tiisg + i * NW;
    if (d < DV) {
      O[o_offset + d] = static_cast<T>(acc[i] * inv_sum);
    }
  }
}


// ============================================================================
// Template Instantiations
// ============================================================================

#define instantiate_stage1(type_name, type, dk, dv) \
  instantiate_kernel("paged_decode_attention_stage1_" #type_name "_dk" #dk "_dv" #dv, \
    paged_decode_attention_stage1, type, dk, dv, 8, 32, 4)

#define instantiate_stage2(type_name, type, dv) \
  instantiate_kernel("paged_decode_attention_stage2_" #type_name "_dv" #dv, \
    paged_decode_attention_stage2, type, dv)

// float16
instantiate_stage1(float16, half, 64, 64)
instantiate_stage1(float16, half, 128, 128)
instantiate_stage1(float16, half, 512, 512)

instantiate_stage2(float16, half, 64)
instantiate_stage2(float16, half, 128)
instantiate_stage2(float16, half, 512)

// bfloat16
instantiate_stage1(bfloat16, bfloat16_t, 64, 64)
instantiate_stage1(bfloat16, bfloat16_t, 128, 128)
instantiate_stage1(bfloat16, bfloat16_t, 512, 512)

instantiate_stage2(bfloat16, bfloat16_t, 64)
instantiate_stage2(bfloat16, bfloat16_t, 128)
instantiate_stage2(bfloat16, bfloat16_t, 512)
