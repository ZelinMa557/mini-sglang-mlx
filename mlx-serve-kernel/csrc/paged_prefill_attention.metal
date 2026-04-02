#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/utils.h"
using namespace metal;

template <typename T>
using vec2_t = vec<T, 2>;

template <typename T>
METAL_FUNC inline void copy_vec2(
    threadgroup T* dst,
    int dst_idx,
    const device T* src,
    int src_idx) {
  *((threadgroup vec2_t<T>*)(dst + dst_idx)) =
      *((const device vec2_t<T>*)(src + src_idx));
}

// ============================================================================
// Paged Prefill (Extend) Attention
//
// Grid: (batch, num_q_heads, ceil(max_q_len / BLOCK_M))
//
// Q stays resident in threadgroup memory because it is reused across all KV
// blocks. K/V are streamed as 32x32 sub-tiles and consumed immediately by
// Apple GPU tensor ops (`mpp::tensor_ops::matmul2d`) so we no longer allocate
// a full K/V block in threadgroup memory.
//
// P (softmax probabilities) is stored as half since it's in [0,1].
// Accumulators and softmax state are float32.
// ============================================================================

template <
    typename T,
    short DK,        // head key dimension
    short DV,        // head value dimension
    short BLOCK_M,   // query tokens per threadgroup (16)
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
    threadgroup char* shmem_raw          [[threadgroup(0)]],
    uint3 tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]) {

  static_assert(BLOCK_M == 16, "paged_prefill_attention expects BLOCK_M == 16");
  static_assert(BLOCK_N == 32, "paged_prefill_attention expects BLOCK_N == 32");
  static_assert(NSG == 4, "paged_prefill_attention expects NSG == 4");

  constexpr int MMA_K = 32;
  constexpr int MMA_DV = 32;

  using Ext2D = dextents<int32_t, 2>;
  using QKMatmul = mpp::tensor_ops::matmul2d<
      mpp::tensor_ops::matmul2d_descriptor(
          BLOCK_M, BLOCK_N, MMA_K, false, true, false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
      execution_simdgroups<NSG>>;
  using PVMatmul = mpp::tensor_ops::matmul2d<
      mpp::tensor_ops::matmul2d_descriptor(
          BLOCK_M, MMA_DV, BLOCK_N, false, false, false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
      execution_simdgroups<NSG>>;

  const int cur_seq = tgpig.x;
  const int cur_head = tgpig.y;
  const int cur_block_m = tgpig.z;

  const int kv_group_num = num_q_heads / num_kv_heads;
  const int cur_kv_head = cur_head / kv_group_num;

  // ---- Sequence lengths ----
  const int q_start = qo_indptr[cur_seq];
  const int cur_seq_q_len = qo_indptr[cur_seq + 1] - q_start;

  const int kv_start = kv_indptr[cur_seq];
  const int cur_seq_kv_len = kv_indptr[cur_seq + 1] - kv_start;

  const int cur_prefix_len = prefix_lens[cur_seq];

  const int q_block_start = cur_block_m * BLOCK_M;
  if (q_block_start >= cur_seq_q_len) return;
  const int valid_m = min((int)BLOCK_M, cur_seq_q_len - q_block_start);

  // ---- Shared memory layout ----
  // sq:     BLOCK_M * DK elements of T
  // sqk:    BLOCK_M * MMA_K elements of T     (streamed Q sub-tile)
  // st:     BLOCK_N * max(MMA_K, MMA_DV) of T (streamed K/V sub-tile)
  // ss:     BLOCK_M * BLOCK_N floats          (QK^T scores)
  // so_tile:BLOCK_M * MMA_DV floats           (contiguous output sub-tile)
  // so:     BLOCK_M * DV floats               (full output accumulator)
  // s_emax: BLOCK_M floats
  // s_esum: BLOCK_M floats
  // sp:     BLOCK_M * BLOCK_N half            (P matrix)
  threadgroup T* sq = (threadgroup T*)shmem_raw;
  threadgroup T* sqk = sq + BLOCK_M * DK;
  threadgroup T* st = sqk + BLOCK_M * MMA_K;
  threadgroup float* ss = (threadgroup float*)(st + BLOCK_N * MMA_K);
  threadgroup float* so_tile = ss + BLOCK_M * BLOCK_N;
  threadgroup float* so = so_tile + BLOCK_M * MMA_DV;
  threadgroup float* s_emax = so + BLOCK_M * DV;
  threadgroup float* s_esum = s_emax + BLOCK_M;
  threadgroup half* sp = (threadgroup half*)(s_esum + BLOCK_M);

  constexpr int NW = 32;
  const int tid = sgitg * NW + tiisg;
  const int total_threads = NSG * NW;

  // ---- Load Q block ----
  const int q_stride_head = DK;
  const int q_stride_token = num_q_heads * DK;

  constexpr int VEC_WIDTH = 2;
  constexpr int DK_VEC = DK / VEC_WIDTH;
  constexpr int DV_VEC = DV / VEC_WIDTH;
  constexpr int MMA_K_VEC = MMA_K / VEC_WIDTH;
  constexpr int MMA_DV_VEC = MMA_DV / VEC_WIDTH;

  for (int m = 0; m < BLOCK_M; m++) {
    for (int d2 = tid; d2 < DK_VEC; d2 += total_threads) {
      const int d = d2 * VEC_WIDTH;
      if (m < valid_m) {
        int global_token = q_start + q_block_start + m;
        copy_vec2(
            sq, m * DK + d,
            Q,
            global_token * q_stride_token + cur_head * q_stride_head + d);
      } else {
        *((threadgroup vec2_t<T>*)(sq + m * DK + d)) = vec2_t<T>(T(0), T(0));
      }
    }
  }

  // Initialize accumulators
  for (int i = tid; i < BLOCK_M * DV; i += total_threads) {
    so[i] = 0.0f;
  }
  if (tid < BLOCK_M) {
    s_emax[tid] = -HUGE_VALF;
    s_esum[tid] = 0.0f;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int kv_cache_stride_k = num_kv_heads * DK;
  const int kv_cache_stride_v = num_kv_heads * DV;

  QKMatmul qk_mma;
  PVMatmul pv_mma;

  auto qk_scores = tensor<threadgroup float, Ext2D, tensor_inline>(
      ss, Ext2D(BLOCK_N, BLOCK_M));
  auto p_tile = tensor<threadgroup half, Ext2D, tensor_inline>(
      sp, Ext2D(BLOCK_N, BLOCK_M));

  // ---- Main loop over KV blocks ----
  for (int kv_block_start = 0; kv_block_start < cur_seq_kv_len; kv_block_start += BLOCK_N) {
    const int kv_block_end = min(kv_block_start + BLOCK_N, cur_seq_kv_len);
    const int valid_n = kv_block_end - kv_block_start;

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += total_threads) {
      ss[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- QK^T via tensor ops with streamed 8x32 / 32x32 sub-tiles ----
    for (int dk_base = 0; dk_base < DK; dk_base += MMA_K) {
      for (int i2 = tid; i2 < BLOCK_M * MMA_K_VEC; i2 += total_threads) {
        const int m = i2 / MMA_K_VEC;
        const int k2 = i2 % MMA_K_VEC;
        const int k = k2 * VEC_WIDTH;
        if (m < valid_m) {
          *((threadgroup vec2_t<T>*)(sqk + m * MMA_K + k)) =
              *((threadgroup vec2_t<T>*)(sq + m * DK + dk_base + k));
        } else {
          *((threadgroup vec2_t<T>*)(sqk + m * MMA_K + k)) =
              vec2_t<T>(T(0), T(0));
        }
      }

      for (int i2 = tid; i2 < BLOCK_N * MMA_K_VEC; i2 += total_threads) {
        const int token_local = i2 / MMA_K_VEC;
        const int k2 = i2 % MMA_K_VEC;
        const int k = k2 * VEC_WIDTH;
        if (token_local < valid_n) {
          int page_idx = kv_indices[kv_start + kv_block_start + token_local];
          copy_vec2(
              st, token_local * MMA_K + k,
              K_cache,
              page_idx * kv_cache_stride_k + cur_kv_head * DK + dk_base + k);
        } else {
          *((threadgroup vec2_t<T>*)(st + token_local * MMA_K + k)) =
              vec2_t<T>(T(0), T(0));
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto q_tile = tensor<threadgroup T, Ext2D, tensor_inline>(
          sqk, Ext2D(MMA_K, BLOCK_M));
      auto k_tile = tensor<threadgroup T, Ext2D, tensor_inline>(
          st, Ext2D(MMA_K, BLOCK_N));
      qk_mma.run(q_tile, k_tile, qk_scores);

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Scale + causal mask ----
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += total_threads) {
      int m = i / BLOCK_N;
      int n = i % BLOCK_N;

      float val = ss[m * BLOCK_N + n] * sm_scale;

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

    // Convert P (float ss) -> half sp
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += total_threads) {
      sp[i] = static_cast<half>(ss[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- P @ V via tensor ops with streamed 32x32 V sub-tiles ----
    for (int dv_base = 0; dv_base < DV; dv_base += MMA_DV) {
      for (int i2 = tid; i2 < BLOCK_M * MMA_DV_VEC; i2 += total_threads) {
        const int m = i2 / MMA_DV_VEC;
        const int d2 = i2 % MMA_DV_VEC;
        const int d = d2 * VEC_WIDTH;
        *((threadgroup float2*)(so_tile + m * MMA_DV + d)) =
            *((threadgroup float2*)(so + m * DV + dv_base + d));
      }

      for (int i2 = tid; i2 < BLOCK_N * MMA_DV_VEC; i2 += total_threads) {
        const int token_local = i2 / MMA_DV_VEC;
        const int d2 = i2 % MMA_DV_VEC;
        const int d = d2 * VEC_WIDTH;
        if (token_local < valid_n) {
          int page_idx = kv_indices[kv_start + kv_block_start + token_local];
          copy_vec2(
              st, token_local * MMA_DV + d,
              V_cache,
              page_idx * kv_cache_stride_v + cur_kv_head * DV + dv_base + d);
        } else {
          *((threadgroup vec2_t<T>*)(st + token_local * MMA_DV + d)) =
              vec2_t<T>(T(0), T(0));
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto v_tile = tensor<threadgroup T, Ext2D, tensor_inline>(
          st, Ext2D(MMA_DV, BLOCK_N));
      auto o_tile = tensor<threadgroup float, Ext2D, tensor_inline>(
          so_tile, Ext2D(MMA_DV, BLOCK_M));
      pv_mma.run(p_tile, v_tile, o_tile);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int i2 = tid; i2 < BLOCK_M * MMA_DV_VEC; i2 += total_threads) {
        const int m = i2 / MMA_DV_VEC;
        const int d2 = i2 % MMA_DV_VEC;
        const int d = d2 * VEC_WIDTH;
        *((threadgroup float2*)(so + m * DV + dv_base + d)) =
            *((threadgroup float2*)(so_tile + m * MMA_DV + d));
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // ---- Write output ----
  const int o_stride_token = num_q_heads * DV;
  const int o_stride_head = DV;

  for (int m = 0; m < valid_m; m++) {
    int global_token = q_start + q_block_start + m;
    float esum = s_esum[m];
    float inv_sum = (esum > 0.0f) ? (1.0f / esum) : 0.0f;

    for (int d2 = tid; d2 < DV_VEC; d2 += total_threads) {
      const int d = d2 * VEC_WIDTH;
      float2 out_f = *((threadgroup float2*)(so + m * DV + d)) * inv_sum;
      *((device vec2_t<T>*)(O + global_token * o_stride_token +
                            cur_head * o_stride_head + d)) =
          vec2_t<T>(static_cast<T>(out_f[0]), static_cast<T>(out_f[1]));
    }
  }
}


// ============================================================================
// Template Instantiations
// ============================================================================

#define instantiate_prefill(type_name, type, dk, dv) \
  instantiate_kernel("paged_prefill_attention_" #type_name "_dk" #dk "_dv" #dv, \
    paged_prefill_attention, type, dk, dv, 16, 32, 4)

// float16
instantiate_prefill(float16, half, 128, 128)
instantiate_prefill(float16, half, 256, 256)

// bfloat16
instantiate_prefill(bfloat16, bfloat16_t, 128, 128)
instantiate_prefill(bfloat16, bfloat16_t, 256, 256)
