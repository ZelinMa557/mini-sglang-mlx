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
// Stage 1: Grouped Paged Decode Attention
//
// Grid: (batch, num_kv_heads, max_kv_splits)
//
// Q stays resident in threadgroup memory across KV blocks. K/V are streamed as
// 32x32 sub-tiles and consumed immediately by Apple GPU tensor ops. We also
// cache the 32 page indices of the current KV block in threadgroup memory so
// both QK and PV reuse the same gathered indirection.
// ============================================================================

template <
    typename T,
    short DK,
    short DV,
    short BLOCK_H,
    short BLOCK_N,
    short NSG>
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
    threadgroup char* shmem_raw       [[threadgroup(0)]],
    uint3 tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]) {

  static_assert(BLOCK_N == 32, "paged_decode_attention expects BLOCK_N == 32");
  static_assert(NSG == 4, "paged_decode_attention expects NSG == 4");
  static_assert(BLOCK_H == 8 || BLOCK_H == 16,
                "paged_decode_attention expects BLOCK_H in {8, 16}");

  constexpr int MMA_K = 32;
  constexpr int MMA_DV = 32;
  constexpr int VEC_WIDTH = 2;
  constexpr int DK_VEC = DK / VEC_WIDTH;
  constexpr int DV_VEC = DV / VEC_WIDTH;
  constexpr int MMA_K_VEC = MMA_K / VEC_WIDTH;
  constexpr int MMA_DV_VEC = MMA_DV / VEC_WIDTH;

  using Ext2D = dextents<int32_t, 2>;
  using QKMatmul = mpp::tensor_ops::matmul2d<
      mpp::tensor_ops::matmul2d_descriptor(
          BLOCK_H, BLOCK_N, MMA_K, false, true, false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
      execution_simdgroups<NSG>>;
  using PVMatmul = mpp::tensor_ops::matmul2d<
      mpp::tensor_ops::matmul2d_descriptor(
          BLOCK_H, MMA_DV, BLOCK_N, false, false, false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
      execution_simdgroups<NSG>>;

  const int cur_batch = tgpig.x;
  const int cur_kv_head = tgpig.y;
  const int split_kv_id = tgpig.z;

  const int kv_group_num = num_q_heads / num_kv_heads;
  if ((BLOCK_H == 8 && kv_group_num > 8) || (BLOCK_H == 16 && kv_group_num <= 8)) {
    return;
  }

  const int actual_heads = kv_group_num;
  const int cur_q_head_start = cur_kv_head * kv_group_num;

  const int kv_start = kv_indptr[cur_batch];
  const int cur_batch_seq_len = kv_indptr[cur_batch + 1] - kv_start;
  if (cur_batch_seq_len <= 0) return;

  const int kv_splits = num_kv_splits_buf[cur_batch];
  constexpr int MIN_BLOCK_KV = 32;
  const int kv_len_per_split =
      ((cur_batch_seq_len + kv_splits - 1) / kv_splits + MIN_BLOCK_KV - 1) /
      MIN_BLOCK_KV * MIN_BLOCK_KV;
  const int split_kv_start = kv_len_per_split * split_kv_id;
  const int split_kv_end = min(split_kv_start + kv_len_per_split, cur_batch_seq_len);
  if (split_kv_start >= split_kv_end) return;

  // ---- Shared memory layout ----
  // sq:      BLOCK_H * DK elements of T
  // sqk:     BLOCK_H * MMA_K elements of T
  // st:      BLOCK_N * max(MMA_K, MMA_DV) elements of T
  // s_idx:   BLOCK_N page indices
  // ss:      BLOCK_H * BLOCK_N floats
  // so_tile: BLOCK_H * MMA_DV floats
  // so:      BLOCK_H * DV floats
  // s_emax:  BLOCK_H floats
  // s_esum:  BLOCK_H floats
  // sp:      BLOCK_H * BLOCK_N half
  threadgroup T* sq = (threadgroup T*)shmem_raw;
  threadgroup T* sqk = sq + BLOCK_H * DK;
  threadgroup T* st = sqk + BLOCK_H * MMA_K;
  threadgroup int32_t* s_idx = (threadgroup int32_t*)(st + BLOCK_N * MMA_K);
  threadgroup float* ss = (threadgroup float*)(s_idx + BLOCK_N);
  threadgroup float* so_tile = ss + BLOCK_H * BLOCK_N;
  threadgroup float* so = so_tile + BLOCK_H * MMA_DV;
  threadgroup float* s_emax = so + BLOCK_H * DV;
  threadgroup float* s_esum = s_emax + BLOCK_H;
  threadgroup half* sp = (threadgroup half*)(s_esum + BLOCK_H);

  constexpr int NW = 32;
  const int tid = sgitg * NW + tiisg;
  const int total_threads = NSG * NW;

  const int q_stride_batch = num_q_heads * DK;
  for (int h = 0; h < BLOCK_H; h++) {
    int global_head = cur_q_head_start + h;
    for (int d2 = tid; d2 < DK_VEC; d2 += total_threads) {
      const int d = d2 * VEC_WIDTH;
      if (h < actual_heads) {
        copy_vec2(
            sq, h * DK + d,
            Q,
            cur_batch * q_stride_batch + global_head * DK + d);
      } else {
        *((threadgroup vec2_t<T>*)(sq + h * DK + d)) = vec2_t<T>(T(0), T(0));
      }
    }
  }

  for (int i = tid; i < BLOCK_H * DV; i += total_threads) {
    so[i] = 0.0f;
  }
  if (tid < BLOCK_H) {
    s_emax[tid] = -HUGE_VALF;
    s_esum[tid] = 0.0f;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int kv_cache_stride_k = num_kv_heads * DK;
  const int kv_cache_stride_v = num_kv_heads * DV;

  QKMatmul qk_mma;
  PVMatmul pv_mma;

  auto qk_scores = tensor<threadgroup float, Ext2D, tensor_inline>(
      ss, Ext2D(BLOCK_N, BLOCK_H));
  auto p_tile = tensor<threadgroup half, Ext2D, tensor_inline>(
      sp, Ext2D(BLOCK_N, BLOCK_H));

  for (int block_start = split_kv_start; block_start < split_kv_end; block_start += BLOCK_N) {
    const int block_end = min(block_start + BLOCK_N, split_kv_end);
    const int valid_n = block_end - block_start;

    for (int i = tid; i < BLOCK_H * BLOCK_N; i += total_threads) {
      ss[i] = 0.0f;
    }

    for (int n = tid; n < BLOCK_N; n += total_threads) {
      if (n < valid_n) {
        s_idx[n] = kv_indices[kv_start + block_start + n];
      } else {
        s_idx[n] = -1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int dk_base = 0; dk_base < DK; dk_base += MMA_K) {
      for (int i2 = tid; i2 < BLOCK_H * MMA_K_VEC; i2 += total_threads) {
        const int h = i2 / MMA_K_VEC;
        const int k2 = i2 % MMA_K_VEC;
        const int k = k2 * VEC_WIDTH;
        if (h < actual_heads) {
          *((threadgroup vec2_t<T>*)(sqk + h * MMA_K + k)) =
              *((threadgroup vec2_t<T>*)(sq + h * DK + dk_base + k));
        } else {
          *((threadgroup vec2_t<T>*)(sqk + h * MMA_K + k)) =
              vec2_t<T>(T(0), T(0));
        }
      }

      for (int i2 = tid; i2 < BLOCK_N * MMA_K_VEC; i2 += total_threads) {
        const int token_local = i2 / MMA_K_VEC;
        const int k2 = i2 % MMA_K_VEC;
        const int k = k2 * VEC_WIDTH;
        if (token_local < valid_n) {
          int page_idx = s_idx[token_local];
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
          sqk, Ext2D(MMA_K, BLOCK_H));
      auto k_tile = tensor<threadgroup T, Ext2D, tensor_inline>(
          st, Ext2D(MMA_K, BLOCK_N));
      qk_mma.run(q_tile, k_tile, qk_scores);

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = tid; i < BLOCK_H * BLOCK_N; i += total_threads) {
      int h = i / BLOCK_N;
      int n = i % BLOCK_N;
      float val = ss[h * BLOCK_N + n] * sm_scale;
      bool is_valid = (h < actual_heads) && (n < valid_n);
      ss[h * BLOCK_N + n] = is_valid ? val : -HUGE_VALF;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    for (int i = tid; i < BLOCK_H * BLOCK_N; i += total_threads) {
      sp[i] = static_cast<half>(ss[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int dv_base = 0; dv_base < DV; dv_base += MMA_DV) {
      for (int i2 = tid; i2 < BLOCK_H * MMA_DV_VEC; i2 += total_threads) {
        const int h = i2 / MMA_DV_VEC;
        const int d2 = i2 % MMA_DV_VEC;
        const int d = d2 * VEC_WIDTH;
        *((threadgroup float2*)(so_tile + h * MMA_DV + d)) =
            *((threadgroup float2*)(so + h * DV + dv_base + d));
      }

      for (int i2 = tid; i2 < BLOCK_N * MMA_DV_VEC; i2 += total_threads) {
        const int token_local = i2 / MMA_DV_VEC;
        const int d2 = i2 % MMA_DV_VEC;
        const int d = d2 * VEC_WIDTH;
        if (token_local < valid_n) {
          int page_idx = s_idx[token_local];
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
          so_tile, Ext2D(MMA_DV, BLOCK_H));
      pv_mma.run(p_tile, v_tile, o_tile);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int i2 = tid; i2 < BLOCK_H * MMA_DV_VEC; i2 += total_threads) {
        const int h = i2 / MMA_DV_VEC;
        const int d2 = i2 % MMA_DV_VEC;
        const int d = d2 * VEC_WIDTH;
        *((threadgroup float2*)(so + h * DV + dv_base + d)) =
            *((threadgroup float2*)(so_tile + h * MMA_DV + d));
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  const int out_stride_batch = num_q_heads * max_kv_splits * DV;
  const int out_stride_head = max_kv_splits * DV;
  const int lse_stride_batch = num_q_heads * max_kv_splits;
  const int lse_stride_head = max_kv_splits;
  const int out_head_base = cur_q_head_start * out_stride_head;
  const int lse_head_base = cur_q_head_start * lse_stride_head;

  for (int h = 0; h < actual_heads; h++) {
    int global_head = cur_q_head_start + h;
    float esum = s_esum[h];
    float inv_sum = (esum > 0.0f) ? (1.0f / esum) : 0.0f;

    int out_offset = cur_batch * out_stride_batch + out_head_base +
                     h * out_stride_head + split_kv_id * DV;
    for (int d2 = tid; d2 < DV_VEC; d2 += total_threads) {
      const int d = d2 * VEC_WIDTH;
      float2 out_f = *((threadgroup float2*)(so + h * DV + d)) * inv_sum;
      *((device float2*)(Att_Out + out_offset + d)) = out_f;
    }

    if (tid == 0) {
      int lse_offset = cur_batch * lse_stride_batch + lse_head_base +
                       h * lse_stride_head + split_kv_id;
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

  (void)sgitg;

  const int cur_batch = tgpig.x;
  const int cur_head = tgpig.y;

  const int cur_batch_seq_len = kv_indptr[cur_batch + 1] - kv_indptr[cur_batch];
  if (cur_batch_seq_len <= 0) return;

  const int kv_splits = num_kv_splits_buf[cur_batch];
  constexpr int MIN_BLOCK_KV = 32;
  const int kv_len_per_split =
      ((cur_batch_seq_len + kv_splits - 1) / kv_splits + MIN_BLOCK_KV - 1) /
      MIN_BLOCK_KV * MIN_BLOCK_KV;

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

#define instantiate_stage1(type_name, type, dk, dv, block_h) \
  instantiate_kernel("paged_decode_attention_stage1_" #type_name "_dk" #dk "_dv" #dv "_bh" #block_h, \
    paged_decode_attention_stage1, type, dk, dv, block_h, 32, 4)

#define instantiate_stage2(type_name, type, dv) \
  instantiate_kernel("paged_decode_attention_stage2_" #type_name "_dv" #dv, \
    paged_decode_attention_stage2, type, dv)

instantiate_stage1(float16, half, 128, 128, 8)
instantiate_stage1(float16, half, 128, 128, 16)
instantiate_stage1(float16, half, 256, 256, 8)
instantiate_stage1(float16, half, 256, 256, 16)

instantiate_stage2(float16, half, 128)
instantiate_stage2(float16, half, 256)

instantiate_stage1(bfloat16, bfloat16_t, 128, 128, 8)
instantiate_stage1(bfloat16, bfloat16_t, 128, 128, 16)
instantiate_stage1(bfloat16, bfloat16_t, 256, 256, 8)
instantiate_stage1(bfloat16, bfloat16_t, 256, 256, 16)
instantiate_stage2(bfloat16, bfloat16_t, 128)
instantiate_stage2(bfloat16, bfloat16_t, 256)
