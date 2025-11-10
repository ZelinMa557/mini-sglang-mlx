#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

// Version 1: Only fuse y = (y * scores[..., None]).sum(axis=-2)
// Input: y [token_num, topk_num, hidden_dim], scores [token_num, topk_num]
// Output: [token_num, hidden_dim]
template <typename T, int N_READS = 4>
[[kernel]] void moe_sum_reduce(
    const device T* y,
    const device float* scores,
    device T* out,
    constant uint& token_num,
    constant uint& topk_num,
    constant uint& hidden_dim,
    constant uint& y_stride_token,
    constant uint& y_stride_topk,
    constant uint& out_stride_token,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  
  // Each threadgroup handles one token
  const uint token_idx = gid;
  if (token_idx >= token_num) return;
  
  // Each thread handles N_READS elements in hidden_dim
  const uint base_hidden_idx = lid * N_READS;
  
  // Accumulate weighted sum for this token across topk experts
  float acc[N_READS];
  for (int i = 0; i < N_READS; i++) {
    acc[i] = 0.0f;
  }
  
  // For each expert output
  for (uint k = 0; k < topk_num; k++) {
    const float score = scores[token_idx * topk_num + k];
    const device T* y_ptr = y + token_idx * y_stride_token + k * y_stride_topk;
    
    // Process N_READS elements per thread
    if (base_hidden_idx + N_READS <= hidden_dim) {
      for (int i = 0; i < N_READS; i++) {
        acc[i] += static_cast<float>(y_ptr[base_hidden_idx + i]) * score;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (base_hidden_idx + i < hidden_dim) {
          acc[i] += static_cast<float>(y_ptr[base_hidden_idx + i]) * score;
        }
      }
    }
  }
  
  // Write output
  device T* out_ptr = out + token_idx * out_stride_token;
  if (base_hidden_idx + N_READS <= hidden_dim) {
    for (int i = 0; i < N_READS; i++) {
      out_ptr[base_hidden_idx + i] = static_cast<T>(acc[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base_hidden_idx + i < hidden_dim) {
        out_ptr[base_hidden_idx + i] = static_cast<T>(acc[i]);
      }
    }
  }
}

// Version 2: Fuse y = y[inv_order] and y = (y * scores[..., None]).sum(axis=-2)
// Input: y [reordered_token_num, topk_num, hidden_dim], scores [token_num, topk_num], inv_order [token_num]
// Output: [token_num, hidden_dim]
template <typename T, int N_READS = 4>
[[kernel]] void moe_sum_reduce_with_reorder(
    const device T* y,
    const device float* scores,
    const device int* inv_order,
    device T* out,
    constant uint& token_num,
    constant uint& topk_num,
    constant uint& hidden_dim,
    constant uint& y_stride_token,
    constant uint& y_stride_topk,
    constant uint& out_stride_token,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  
  // Each threadgroup handles one output token
  const uint out_token_idx = gid;
  if (out_token_idx >= token_num) return;
  
  // Get the original token index in reordered y
  const int reordered_token_idx = inv_order[out_token_idx];
  
  // Each thread handles N_READS elements in hidden_dim
  const uint base_hidden_idx = lid * N_READS;
  
  // Accumulate weighted sum for this token across topk experts
  float acc[N_READS];
  for (int i = 0; i < N_READS; i++) {
    acc[i] = 0.0f;
  }
  
  // For each expert output
  for (uint k = 0; k < topk_num; k++) {
    const float score = scores[out_token_idx * topk_num + k];
    const device T* y_ptr = y + reordered_token_idx * y_stride_token + k * y_stride_topk;
    
    // Process N_READS elements per thread
    if (base_hidden_idx + N_READS <= hidden_dim) {
      for (int i = 0; i < N_READS; i++) {
        acc[i] += static_cast<float>(y_ptr[base_hidden_idx + i]) * score;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (base_hidden_idx + i < hidden_dim) {
          acc[i] += static_cast<float>(y_ptr[base_hidden_idx + i]) * score;
        }
      }
    }
  }
  
  // Write output
  device T* out_ptr = out + out_token_idx * out_stride_token;
  if (base_hidden_idx + N_READS <= hidden_dim) {
    for (int i = 0; i < N_READS; i++) {
      out_ptr[base_hidden_idx + i] = static_cast<T>(acc[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base_hidden_idx + i < hidden_dim) {
        out_ptr[base_hidden_idx + i] = static_cast<T>(acc[i]);
      }
    }
  }
}

// clang-format off
#define instantiate_moe_sum_reduce(type_name, type)                     \
  instantiate_kernel("moe_sum_reduce_" #type_name, moe_sum_reduce, type) \
  instantiate_kernel("moe_sum_reduce_with_reorder_" #type_name, moe_sum_reduce_with_reorder, type)

instantiate_moe_sum_reduce(float16, half);
instantiate_moe_sum_reduce(float32, float);
instantiate_moe_sum_reduce(bfloat16, bfloat16_t);
// clang-format on

