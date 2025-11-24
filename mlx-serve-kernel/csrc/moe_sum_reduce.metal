#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

template <typename T, int N_READS = 4>
[[kernel]] void moe_sum_reduce(
    const device T* y,
    const device T* scores,
    device T* out,
    constant uint& topk_num,
    constant uint& hidden_dim,
    constant uint& y_stride_row,
    constant uint& scores_stride_0,
    constant uint& scores_stride_1,
    constant uint& out_stride_token,
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {

  const uint token_idx = pos.x;
  const uint base_hidden_idx = pos.y * N_READS;
  
  // Accumulate weighted sum for this token across topk experts
  float acc[N_READS];
  for (int i = 0; i < N_READS; i++) {
    acc[i] = 0.0f;
  }
  
  // For each expert output
  for (uint k = 0; k < topk_num; k++) {
    //const T score = scores[token_idx * scores_stride_0 + k * scores_stride_1];
    const float score = static_cast<float>(scores[token_idx * scores_stride_0 + k * scores_stride_1]);
    const uint y_row_idx = token_idx * topk_num + k;
    const device T* y_ptr = y + y_row_idx * y_stride_row;
    
    // Host code ensures that hidden_dim  % N_READS == 0
    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      acc[i] += static_cast<float>(y_ptr[base_hidden_idx + i]) * score;
    }
  }
  
  // Write output
  device T* out_ptr = out + token_idx * out_stride_token;
  #pragma unroll
  for (int i = 0; i < N_READS; i++) {
    out_ptr[base_hidden_idx + i] = static_cast<T>(acc[i]);
  }
}

template <typename T, int N_READS = 4>
[[kernel]] void moe_sum_reduce_with_reorder(
    const device T* y,
    const device T* scores,
    const device uint32_t* inv_order,
    device T* out,
    constant uint& topk_num,
    constant uint& hidden_dim,
    constant uint& y_stride_row,
    constant uint& scores_stride_0,
    constant uint& scores_stride_1,
    constant uint& out_stride_token,
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const uint out_token_idx = pos.x;
  const uint base_hidden_idx = pos.y * N_READS;
  
  // Accumulate weighted sum for this token across topk experts
  float acc[N_READS];
  for (int i = 0; i < N_READS; i++) {
    acc[i] = 0.0f;
  }
  
  // For each expert output
  for (uint k = 0; k < topk_num; k++) {
    const float score = scores[out_token_idx * scores_stride_0 + k * scores_stride_1];
    // Get the row index in y for this expert
    const int y_row_idx = inv_order[out_token_idx * topk_num + k];
    const device T* y_ptr = y + y_row_idx * y_stride_row;
    
    // Process N_READS elements per thread
    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      acc[i] += static_cast<float>(y_ptr[base_hidden_idx + i] * score);
    }
  }
  
  // Write output
  device T* out_ptr = out + out_token_idx * out_stride_token;
  for (int i = 0; i < N_READS; i++) {
    out_ptr[base_hidden_idx + i] = static_cast<T>(acc[i]);
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

