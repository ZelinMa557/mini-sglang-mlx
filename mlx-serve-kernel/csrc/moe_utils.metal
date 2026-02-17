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

// Src-first scatter broadcast for MoE prepare.
//
// Replaces the dst-first gather: out[i] = x[order[i] // K]
// with a src-first broadcast:    for each token t, write x[t] to K destinations.
//
// Uses inv_order to find destinations: out[inv_order[t*K + k]] = x[t]
//
// Dispatch: threadgroups = (token_num, 1, 1)
//           threads_per_group = (THREADS_PER_GROUP, 1, 1)
//
// Each threadgroup handles one src token. Threads within the group cooperatively:
//   1. Load the entire src row into threadgroup memory (coalesced read)
//   2. For each of K destinations, write the row out (coalesced write per dst)
template <typename T, int N_READS = 4, int THREADS_PER_GROUP = 256>
[[kernel]] void moe_scatter_broadcast(
    const device T* x,
    const device uint32_t* inv_order,
    device T* out,
    constant uint& topk_num,
    constant uint& hidden_dim,
    constant uint& x_stride_row,
    constant uint& out_stride_row,
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {

  const uint src_token = gid.x;
  const uint thread_id = tid.x;

  // Each thread handles a strided portion of the hidden dim
  const device T* src_ptr = x + src_token * x_stride_row;
  const device uint32_t* inv_ptr = inv_order + src_token * topk_num;

  // For each destination (K total)
  for (uint k = 0; k < topk_num; k++) {
    const uint dst_row = inv_ptr[k];
    device T* dst_ptr = out + dst_row * out_stride_row;

    // Threads cooperatively copy the entire row
    // Each thread copies N_READS elements at a time, striding by THREADS_PER_GROUP * N_READS
    for (uint offset = thread_id * N_READS; offset < hidden_dim; offset += THREADS_PER_GROUP * N_READS) {
      #pragma unroll
      for (int i = 0; i < N_READS; i++) {
        dst_ptr[offset + i] = src_ptr[offset + i];
      }
    }
  }
}

// clang-format off
#define instantiate_moe_utils(type_name, type)                     \
  instantiate_kernel("moe_sum_reduce_" #type_name, moe_sum_reduce, type) \
  instantiate_kernel("moe_sum_reduce_with_reorder_" #type_name, moe_sum_reduce_with_reorder, type) \
  instantiate_kernel("moe_scatter_broadcast_" #type_name, moe_scatter_broadcast, type)

instantiate_moe_utils(float16, half);
instantiate_moe_utils(float32, float);
instantiate_moe_utils(bfloat16, bfloat16_t);
// clang-format on

