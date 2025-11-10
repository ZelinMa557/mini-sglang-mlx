#include <assert.h>
#include <dlfcn.h>
#include <iostream>
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "moe_sum_reduce.h"
#include "util.h"
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif
namespace mlx_serve {

mx::array moe_sum_reduce(
    const mx::array& y,
    const mx::array& scores,
    mx::StreamOrDevice s_
) {
  auto out_type = y.dtype();

  auto s = to_stream(s_);

  // Validate inputs
  if (y.ndim() != 3) {
    throw std::runtime_error("moe_sum_reduce: y must be 3D [token_num, topk_num, hidden_dim]");
  }
  if (scores.ndim() != 2) {
    throw std::runtime_error("moe_sum_reduce: scores must be 2D [token_num, topk_num]");
  }
  
  const size_t token_num = y.shape(0);
  const size_t topk_num = y.shape(1);
  const size_t hidden_dim = y.shape(2);
  
  if (scores.shape(0) != token_num || scores.shape(1) != topk_num) {
    throw std::runtime_error("moe_sum_reduce: scores shape mismatch");
  }

  std::vector<int> out_shape = {static_cast<int>(token_num), static_cast<int>(hidden_dim)};

  return mx::array(
      out_shape,
      out_type,
      std::make_shared<MoeSumReduce>(s),
      {y, scores});
}

mx::array moe_sum_reduce_with_reorder(
    const mx::array& y,
    const mx::array& scores,
    const mx::array& inv_order,
    mx::StreamOrDevice s_
) {
  auto out_type = y.dtype();

  auto s = to_stream(s_);

  // Validate inputs
  if (y.ndim() != 3) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: y must be 3D [reordered_token_num, topk_num, hidden_dim]");
  }
  if (scores.ndim() != 2) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: scores must be 2D [token_num, topk_num]");
  }
  if (inv_order.ndim() != 1) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: inv_order must be 1D [token_num]");
  }
  
  const size_t reordered_token_num = y.shape(0);
  const size_t topk_num = y.shape(1);
  const size_t hidden_dim = y.shape(2);
  const size_t token_num = scores.shape(0);
  
  if (scores.shape(1) != topk_num) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: scores topk_num mismatch");
  }
  if (inv_order.shape(0) != token_num) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: inv_order shape mismatch");
  }
  if (inv_order.dtype() != mx::int32) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: inv_order must be int32");
  }

  std::vector<int> out_shape = {static_cast<int>(token_num), static_cast<int>(hidden_dim)};

  return mx::array(
      out_shape,
      out_type,
      std::make_shared<MoeSumReduceWithReorder>(s),
      {y, scores, inv_order});
}

#ifdef _METAL_
void MoeSumReduce::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const mx::array& y = inputs[0];
  const mx::array& scores = inputs[1];

  const uint32_t token_num = static_cast<uint32_t>(y.shape(0));
  const uint32_t topk_num = static_cast<uint32_t>(y.shape(1));
  const uint32_t hidden_dim = static_cast<uint32_t>(y.shape(2));

  const uint32_t y_stride_token = static_cast<uint32_t>(y.strides()[0]);
  const uint32_t y_stride_topk = static_cast<uint32_t>(y.strides()[1]);
  const uint32_t out_stride_token = static_cast<uint32_t>(out.strides()[0]);

  const int n_reads = 4;
  const int simd_size = 32;
  
  // Calculate threadgroup size
  size_t threadgroup_needed = (hidden_dim + n_reads - 1) / n_reads;
  size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
  size_t threadgroup_size = simd_size * simds_needed;
  
  std::string op_name = "moe_sum_reduce_" + type_to_name(out);
  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());
  auto kernel = d.get_kernel(op_name, lib);
  
  // Ensure threadgroup size doesn't exceed max
  threadgroup_size = std::min(threadgroup_size, static_cast<size_t>(kernel->maxTotalThreadsPerThreadgroup()));
  
  auto& compute_encoder = d.get_command_encoder(s.index);
  out.set_data(mx::allocator::malloc(out.nbytes()));
  
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(y, 0);
  compute_encoder.set_input_array(scores, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_bytes(token_num, 3);
  compute_encoder.set_bytes(topk_num, 4);
  compute_encoder.set_bytes(hidden_dim, 5);
  compute_encoder.set_bytes(y_stride_token, 6);
  compute_encoder.set_bytes(y_stride_topk, 7);
  compute_encoder.set_bytes(out_stride_token, 8);
  
  MTL::Size grid_dims(token_num, 1, 1);
  MTL::Size group_dims(threadgroup_size, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void MoeSumReduceWithReorder::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const mx::array& y = inputs[0];
  const mx::array& scores = inputs[1];
  const mx::array& inv_order = inputs[2];

  const uint32_t token_num = static_cast<uint32_t>(scores.shape(0));
  const uint32_t topk_num = static_cast<uint32_t>(y.shape(1));
  const uint32_t hidden_dim = static_cast<uint32_t>(y.shape(2));

  const uint32_t y_stride_token = static_cast<uint32_t>(y.strides()[0]);
  const uint32_t y_stride_topk = static_cast<uint32_t>(y.strides()[1]);
  const uint32_t out_stride_token = static_cast<uint32_t>(out.strides()[0]);

  const int n_reads = 4;
  const int simd_size = 32;
  
  // Calculate threadgroup size
  size_t threadgroup_needed = (hidden_dim + n_reads - 1) / n_reads;
  size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
  size_t threadgroup_size = simd_size * simds_needed;
  
  std::string op_name = "moe_sum_reduce_with_reorder_" + type_to_name(out);
  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());
  auto kernel = d.get_kernel(op_name, lib);
  
  // Ensure threadgroup size doesn't exceed max
  threadgroup_size = std::min(threadgroup_size, static_cast<size_t>(kernel->maxTotalThreadsPerThreadgroup()));
  
  auto& compute_encoder = d.get_command_encoder(s.index);
  out.set_data(mx::allocator::malloc(out.nbytes()));
  
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(y, 0);
  compute_encoder.set_input_array(scores, 1);
  compute_encoder.set_input_array(inv_order, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(token_num, 4);
  compute_encoder.set_bytes(topk_num, 5);
  compute_encoder.set_bytes(hidden_dim, 6);
  compute_encoder.set_bytes(y_stride_token, 7);
  compute_encoder.set_bytes(y_stride_topk, 8);
  compute_encoder.set_bytes(out_stride_token, 9);
  
  MTL::Size grid_dims(token_num, 1, 1);
  MTL::Size group_dims(threadgroup_size, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}
#endif
} // namespace mlx_serve

