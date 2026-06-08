#include <assert.h>
#include <dlfcn.h>
#include <iostream>
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "moe_utils.h"
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
  if (y.ndim() != 2) {
    throw std::runtime_error("moe_sum_reduce: y must be 2D [token_num * topk_num, hidden_dim]");
  }
  if (scores.ndim() != 2) {
    throw std::runtime_error("moe_sum_reduce: scores must be 2D [token_num, topk_num]");
  }
  
  const size_t token_num = scores.shape(0);
  const size_t topk_num = scores.shape(1);
  const size_t hidden_dim = y.shape(1);
  
  if (y.shape(0) != token_num * topk_num) {
    throw std::runtime_error("moe_sum_reduce: y shape mismatch, expected [token_num * topk_num, hidden_dim]");
  }

  return mx::array(
      {static_cast<int>(token_num), static_cast<int>(hidden_dim)},
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
  if (y.ndim() != 2) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: y must be 2D [token_num * topk_num, hidden_dim]");
  }
  if (scores.ndim() != 2) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: scores must be 2D [token_num, topk_num]");
  }
  if (inv_order.ndim() != 1) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: inv_order must be 1D [token_num * topk_num]");
  }
  
  const size_t token_num = scores.shape(0);
  const size_t topk_num = scores.shape(1);
  const size_t hidden_dim = y.shape(1);
  
  if (y.shape(0) != token_num * topk_num) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: y shape mismatch, expected [token_num * topk_num, hidden_dim]");
  }
  if (inv_order.shape(0) != token_num * topk_num) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: inv_order shape mismatch, expected [token_num * topk_num]");
  }
  if (inv_order.dtype() != mx::uint32) {
    throw std::runtime_error("moe_sum_reduce_with_reorder: inv_order must be uint32");
  }

  return mx::array(
      {static_cast<int>(token_num), static_cast<int>(hidden_dim)},
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

  const uint32_t token_num = static_cast<uint32_t>(scores.shape(0));
  const uint32_t topk_num = static_cast<uint32_t>(scores.shape(1));
  const uint32_t hidden_dim = static_cast<uint32_t>(y.shape(1));

  const uint32_t y_stride_row = static_cast<uint32_t>(y.strides()[0]);
  const uint32_t scores_stride_0 = static_cast<uint32_t>(scores.strides()[0]);
  const uint32_t scores_stride_1 = static_cast<uint32_t>(scores.strides()[1]);
  const uint32_t out_stride_token = static_cast<uint32_t>(out.strides()[0]);

  const int n_reads = 4;
  if (hidden_dim % n_reads != 0) {
    throw std::runtime_error("moe sum reduce kernel: hidden dim must be times of 4");
  }
  
  std::string op_name = "moe_sum_reduce_" + type_to_name(out);
  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());
  auto kernel = d.get_kernel(op_name, lib);
  
  auto& compute_encoder = mx::metal::get_command_encoder(s);
  out.set_data(mx::allocator::malloc(out.nbytes()));
  
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(y, 0);
  compute_encoder.set_input_array(scores, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_bytes(topk_num, 3);
  compute_encoder.set_bytes(y_stride_row, 4);
  compute_encoder.set_bytes(scores_stride_0, 5);
  compute_encoder.set_bytes(scores_stride_1, 6);
  compute_encoder.set_bytes(out_stride_token, 7);

  uint32_t dim0 = token_num;
  uint32_t dim1 = hidden_dim / n_reads;
  uint32_t dim2 = 1;
  // MTL::Size group_dims = mlx::core::get_block_dims(dim0, dim1, dim2);
  MTL::Size group_dims = MTL::Size(1,1,1);
  MTL::Size grid_dims = MTL::Size(dim0, dim1, dim2);
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
  const uint32_t topk_num = static_cast<uint32_t>(scores.shape(1));
  const uint32_t hidden_dim = static_cast<uint32_t>(y.shape(1));

  const uint32_t y_stride_row = static_cast<uint32_t>(y.strides()[0]);
  const uint32_t scores_stride_0 = static_cast<uint32_t>(scores.strides()[0]);
  const uint32_t scores_stride_1 = static_cast<uint32_t>(scores.strides()[1]);
  const uint32_t out_stride_token = static_cast<uint32_t>(out.strides()[0]);

  const int n_reads = 4;
  if (hidden_dim % n_reads != 0) {
    throw std::runtime_error("moe sum reduce kernel: hidden dim must be times of 4");
  }
  
  std::string op_name = "moe_sum_reduce_with_reorder_" + type_to_name(out);
  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());
  auto kernel = d.get_kernel(op_name, lib);
  
  auto& compute_encoder = mx::metal::get_command_encoder(s);
  out.set_data(mx::allocator::malloc(out.nbytes()));
  
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(y, 0);
  compute_encoder.set_input_array(scores, 1);
  compute_encoder.set_input_array(inv_order, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(topk_num, 4);
  compute_encoder.set_bytes(y_stride_row, 5);
  compute_encoder.set_bytes(scores_stride_0, 6);
  compute_encoder.set_bytes(scores_stride_1, 7);
  compute_encoder.set_bytes(out_stride_token, 8);
  
  uint32_t dim0 = token_num;
  uint32_t dim1 = hidden_dim / n_reads;
  uint32_t dim2 = 1;
  // MTL::Size group_dims = mlx::core::get_block_dims(dim0, dim1, dim2);
  MTL::Size group_dims = MTL::Size(1,1,1);
  MTL::Size grid_dims = MTL::Size(dim0, dim1, dim2);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}
#endif
} // namespace mlx_serve

