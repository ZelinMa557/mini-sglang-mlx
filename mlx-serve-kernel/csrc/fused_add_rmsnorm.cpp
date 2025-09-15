#include <assert.h>
#include <dlfcn.h>
#include <iostream>
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "fused_add_rmsnorm.h"
#include "util.h"
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif
namespace mlx_serve {

mx::array fused_add_rmsnorm(
    const mx::array& x,
    const mx::array& y,
    const mx::array& weight,
    float eps,
    mx::StreamOrDevice s_
) {
  auto out_type = x.dtype();

  auto s = to_stream(s_);

  return mx::array(
      x.shape(),
      out_type,
      std::make_shared<FusedAddRmsnorm>(s, eps),
      {x, y, weight});
}

#ifdef _METAL_
void FusedAddRmsnorm::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const mx::array& x = inputs[0];
  const mx::array& y = inputs[1];
  const mx::array& w = inputs[2];

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = 4;
  const int looped_limit = 4096;
  std::string op_name = "fused_add_rmsnorm";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += "_" + type_to_name(out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  out.set_data(mx::allocator::malloc(out.nbytes()));
  {
    auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());
    auto kernel = d.get_kernel(op_name, lib);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(y, 1);
    compute_encoder.set_input_array(w, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(eps_, 4);
    compute_encoder.set_bytes(axis_size, 5);
    compute_encoder.set_bytes(w_stride, 6);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}
#endif
}