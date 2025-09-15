
#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include <assert.h>
#include "util.h"
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif
namespace mx = mlx::core;

namespace mlx_serve {
namespace {
constexpr int n_per_thread = 4;
}
mx::array varlen_rope(
    const mx::array& x,
    const mx::array& positions,
    const int dims,
    const float base,
    mx::StreamOrDevice s = {});

class VarlenRope : public mx::Primitive {
 public:
  explicit VarlenRope(mx::Stream stream, const int dims, const float base)
      : mx::Primitive(stream), dims_(dims), base_(base) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override { assert(false);}
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  /** The name of primitive. */
  const char* name() const override {
    return "VarlenRope";
  }
 private:
  const int dims_;
  const float base_;
};

void VarlenRope::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  assert(inputs.size() == 2); // in, positions
  assert(outputs.size() == 1);
  auto& in = inputs[0];
  auto& positions = inputs[1];
  auto& out = outputs[0];

  assert(in.shape(1) == positions.shape(0));
  assert(positions.dtype() == mx::int32);
  if (in.ndim() != 3) { // n_heads, n_batch, n_dim
    throw std::runtime_error("[CachedRope] Input must have 3 dimensions");
  }

  auto& s = out.primitive().stream();
  auto& d = mx::metal::device(s.device);

  size_t strides[3];
  size_t out_strides[3];
  size_t mat_size = in.shape(-2) * in.shape(-1);
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  strides[0] = in.strides()[0];
  strides[1] = in.strides()[1];
  strides[2] = in.strides()[2];
  out_strides[0] = out.strides()[0];
  out_strides[1] = out.strides()[1];
  out_strides[2] = out.strides()[2];
  assert(in.flags().row_contiguous);

  // Special case for inference (single time step)
  bool single = in.shape(1) == 1;

  bool with_freqs = inputs.size() == 3;
  std::ostringstream kname;
  kname << "varlen_rope_" << (single ? "single_" : "") << type_to_name(in);
  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());
  auto kernel = d.get_kernel(kname.str(), lib);
  auto& compute_encoder = d.get_command_encoder(s.index);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(positions, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_bytes(base_, 3);

  size_t n_batch = in.size() / mat_size;
  MTL::Size group_dims;
  MTL::Size grid_dims;
  if (single) {
    compute_encoder.set_bytes(out_strides, 1, 4);
    uint32_t dim0 = dims_ / 2;
    group_dims = mlx::core::get_block_dims(dim0, n_batch, 1);
    grid_dims = MTL::Size(dim0, n_batch, 1);
  } else {
    compute_encoder.set_bytes(strides, 3, 4);
    compute_encoder.set_bytes(out_strides, 3, 5);
    compute_encoder.set_bytes(n_batch, 6);
    uint32_t dim0 = dims_ / 2;
    uint32_t dim1 = in.shape(-2);
    uint32_t dim2 = (n_batch + n_per_thread - 1) / n_per_thread;
    group_dims = mlx::core::get_block_dims(dim0, dim1, dim2);
    grid_dims = MTL::Size(dim0, dim1, dim2);
  }
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

mx::array varlen_rope(
    const mx::array& x,
    const mx::array& positions,
    const int dims,
    const float base,
    mx::StreamOrDevice s_
) {
  auto out_type = x.dtype();
  auto s = to_stream(s_);
  return mx::array(
      x.shape(),
      out_type,
      std::make_shared<VarlenRope>(s, dims, base),
      {x, positions});
}

} // namespace mlx_serve