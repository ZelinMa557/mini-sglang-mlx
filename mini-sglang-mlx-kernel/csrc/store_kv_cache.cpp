#include "store_kv_cache.h"
#include "util.h"

#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mini_sglang_mlx {

void store_kv_cache(
    mx::array& k_cache,
    mx::array& v_cache,
    const mx::array& indices,
    const mx::array& k,
    const mx::array& v,
    mx::StreamOrDevice s_) {
#ifdef _METAL_
  if (k_cache.ndim() != 3 || v_cache.ndim() != 3) {
    throw std::runtime_error(
        "store_kv_cache: k_cache and v_cache must be 3D "
        "(num_pages, num_kv_heads, head_dim)");
  }
  if (indices.ndim() != 1) {
    throw std::runtime_error("store_kv_cache: indices must be 1D");
  }
  if (k.ndim() != 3 || v.ndim() != 3) {
    throw std::runtime_error(
        "store_kv_cache: k and v must be 3D (length, num_kv_heads, head_dim)");
  }
  if (k_cache.dtype() != v_cache.dtype() || k_cache.dtype() != k.dtype() ||
      k_cache.dtype() != v.dtype()) {
    throw std::runtime_error(
        "store_kv_cache: k_cache, v_cache, k, v must have the same dtype");
  }
  if (indices.dtype() != mx::int32) {
    throw std::runtime_error("store_kv_cache: indices must be int32");
  }

  const size_t num_pages = k_cache.shape(0);
  const size_t num_kv_heads = k_cache.shape(1);
  const size_t head_dim = k_cache.shape(2);
  const size_t length = indices.shape(0);

  if (head_dim % 4 != 0) {
    throw std::runtime_error(
        "store_kv_cache: head_dim must be a multiple of 4");
  }

  if (v_cache.shape(0) != num_pages || v_cache.shape(1) != num_kv_heads ||
      v_cache.shape(2) != head_dim) {
    throw std::runtime_error(
        "store_kv_cache: v_cache shape must match k_cache");
  }
  if (k.shape(0) != length || k.shape(1) != num_kv_heads ||
      k.shape(2) != head_dim || v.shape(0) != length ||
      v.shape(1) != num_kv_heads || v.shape(2) != head_dim) {
    throw std::runtime_error(
        "store_kv_cache: k and v shape must be (length, num_kv_heads, head_dim)");
  }

  const size_t kv_cache_stride = k_cache.strides()[0];
  const size_t kv_input_stride = k.strides()[0];
  const uint32_t n_chunks =
      static_cast<uint32_t>((num_kv_heads * head_dim) / 4);

  auto s = to_stream(s_);
  auto& d = mx::metal::device(s.device);
  auto& compute_encoder = mx::metal::get_command_encoder(s);

  std::string op_name = "store_kv_cache_" + type_to_name(k_cache);
  auto lib = d.get_library("mini_sglang_mlx_kernel", util::current_binary_dir());
  auto kernel = d.get_kernel(op_name, lib);

  const uint32_t length_u = static_cast<uint32_t>(length);
  const uint32_t kv_cache_stride_u = static_cast<uint32_t>(kv_cache_stride);
  const uint32_t kv_input_stride_u = static_cast<uint32_t>(kv_input_stride);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(k_cache, 0);
  compute_encoder.set_input_array(v_cache, 1);
  compute_encoder.set_input_array(indices, 2);
  compute_encoder.set_input_array(k, 3);
  compute_encoder.set_input_array(v, 4);
  compute_encoder.set_output_array(k_cache, 0);
  compute_encoder.set_output_array(v_cache, 1);
  compute_encoder.set_bytes(length_u, 5);
  compute_encoder.set_bytes(kv_cache_stride_u, 6);
  compute_encoder.set_bytes(kv_input_stride_u, 7);
  compute_encoder.set_bytes(n_chunks, 8);

  MTL::Size grid_dims(length_u, n_chunks, 1);
  MTL::Size group_dims(1, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
#else
  (void)k_cache;
  (void)v_cache;
  (void)indices;
  (void)k;
  (void)v;
  (void)s_;
  throw std::runtime_error("store_kv_cache: Metal backend required");
#endif
}

}  // namespace mini_sglang_mlx
