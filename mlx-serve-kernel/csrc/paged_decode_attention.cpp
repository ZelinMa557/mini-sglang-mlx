#include "paged_decode_attention.h"
#include "util.h"

#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx_serve {

mx::array paged_decode_attention(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& kv_indptr,
    const mx::array& kv_indices,
    const mx::array& num_kv_splits,
    float sm_scale,
    int max_kv_splits,
    mx::StreamOrDevice s_) {
  if (q.ndim() != 3) {
    throw std::runtime_error(
        "paged_decode_attention: q must be 3D (batch, num_q_heads, head_dim)");
  }
  if (k_cache.ndim() != 3 || v_cache.ndim() != 3) {
    throw std::runtime_error(
        "paged_decode_attention: k_cache and v_cache must be 3D "
        "(num_pages, num_kv_heads, head_dim)");
  }
  if (kv_indptr.ndim() != 1 || kv_indices.ndim() != 1 ||
      num_kv_splits.ndim() != 1) {
    throw std::runtime_error(
        "paged_decode_attention: kv_indptr, kv_indices, num_kv_splits must be "
        "1D");
  }

  int batch = q.shape(0);
  int num_q_heads = q.shape(1);
  int head_dim = q.shape(2);
  int num_kv_heads = k_cache.shape(1);

  if (head_dim != 128 && head_dim != 256) {
    throw std::runtime_error(
        "paged_decode_attention: head_dim must be 128 or 256");
  }
  if (num_q_heads % num_kv_heads != 0) {
    throw std::runtime_error(
        "paged_decode_attention: num_q_heads must be a multiple of "
        "num_kv_heads");
  }
  if (num_q_heads / num_kv_heads > 16) {
    throw std::runtime_error(
        "paged_decode_attention: only q_heads/kv_heads <= 16 is supported");
  }
  if (k_cache.shape(2) != head_dim || v_cache.shape(2) != head_dim) {
    throw std::runtime_error(
        "paged_decode_attention: cache head_dim must match q head_dim");
  }

  auto s = to_stream(s_);

  return mx::array(
      {batch, num_q_heads, head_dim},
      q.dtype(),
      std::make_shared<PagedDecodeAttention>(
          s, sm_scale, max_kv_splits, num_q_heads, num_kv_heads,
          head_dim),
      {q, k_cache, v_cache, kv_indptr, kv_indices, num_kv_splits});
}

#ifdef _METAL_

namespace {

std::string dtype_to_str(const mx::array& a) {
  switch (a.dtype()) {
    case mx::float16:
      return "float16";
    case mx::bfloat16:
      return "bfloat16";
    default:
      throw std::runtime_error(
          "paged_decode_attention: unsupported dtype, must be float16 or "
          "bfloat16");
  }
}

}  // namespace

void PagedDecodeAttention::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const mx::array& q = inputs[0];
  const mx::array& k_cache = inputs[1];
  const mx::array& v_cache = inputs[2];
  const mx::array& kv_indptr = inputs[3];
  const mx::array& kv_indices = inputs[4];
  const mx::array& num_kv_splits_arr = inputs[5];

  int batch = q.shape(0);
  std::string tname = dtype_to_str(q);

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());

  // Allocate intermediate buffers
  // att_out: (batch, num_q_heads, max_kv_splits, head_dim) float32
  // att_lse: (batch, num_q_heads, max_kv_splits) float32
  size_t att_out_size =
      (size_t)batch * num_q_heads_ * max_kv_splits_ * head_dim_;
  size_t att_lse_size = (size_t)batch * num_q_heads_ * max_kv_splits_;

  mx::array att_out(
      {batch, num_q_heads_, max_kv_splits_, head_dim_}, mx::float32, nullptr,
      {});
  att_out.set_data(mx::allocator::malloc(att_out_size * sizeof(float)));

  mx::array att_lse(
      {batch, num_q_heads_, max_kv_splits_}, mx::float32, nullptr, {});
  att_lse.set_data(mx::allocator::malloc(att_lse_size * sizeof(float)));

  // ---- Stage 1: Partial attention ----
  {
    constexpr int block_h = 16;

    std::string stage1_name = "paged_decode_attention_stage1_" + tname +
                              "_dk" + std::to_string(head_dim_) + "_dv" +
                              std::to_string(head_dim_) + "_bh" +
                              std::to_string(block_h);
    auto kernel = d.get_kernel(stage1_name, lib);

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k_cache, 1);
    compute_encoder.set_input_array(v_cache, 2);
    compute_encoder.set_input_array(kv_indptr, 3);
    compute_encoder.set_input_array(kv_indices, 4);
    compute_encoder.set_input_array(num_kv_splits_arr, 5);
    compute_encoder.set_output_array(att_out, 6);
    compute_encoder.set_output_array(att_lse, 7);
    compute_encoder.set_bytes(sm_scale_, 8);
    compute_encoder.set_bytes(num_q_heads_, 9);
    compute_encoder.set_bytes(num_kv_heads_, 10);
    compute_encoder.set_bytes(max_kv_splits_, 11);

    MTL::Size grid_dims(batch, num_kv_heads_, max_kv_splits_);
    MTL::Size group_dims(32, 1, 1);

    compute_encoder.set_threadgroup_memory_length(0, 0);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // ---- Stage 2: Reduction ----
  {
    std::string stage2_name = "paged_decode_attention_stage2_" + tname +
                              "_dv" + std::to_string(head_dim_);
    auto kernel = d.get_kernel(stage2_name, lib);

    out.set_data(mx::allocator::malloc(out.nbytes()));

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(att_out, 0);
    compute_encoder.set_input_array(att_lse, 1);
    compute_encoder.set_output_array(out, 2);
    compute_encoder.set_input_array(kv_indptr, 3);
    compute_encoder.set_input_array(num_kv_splits_arr, 4);
    compute_encoder.set_bytes(num_q_heads_, 5);
    compute_encoder.set_bytes(max_kv_splits_, 6);

    MTL::Size grid_dims(batch, num_q_heads_, 1);
    MTL::Size group_dims(32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Free intermediate buffers
  d.add_temporary(att_out, s.index);
  d.add_temporary(att_lse, s.index);
}

#endif

}  // namespace mlx_serve
