#include "paged_prefill_attention.h"
#include "util.h"

#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx_serve {

mx::array paged_prefill_attention(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& qo_indptr,
    const mx::array& kv_indptr,
    const mx::array& kv_indices,
    const mx::array& prefix_lens,
    float sm_scale,
    int max_len_extend,
    bool is_cross_attention,
    int sliding_window_size,
    mx::StreamOrDevice s_) {
  if (q.ndim() != 3) {
    throw std::runtime_error(
        "paged_prefill_attention: q must be 3D "
        "(total_q_tokens, num_q_heads, head_dim)");
  }
  if (k_cache.ndim() != 3 || v_cache.ndim() != 3) {
    throw std::runtime_error(
        "paged_prefill_attention: k_cache and v_cache must be 3D "
        "(num_pages, num_kv_heads, head_dim)");
  }
  if (qo_indptr.ndim() != 1 || kv_indptr.ndim() != 1 ||
      kv_indices.ndim() != 1 || prefix_lens.ndim() != 1) {
    throw std::runtime_error(
        "paged_prefill_attention: qo_indptr, kv_indptr, kv_indices, "
        "prefix_lens must be 1D");
  }

  int total_q_tokens = q.shape(0);
  int num_q_heads = q.shape(1);
  int head_dim = q.shape(2);
  int num_kv_heads = k_cache.shape(1);

  if (head_dim != 128 && head_dim != 256) {
    throw std::runtime_error(
        "paged_prefill_attention: head_dim must be 128 or 256");
  }
  if (num_q_heads % num_kv_heads != 0) {
    throw std::runtime_error(
        "paged_prefill_attention: num_q_heads must be a multiple of "
        "num_kv_heads");
  }
  if (k_cache.shape(2) != head_dim || v_cache.shape(2) != head_dim) {
    throw std::runtime_error(
        "paged_prefill_attention: cache head_dim must match q head_dim");
  }
  if (sliding_window_size < 0) {
    throw std::runtime_error(
        "paged_prefill_attention: sliding_window_size must be >= 0");
  }

  // batch = qo_indptr.size - 1
  int batch = qo_indptr.shape(0) - 1;
  if (batch <= 0) {
    throw std::runtime_error(
        "paged_prefill_attention: batch must be > 0");
  }

  auto s = to_stream(s_);

  return mx::array(
      {total_q_tokens, num_q_heads, head_dim},
      q.dtype(),
      std::make_shared<PagedPrefillAttention>(
          s, sm_scale, max_len_extend, num_q_heads, num_kv_heads,
          head_dim, total_q_tokens, is_cross_attention, sliding_window_size),
      {q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens});
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
          "paged_prefill_attention: unsupported dtype, must be float16 or "
          "bfloat16");
  }
}

}  // namespace

void PagedPrefillAttention::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const mx::array& q = inputs[0];
  const mx::array& k_cache = inputs[1];
  const mx::array& v_cache = inputs[2];
  const mx::array& qo_indptr = inputs[3];
  const mx::array& kv_indptr = inputs[4];
  const mx::array& kv_indices = inputs[5];
  const mx::array& prefix_lens = inputs[6];

  int batch = qo_indptr.shape(0) - 1;
  std::string tname = dtype_to_str(q);

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());

  std::string kernel_name = "paged_prefill_attention_" + tname + "_dk" +
                            std::to_string(head_dim_) + "_dv" +
                            std::to_string(head_dim_);
  auto kernel = d.get_kernel(kernel_name, lib);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_input_array(k_cache, 2);
  compute_encoder.set_input_array(v_cache, 3);
  compute_encoder.set_input_array(qo_indptr, 4);
  compute_encoder.set_input_array(kv_indptr, 5);
  compute_encoder.set_input_array(kv_indices, 6);
  compute_encoder.set_input_array(prefix_lens, 7);
  compute_encoder.set_bytes(sm_scale_, 8);
  compute_encoder.set_bytes(num_q_heads_, 9);
  compute_encoder.set_bytes(num_kv_heads_, 10);
  compute_encoder.set_bytes(is_cross_attention_, 11);
  compute_encoder.set_bytes(sliding_window_size_, 12);

  constexpr int BLOCK_M = 64;
  int num_q_blocks = (max_len_extend_ + BLOCK_M - 1) / BLOCK_M;

  MTL::Size grid_dims(batch, num_q_heads_, num_q_blocks);
  MTL::Size group_dims(32, 4, 1);

  compute_encoder.set_threadgroup_memory_length(0, 0);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#endif

}  // namespace mlx_serve
