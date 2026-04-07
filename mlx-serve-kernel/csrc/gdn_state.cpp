#include "gdn_state.h"
#include "util.h"

#include "mlx/utils.h"

#include <stdexcept>
#include <string>

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_serve {

namespace {

void validate_common_inputs(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const mx::array& slot_ids,
    const char* op_name) {
  if (q.ndim() != 3 || k.ndim() != 3) {
    throw std::runtime_error(
        std::string(op_name) + ": q and k must be 3D");
  }
  if (v.ndim() != 3) {
    throw std::runtime_error(
        std::string(op_name) + ": v must be 3D");
  }
  if (g.ndim() != 2 || beta.ndim() != 2) {
    throw std::runtime_error(
        std::string(op_name) + ": g and beta must be 2D");
  }
  if (state.ndim() != 4) {
    throw std::runtime_error(
        std::string(op_name) + ": state must be 4D");
  }
  if (slot_ids.ndim() != 1 || slot_ids.dtype() != mx::int32) {
    throw std::runtime_error(
        std::string(op_name) + ": slot_ids must be int32 1D");
  }
  if (q.dtype() != mx::float32 || k.dtype() != mx::float32 ||
      v.dtype() != mx::float32 || g.dtype() != mx::float32 ||
      beta.dtype() != mx::float32 || state.dtype() != mx::float32) {
    throw std::runtime_error(
        std::string(op_name) + ": only float32 inputs are supported");
  }
  if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0) ||
      q.shape(0) != g.shape(0) || q.shape(0) != beta.shape(0)) {
    throw std::runtime_error(
        std::string(op_name) + ": leading dimensions must match");
  }
  if (q.shape(1) != k.shape(1) || q.shape(2) != k.shape(2)) {
    throw std::runtime_error(
        std::string(op_name) + ": q and k shapes must match");
  }
  if (v.shape(1) != g.shape(1) || v.shape(1) != beta.shape(1)) {
    throw std::runtime_error(
        std::string(op_name) + ": hv dimension mismatch");
  }
  if (state.shape(1) != v.shape(1) || state.shape(2) != v.shape(2) ||
      state.shape(3) != q.shape(2)) {
    throw std::runtime_error(
        std::string(op_name) + ": state shape must be (slots, hv, dv, dk)");
  }
  if (v.shape(1) % q.shape(1) != 0) {
    throw std::runtime_error(
        std::string(op_name) + ": hv must be a multiple of hk");
  }
  if (q.shape(2) % 8 != 0 || v.shape(2) % 8 != 0) {
    throw std::runtime_error(
        std::string(op_name) + ": dk and dv must be multiples of 8");
  }
}

}  // namespace

mx::array gdn_decode_inplace(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const mx::array& slot_ids,
    mx::StreamOrDevice s_) {
  validate_common_inputs(q, k, v, g, beta, state, slot_ids, "gdn_decode_inplace");

  const int batch = q.shape(0);
  const int hk = q.shape(1);
  const int dk = q.shape(2);
  const int hv = v.shape(1);
  const int dv = v.shape(2);

  if (slot_ids.shape(0) != batch) {
    throw std::runtime_error(
        "gdn_decode_inplace: slot_ids must have shape (batch,)");
  }

  auto s = to_stream(s_);
  return mx::array(
      {batch, hv, dv},
      mx::float32,
      std::make_shared<GDNDecodeInplace>(s, hk, hv, dk, dv),
      {q, k, v, g, beta, state, slot_ids});
}

mx::array gdn_prefill_inplace(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const mx::array& slot_ids,
    const mx::array& qo_indptr,
    mx::StreamOrDevice s_) {
  validate_common_inputs(q, k, v, g, beta, state, slot_ids, "gdn_prefill_inplace");

  if (qo_indptr.ndim() != 1 || qo_indptr.dtype() != mx::int32) {
    throw std::runtime_error(
        "gdn_prefill_inplace: qo_indptr must be int32 1D");
  }

  const int total_tokens = q.shape(0);
  const int batch = slot_ids.shape(0);
  const int hk = q.shape(1);
  const int dk = q.shape(2);
  const int hv = v.shape(1);
  const int dv = v.shape(2);

  if (qo_indptr.shape(0) != batch + 1) {
    throw std::runtime_error(
        "gdn_prefill_inplace: qo_indptr must have shape (batch + 1,)");
  }

  auto s = to_stream(s_);
  return mx::array(
      {total_tokens, hv, dv},
      mx::float32,
      std::make_shared<GDNPrefillInplace>(s, hk, hv, dk, dv),
      {q, k, v, g, beta, state, slot_ids, qo_indptr});
}

#ifdef _METAL_

void GDNDecodeInplace::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  constexpr uint32_t dv_tile_u = 4;

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const mx::array& q = inputs[0];
  const mx::array& k = inputs[1];
  const mx::array& v = inputs[2];
  const mx::array& g = inputs[3];
  const mx::array& beta = inputs[4];
  mx::array& state = const_cast<mx::array&>(inputs[5]);
  const mx::array& slot_ids = inputs[6];

  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());
  auto kernel = d.get_kernel("gdn_decode_inplace_float32", lib);
  auto& compute_encoder = d.get_command_encoder(s.index);

  const uint32_t batch_u = static_cast<uint32_t>(q.shape(0));
  const uint32_t hk_u = static_cast<uint32_t>(hk_);
  const uint32_t hv_u = static_cast<uint32_t>(hv_);
  const uint32_t dk_u = static_cast<uint32_t>(dk_);
  const uint32_t dv_u = static_cast<uint32_t>(dv_);
  const uint32_t hk_per_hv_u = static_cast<uint32_t>(hv_ / hk_);
  const uint32_t state_slot_stride_u = static_cast<uint32_t>(state.strides()[0]);
  const uint32_t q_batch_stride_u = static_cast<uint32_t>(q.strides()[0]);
  const uint32_t v_batch_stride_u = static_cast<uint32_t>(v.strides()[0]);

  out.set_data(mx::allocator::malloc(out.nbytes()));
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(state, 0);
  compute_encoder.set_output_array(state, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_input_array(q, 2);
  compute_encoder.set_input_array(k, 3);
  compute_encoder.set_input_array(v, 4);
  compute_encoder.set_input_array(g, 5);
  compute_encoder.set_input_array(beta, 6);
  compute_encoder.set_input_array(slot_ids, 7);
  compute_encoder.set_bytes(batch_u, 8);
  compute_encoder.set_bytes(hk_u, 9);
  compute_encoder.set_bytes(hv_u, 10);
  compute_encoder.set_bytes(dk_u, 11);
  compute_encoder.set_bytes(dv_u, 12);
  compute_encoder.set_bytes(hk_per_hv_u, 13);
  compute_encoder.set_bytes(state_slot_stride_u, 14);
  compute_encoder.set_bytes(q_batch_stride_u, 15);
  compute_encoder.set_bytes(v_batch_stride_u, 16);

  compute_encoder.set_threadgroup_memory_length(
      2 * dk_u * sizeof(float), 0);
  MTL::Size grid_dims(1, (dv_u + dv_tile_u - 1) / dv_tile_u, batch_u * hv_u);
  MTL::Size group_dims(32, dv_tile_u, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void GDNPrefillInplace::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const mx::array& q = inputs[0];
  const mx::array& k = inputs[1];
  const mx::array& v = inputs[2];
  const mx::array& g = inputs[3];
  const mx::array& beta = inputs[4];
  mx::array& state = const_cast<mx::array&>(inputs[5]);
  const mx::array& slot_ids = inputs[6];
  const mx::array& qo_indptr = inputs[7];

  auto lib = d.get_library("mlx_serve_kernel", util::current_binary_dir());
  auto kernel = d.get_kernel("gdn_prefill_inplace_float32", lib);
  auto& compute_encoder = d.get_command_encoder(s.index);

  const uint32_t batch_u = static_cast<uint32_t>(slot_ids.shape(0));
  const uint32_t hk_u = static_cast<uint32_t>(hk_);
  const uint32_t hv_u = static_cast<uint32_t>(hv_);
  const uint32_t dk_u = static_cast<uint32_t>(dk_);
  const uint32_t dv_u = static_cast<uint32_t>(dv_);
  const uint32_t hk_per_hv_u = static_cast<uint32_t>(hv_ / hk_);
  const uint32_t state_slot_stride_u = static_cast<uint32_t>(state.strides()[0]);
  const uint32_t q_token_stride_u = static_cast<uint32_t>(q.strides()[0]);
  const uint32_t v_token_stride_u = static_cast<uint32_t>(v.strides()[0]);

  out.set_data(mx::allocator::malloc(out.nbytes()));
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(state, 0);
  compute_encoder.set_output_array(state, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_input_array(q, 2);
  compute_encoder.set_input_array(k, 3);
  compute_encoder.set_input_array(v, 4);
  compute_encoder.set_input_array(g, 5);
  compute_encoder.set_input_array(beta, 6);
  compute_encoder.set_input_array(state, 7);
  compute_encoder.set_input_array(slot_ids, 8);
  compute_encoder.set_input_array(qo_indptr, 9);
  compute_encoder.set_bytes(batch_u, 10);
  compute_encoder.set_bytes(hk_u, 11);
  compute_encoder.set_bytes(hv_u, 12);
  compute_encoder.set_bytes(dk_u, 13);
  compute_encoder.set_bytes(dv_u, 14);
  compute_encoder.set_bytes(hk_per_hv_u, 15);
  compute_encoder.set_bytes(state_slot_stride_u, 16);
  compute_encoder.set_bytes(q_token_stride_u, 17);
  compute_encoder.set_bytes(v_token_stride_u, 18);

  compute_encoder.set_threadgroup_memory_length(dk_u * sizeof(float), 0);
  MTL::Size grid_dims(32, dv_u, batch_u * hv_u);
  MTL::Size group_dims(32, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

#else

void GDNDecodeInplace::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  (void)inputs;
  (void)outputs;
  throw std::runtime_error("GDNDecodeInplace: Metal backend required");
}

void GDNPrefillInplace::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  (void)inputs;
  (void)outputs;
  throw std::runtime_error("GDNPrefillInplace: Metal backend required");
}

#endif

}  // namespace mlx_serve
