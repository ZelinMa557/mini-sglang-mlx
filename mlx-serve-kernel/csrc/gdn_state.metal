#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

METAL_FUNC float4 load_bfloat16x4(const device bfloat16_t* ptr) {
  return float4(
      static_cast<float>(ptr[0]),
      static_cast<float>(ptr[1]),
      static_cast<float>(ptr[2]),
      static_cast<float>(ptr[3]));
}

template <short DK, short DV>
[[kernel]] void gdn_state_inplace_float32(
    device float* state,
    device bfloat16_t* y,
    const device bfloat16_t* q,
    const device bfloat16_t* k,
    const device bfloat16_t* v,
    const device bfloat16_t* g,
    const device bfloat16_t* beta,
    const device int32_t* slot_ids,
    const device int32_t* qo_indptr,
    constant uint& batch,
    constant uint& hv,
    constant uint& dv,
    constant uint& hv_per_hk,
    constant uint& state_slot_stride,
    constant uint& q_token_stride,
    constant uint& v_token_stride,
    constant uint& single_token_mode,
    uint3 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  const uint dv_idx = gid.y;
  const uint n = gid.z;
  if (dv_idx >= dv || n >= batch * hv) {
    return;
  }

  const uint b_idx = n / hv;
  const uint hv_idx = n % hv;
  const uint hk_idx = hv_idx / hv_per_hk;
  const uint slot = static_cast<uint>(slot_ids[b_idx]);
  constexpr uint VEC_WIDTH = 4;
  constexpr uint N_PER_THREAD = DK / 32;
  static_assert(N_PER_THREAD == VEC_WIDTH);
  const uint base = simd_lane * VEC_WIDTH;
  device float* state_ptr =
      state + size_t(slot) * size_t(state_slot_stride) +
      size_t(hv_idx * DV + dv_idx) * size_t(DK);
  device float4* state4_ptr = reinterpret_cast<device float4*>(state_ptr);
  float4 state4 = state4_ptr[simd_lane];

  const int start = single_token_mode ? static_cast<int>(b_idx) : qo_indptr[b_idx];
  const int end = single_token_mode ? static_cast<int>(b_idx + 1) : qo_indptr[b_idx + 1];
  const size_t q_head_offset = size_t(hk_idx) * size_t(DK);
  const size_t v_head_offset = size_t(hv_idx) * size_t(DV) + size_t(dv_idx);

  const device bfloat16_t* q_ptr =
      q + size_t(start) * size_t(q_token_stride) + q_head_offset;
  const device bfloat16_t* k_ptr =
      k + size_t(start) * size_t(q_token_stride) + q_head_offset;
  const device bfloat16_t* v_ptr =
      v + size_t(start) * size_t(v_token_stride) + v_head_offset;
  const device bfloat16_t* g_ptr =
      g + size_t(start) * size_t(hv) + size_t(hv_idx);
  const device bfloat16_t* beta_ptr =
      beta + size_t(start) * size_t(hv) + size_t(hv_idx);
  device bfloat16_t* y_ptr =
      y + (size_t(start) * size_t(hv) + size_t(hv_idx)) * size_t(DV) + size_t(dv_idx);

  for (int token = start; token < end; ++token) {
    const float4 q4 = load_bfloat16x4(q_ptr + base);
    const float4 k4 = load_bfloat16x4(k_ptr + base);
    const float g_scalar = static_cast<float>(*g_ptr);
    const float beta_scalar = static_cast<float>(*beta_ptr);

    state4 *= g_scalar;
    const float kv_mem = simd_sum(dot(state4, k4));
    const float delta =
        (static_cast<float>(*v_ptr) - kv_mem) * beta_scalar;

    state4 = fma(k4, float4(delta), state4);
    const float out = simd_sum(dot(state4, q4));
    if (simd_lane == 0) {
      *y_ptr = static_cast<bfloat16_t>(out);
    }

    q_ptr += q_token_stride;
    k_ptr += q_token_stride;
    v_ptr += v_token_stride;
    g_ptr += hv;
    beta_ptr += hv;
    y_ptr += hv * DV;
  }

  state4_ptr[simd_lane] = state4;
}

#define instantiate_gdn_state(dk, dv) \
  instantiate_kernel( \
      "gdn_state_inplace_float32_dk" #dk "_dv" #dv, \
      gdn_state_inplace_float32, dk, dv)

instantiate_gdn_state(128, 128)
