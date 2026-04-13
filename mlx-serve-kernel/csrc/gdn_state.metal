#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

METAL_FUNC float sum4(float4 x) {
  return x.x + x.y + x.z + x.w;
}

[[kernel]] void gdn_decode_inplace_float32(
    device float* state,
    device bfloat16_t* y,
    const device bfloat16_t* q,
    const device bfloat16_t* k,
    const device bfloat16_t* v,
    const device bfloat16_t* g,
    const device bfloat16_t* beta,
    const device int32_t* slot_ids,
    constant uint& batch,
    constant uint& hk,
    constant uint& hv,
    constant uint& dk,
    constant uint& dv,
    constant uint& hk_per_hv,
    constant uint& state_slot_stride,
    constant uint& q_batch_stride,
    constant uint& v_batch_stride,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    threadgroup float* qk_pack_raw [[threadgroup(0)]]) {
  constexpr uint DV_TILE = 4;

  const uint lane = tid.x;
  const uint dv_idx = uint(tgpig.y) * DV_TILE + tid.y;
  const uint n = tgpig.z;
  if (lane >= 32 || tid.y >= DV_TILE || dv_idx >= dv || n >= batch * hv) {
    return;
  }

  const uint b_idx = n / hv;
  const uint hv_idx = n % hv;
  const uint hk_idx = hv_idx / hk_per_hv;
  const uint slot = static_cast<uint>(slot_ids[b_idx]);

  const device bfloat16_t* q_ptr =
      q + size_t(b_idx) * size_t(q_batch_stride) + size_t(hk_idx) * size_t(dk);
  const device bfloat16_t* k_ptr =
      k + size_t(b_idx) * size_t(q_batch_stride) + size_t(hk_idx) * size_t(dk);
  const device bfloat16_t* v_ptr =
      v + size_t(b_idx) * size_t(v_batch_stride) + size_t(hv_idx) * size_t(dv);
  const device bfloat16_t* g_ptr = g + size_t(b_idx) * size_t(hv) + size_t(hv_idx);
  const device bfloat16_t* beta_ptr =
      beta + size_t(b_idx) * size_t(hv) + size_t(hv_idx);
  device float* state_ptr =
      state + size_t(slot) * size_t(state_slot_stride) +
      size_t(hv_idx * dv + dv_idx) * size_t(dk);
  device float4* state4_ptr = reinterpret_cast<device float4*>(state_ptr);
  const uint dk_vec = dk / 4;
  threadgroup float4* q_pack4 = reinterpret_cast<threadgroup float4*>(qk_pack_raw);
  threadgroup float4* k_pack4 = q_pack4 + dk_vec;

  if (tid.y == 0) {
    for (uint c = lane; c < dk_vec; c += 32) {
      const uint base = c * 4;
      q_pack4[c] = float4(
          static_cast<float>(q_ptr[base + 0]),
          static_cast<float>(q_ptr[base + 1]),
          static_cast<float>(q_ptr[base + 2]),
          static_cast<float>(q_ptr[base + 3]));
      k_pack4[c] = float4(
          static_cast<float>(k_ptr[base + 0]),
          static_cast<float>(k_ptr[base + 1]),
          static_cast<float>(k_ptr[base + 2]),
          static_cast<float>(k_ptr[base + 3]));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float kv_partial = 0.0f;
  for (uint c = lane; c < dk_vec; c += 32) {
    const float4 s = state4_ptr[c] * static_cast<float>(g_ptr[0]);
    state4_ptr[c] = s;
    kv_partial += sum4(s * k_pack4[c]);
  }
  const float kv_mem = simd_sum(kv_partial);
  const float delta =
      (static_cast<float>(v_ptr[dv_idx]) - kv_mem) * static_cast<float>(beta_ptr[0]);

  float out_partial = 0.0f;
  for (uint c = lane; c < dk_vec; c += 32) {
    const float4 s = state4_ptr[c] + k_pack4[c] * delta;
    state4_ptr[c] = s;
    out_partial += sum4(s * q_pack4[c]);
  }
  const float out = simd_sum(out_partial);
  if (simd_lane == 0) {
    y[(size_t(b_idx) * size_t(hv) + size_t(hv_idx)) * size_t(dv) + size_t(dv_idx)] =
        static_cast<bfloat16_t>(out);
  }
}

[[kernel]] void gdn_prefill_inplace_float32(
    device float* state,
    device bfloat16_t* y,
    const device bfloat16_t* q,
    const device bfloat16_t* k,
    const device bfloat16_t* v,
    const device bfloat16_t* g,
    const device bfloat16_t* beta,
    const device float* state_in,
    const device int32_t* slot_ids,
    const device int32_t* qo_indptr,
    constant uint& batch,
    constant uint& hk,
    constant uint& hv,
    constant uint& dk,
    constant uint& dv,
    constant uint& hk_per_hv,
    constant uint& state_slot_stride,
    constant uint& q_token_stride,
    constant uint& v_token_stride,
    uint3 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    threadgroup float* local_state_raw [[threadgroup(0)]]) {
  const uint lane = gid.x;
  const uint dv_idx = gid.y;
  const uint n = gid.z;
  if (lane >= 32 || dv_idx >= dv || n >= batch * hv) {
    return;
  }

  const uint b_idx = n / hv;
  const uint hv_idx = n % hv;
  const uint hk_idx = hv_idx / hk_per_hv;
  const uint slot = static_cast<uint>(slot_ids[b_idx]);

  const int start = qo_indptr[b_idx];
  const int end = qo_indptr[b_idx + 1];

  const device float* state_in_ptr =
      state_in + size_t(slot) * size_t(state_slot_stride) +
      size_t(hv_idx * dv + dv_idx) * size_t(dk);
  device float* state_ptr =
      state + size_t(slot) * size_t(state_slot_stride) +
      size_t(hv_idx * dv + dv_idx) * size_t(dk);
  const uint dk_vec = dk / 4;
  const device float4* state_in4_ptr =
      reinterpret_cast<const device float4*>(state_in_ptr);
  device float4* state4_ptr = reinterpret_cast<device float4*>(state_ptr);
  threadgroup float4* local_state4 =
      reinterpret_cast<threadgroup float4*>(local_state_raw);

  for (uint c = lane; c < dk_vec; c += 32) {
    local_state4[c] = state_in4_ptr[c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int token = start; token < end; ++token) {
    const device bfloat16_t* q_ptr =
        q + size_t(token) * size_t(q_token_stride) + size_t(hk_idx) * size_t(dk);
    const device bfloat16_t* k_ptr =
        k + size_t(token) * size_t(q_token_stride) + size_t(hk_idx) * size_t(dk);
    const device bfloat16_t* v_ptr =
        v + size_t(token) * size_t(v_token_stride) + size_t(hv_idx) * size_t(dv);
    const float g_scalar =
        static_cast<float>(g[size_t(token) * size_t(hv) + size_t(hv_idx)]);
    const float beta_scalar =
        static_cast<float>(beta[size_t(token) * size_t(hv) + size_t(hv_idx)]);

    float kv_partial = 0.0f;
    for (uint c = lane; c < dk_vec; c += 32) {
      const uint base = c * 4;
      const float4 k4 = float4(
          static_cast<float>(k_ptr[base + 0]),
          static_cast<float>(k_ptr[base + 1]),
          static_cast<float>(k_ptr[base + 2]),
          static_cast<float>(k_ptr[base + 3]));
      const float4 s = local_state4[c] * g_scalar;
      local_state4[c] = s;
      kv_partial += sum4(s * k4);
    }
    const float kv_mem = simd_sum(kv_partial);
    const float delta =
        (static_cast<float>(v_ptr[dv_idx]) - kv_mem) * beta_scalar;

    float out_partial = 0.0f;
    for (uint c = lane; c < dk_vec; c += 32) {
      const uint base = c * 4;
      const float4 q4 = float4(
          static_cast<float>(q_ptr[base + 0]),
          static_cast<float>(q_ptr[base + 1]),
          static_cast<float>(q_ptr[base + 2]),
          static_cast<float>(q_ptr[base + 3]));
      const float4 k4 = float4(
          static_cast<float>(k_ptr[base + 0]),
          static_cast<float>(k_ptr[base + 1]),
          static_cast<float>(k_ptr[base + 2]),
          static_cast<float>(k_ptr[base + 3]));
      const float4 s = local_state4[c] + k4 * delta;
      local_state4[c] = s;
      out_partial += sum4(s * q4);
    }
    const float out = simd_sum(out_partial);
    if (simd_lane == 0) {
      y[(size_t(token) * size_t(hv) + size_t(hv_idx)) * size_t(dv) + size_t(dv_idx)] =
          static_cast<bfloat16_t>(out);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = lane; c < dk_vec; c += 32) {
    state4_ptr[c] = local_state4[c];
  }
}
