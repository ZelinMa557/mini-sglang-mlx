#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

// Scatter k[r], v[r] into k_cache[indices[r]], v_cache[indices[r]] (in-place).
// Each thread copies 4 elements. head_dim is a multiple of 4.
// Grid: (length, n_chunks) with n_chunks = (num_kv_heads * head_dim) / 4.
template <typename T>
[[kernel]] void store_kv_cache(
    device T* k_cache,
    device T* v_cache,
    const device int32_t* indices,
    const device T* k,
    const device T* v,
    constant uint& length,
    constant uint& kv_cache_stride,
    constant uint& kv_input_stride,
    constant uint& n_chunks,
    uint2 gid [[thread_position_in_grid]]) {
  const uint r = gid.x;
  const uint c = gid.y;
  if (r >= length || c >= n_chunks) {
    return;
  }
  const uint pos = indices[r];
  const size_t src_base = size_t(r) * size_t(kv_input_stride) + size_t(c) * 4;
  const size_t dst_base = size_t(pos) * size_t(kv_cache_stride) + size_t(c) * 4;
  for (uint i = 0; i < 4; i++) {
    k_cache[dst_base + i] = k[src_base + i];
    v_cache[dst_base + i] = v[src_base + i];
  }
}

// clang-format off
#define instantiate_store_kv_cache(type_name, type) \
  instantiate_kernel("store_kv_cache_" #type_name, store_kv_cache, type)

instantiate_store_kv_cache(float16, half);
instantiate_store_kv_cache(bfloat16, bfloat16_t);
// clang-format on
