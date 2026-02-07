#pragma once

#include "mlx/ops.h"
#include <cstddef>

namespace mx = mlx::core;

namespace mlx_serve {

// Scatter k[r], v[r] into k_cache[indices[r]], v_cache[indices[r]] (in-place).
// k_cache, v_cache: shape (num_pages, num_kv_heads, head_dim).
// indices: shape (length,) int32.
// k, v: shape (length, num_kv_heads, head_dim); same stride allowed.
// head_dim must be a multiple of 4.
void store_kv_cache(
    mx::array& k_cache,
    mx::array& v_cache,
    const mx::array& indices,
    const mx::array& k,
    const mx::array& v,
    mx::StreamOrDevice s = {});

}  // namespace mlx_serve
