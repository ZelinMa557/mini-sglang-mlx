#pragma once

#include "mlx/ops.h"
#include <cstddef>

namespace mx = mlx::core;

namespace mini_sglang_mlx {

// Compare two 1D int arrays and return the length of the matching prefix.
// Both arrays must be 1D, contiguous, int32 or int64, and on CPU.
size_t fast_compare_key(const mx::array& a, const mx::array& b);

}  // namespace mini_sglang_mlx
