#include "fast_compare_key.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"
#include <algorithm>
#include <stdexcept>

namespace mlx_serve {

size_t fast_compare_key(const mx::array& a, const mx::array& b) {
  if (a.ndim() != 1 || b.ndim() != 1) {
    throw std::runtime_error(
        "fast_compare_key: Both arrays must be 1D.");
  }
  if (!a.flags().row_contiguous || !b.flags().row_contiguous) {
    throw std::runtime_error(
        "fast_compare_key: Both arrays must be contiguous.");
  }
  if (a.dtype() != b.dtype()) {
    throw std::runtime_error(
        "fast_compare_key: Both arrays must have the same dtype.");
  }
  if (a.dtype() != mx::int32 && a.dtype() != mx::int64) {
    throw std::runtime_error(
        "fast_compare_key: Arrays must be int32 or int64.");
  }

  const size_t common_len = std::min(a.size(), b.size());
  if (common_len == 0) {
    return 0;
  }

  if (a.dtype() == mx::int64) {
    const int64_t* a_ptr = a.data<int64_t>();
    const int64_t* b_ptr = b.data<int64_t>();
    const auto diff_pos =
        std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
    return static_cast<size_t>(diff_pos.first - a_ptr);
  } else {
    const int32_t* a_ptr = a.data<int32_t>();
    const int32_t* b_ptr = b.data<int32_t>();
    const auto diff_pos =
        std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
    return static_cast<size_t>(diff_pos.first - a_ptr);
  }
}

}  // namespace mlx_serve
