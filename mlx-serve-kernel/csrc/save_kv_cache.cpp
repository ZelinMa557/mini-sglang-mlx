#include <cstdlib>
#include <omp.h>
namespace mlx_serve {
void save_kv_cache_impl(const uint16_t *kv, // NHD layout
                        uint16_t *cache,    // NHD layout
                        const int *kv_cache_indices, const int *seq_ids,
                        const int *position_ids, const int total_len,
                        const int head_num, const int head_dim,
                        const int kv_cache_indices_stride,
                        const int kv_stride_n,
                        const int kv_stride_h,
) {
  const size_t single_token_size = head_num * head_dim;
  #pragma omp parallel for if(total_len > 4)
  for (int i = 0; i < total_len; i++) {
    const uint16_t *src = kv + i * kv_stride_n;
    const int kv_cache_index =
        kv_cache_indices[seq_ids[i] * kv_cache_indices_stride +
                         position_ids[i]];
    const uint16_t *dst = cache + kv_cache_index * single_token_size;
    memcpy((char *)dst, (char *)src, single_token_size * sizeof(uint16_t));
  }
}
} // namespace mlx_serve