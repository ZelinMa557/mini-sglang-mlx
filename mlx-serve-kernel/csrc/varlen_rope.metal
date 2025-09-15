#include <metal_math>
#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void varlen_rope_single(
    const device T* in,
    const device int* positions,
    device T* out,
    constant float& base,
    constant const size_t& stride,
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  // Compute costheta, sintheta
  const float exponent = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  const float inv_freq = 1.0f / metal::pow(base, exponent);
  const float theta = static_cast<float>(positions[0]) * inv_freq;
  const float costheta = metal::fast::cos(theta);
  const float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices
  uint index_1, index_2;
  index_1 = pos.x + pos.y * stride;
  index_2 = index_1 + grid.x;

  // Read and write the output
  const float x1 = static_cast<float>(in[index_1]);
  const float x2 = static_cast<float>(in[index_2]);
  const float rx1 = x1 * costheta - x2 * sintheta;
  const float rx2 = x1 * sintheta + x2 * costheta;
  out[index_1] = static_cast<T>(rx1);
  out[index_2] = static_cast<T>(rx2);
}

template <typename T, int N = 4>
[[kernel]] void varlen_rope(
    const device T* in,
    const device int* positions,
    device T* out,
    constant float& base,
    constant const size_t strides[3],
    constant const size_t out_strides[3],
    constant const size_t& n_batch,
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // Compute costheta, sintheta
  // const float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  // const float inv_freq = metal::exp2(-d * base);
  const float exponent = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  const float inv_freq = 1.0f / metal::pow(base, exponent);
  const float theta = static_cast<float>(positions[pos.y]) * inv_freq;
  const float costheta = metal::fast::cos(theta);
  const float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices
  size_t out_index_1 = pos.x * out_strides[2] + pos.y * out_strides[1] +
        N * pos.z * out_strides[0];
  size_t out_index_2 = out_index_1 + grid.x * out_strides[2];
  size_t in_index_1 =
        pos.x * strides[2] + pos.y * strides[1] + N * pos.z * strides[0];
  size_t in_index_2 = in_index_1 + grid.x * strides[2];

  for (int i = 0; i < N && pos.z * N + i < n_batch; ++i) {
    // Read and write the output
    const float x1 = static_cast<float>(in[in_index_1]);
    const float x2 = static_cast<float>(in[in_index_2]);
    const float rx1 = x1 * costheta - x2 * sintheta;
    const float rx2 = x1 * sintheta + x2 * costheta;
    out[out_index_1] = static_cast<T>(rx1);
    out[out_index_2] = static_cast<T>(rx2);
    in_index_1 += strides[0];
    in_index_2 += strides[0];
    out_index_1 += out_strides[0];
    out_index_2 += out_strides[0];
  }
}

// clang-format off
#define instantiate_varlen_rope(type_name, type)                     \
  instantiate_kernel("varlen_rope_single_" #type_name, varlen_rope_single, type) \
  instantiate_kernel("varlen_rope_" #type_name, varlen_rope, type)   \

instantiate_varlen_rope(float16, half);
instantiate_varlen_rope(float32, float);
instantiate_varlen_rope(bfloat16, bfloat16_t);
// clang-format on