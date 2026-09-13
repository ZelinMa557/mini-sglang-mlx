// Minimal subset of mlx/backend/metal/kernels/utils.h inlined into
// runtime-JIT-compiled kernel sources (see tools/gen_jit_source.py).
// The runtime compiler (MTL::newLibrary) resolves system headers
// (<metal_stdlib>, MetalPerformancePrimitives) natively but cannot take
// -I include paths, so the small MLX helper headers are inlined here.

#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
typedef half float16_t;
