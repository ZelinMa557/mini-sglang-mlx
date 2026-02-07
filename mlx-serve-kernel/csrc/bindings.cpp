// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "varlen_rope.hpp"
#include "fused_add_rmsnorm.h"
#include "moe_sum_reduce.h"
#include "store_kv_cache.h"
#include "fast_compare_key.h"
namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Kernel library for mlx serve";
  m.def("fused_add_rmsnorm", &mlx_serve::fused_add_rmsnorm, "x"_a, "y"_a,
        "weight"_a, "eps"_a, nb::kw_only(), "stream"_a = nb::none(),
        nb::sig(
            "def fused_add_rmsnorm(x: array, y: array, weight: array, eps: "
            "float, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def(
      "varlen_rope", &mlx_serve::varlen_rope, "x"_a, "positions"_a, "dims"_a,
      "base"_a, nb::kw_only(), "stream"_a = nb::none(),
      nb::sig("def varlen_rope(x: array, positions: array, "
              "dims:int, base:float, *, stream: Union[None, Stream, Device] = "
              "None) -> array"));
  m.def("moe_sum_reduce", &mlx_serve::moe_sum_reduce, "y"_a, "scores"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        nb::sig(
            "def moe_sum_reduce(y: array, scores: array, *, "
            "stream: Union[None, Stream, Device] = None) -> array"));
  m.def("moe_sum_reduce_with_reorder", &mlx_serve::moe_sum_reduce_with_reorder,
        "y"_a, "scores"_a, "inv_order"_a, nb::kw_only(), "stream"_a = nb::none(),
        nb::sig(
            "def moe_sum_reduce_with_reorder(y: array, scores: array, "
            "inv_order: array, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("store_kv_cache", &mlx_serve::store_kv_cache,
        "k_cache"_a, "v_cache"_a, "indices"_a, "k"_a, "v"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        nb::sig(
            "def store_kv_cache(k_cache: array, v_cache: array, indices: array, "
            "k: array, v: array, *, stream: Union[None, Stream, Device] = None) -> None"));
  m.def("fast_compare_key", &mlx_serve::fast_compare_key,
        "a"_a, "b"_a,
        nb::sig("def fast_compare_key(a: array, b: array) -> int"));
}