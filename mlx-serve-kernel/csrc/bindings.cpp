// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "varlen_rope.hpp"
#include "fused_add_rmsnorm.h"
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
}