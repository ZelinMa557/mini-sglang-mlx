// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "moe_utils.h"
#include "store_kv_cache.h"
#include "fast_compare_key.h"
#include "gdn_state.h"
#include "paged_decode_attention.h"
#include "paged_prefill_attention.h"
namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Kernel library for mlx serve";
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
  m.def("paged_decode_attention", &mlx_serve::paged_decode_attention,
        "q"_a, "k_cache"_a, "v_cache"_a, "kv_indptr"_a, "kv_indices"_a,
        "num_kv_splits"_a, "sm_scale"_a, "max_kv_splits"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        nb::sig(
            "def paged_decode_attention(q: array, k_cache: array, v_cache: array, "
            "kv_indptr: array, kv_indices: array, num_kv_splits: array, "
            "sm_scale: float, max_kv_splits: int, "
            "*, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("paged_prefill_attention", &mlx_serve::paged_prefill_attention,
        "q"_a, "k_cache"_a, "v_cache"_a, "qo_indptr"_a, "kv_indptr"_a,
        "kv_indices"_a, "prefix_lens"_a, "sm_scale"_a, "max_len_extend"_a,
        nb::kw_only(), "is_cross_attention"_a = false,
        "sliding_window_size"_a = 0, "stream"_a = nb::none(),
        nb::sig(
            "def paged_prefill_attention(q: array, k_cache: array, v_cache: array, "
            "qo_indptr: array, kv_indptr: array, kv_indices: array, "
            "prefix_lens: array, sm_scale: float, max_len_extend: int, *, "
            "is_cross_attention: bool = False, sliding_window_size: int = 0, "
            "stream: Union[None, Stream, Device] = None) -> array"));
  m.def("gdn_state_inplace", &mlx_serve::gdn_state_inplace,
        "q"_a, "k"_a, "v"_a, "g"_a, "beta"_a, "state"_a, "slot_ids"_a,
        "qo_indptr"_a, "single_token_mode"_a = false, nb::kw_only(),
        "stream"_a = nb::none(),
        nb::sig(
            "def gdn_state_inplace(q: array, k: array, v: array, g: array, "
            "beta: array, state: array, slot_ids: array, qo_indptr: array, "
            "single_token_mode: bool = False, "
            "*, stream: Union[None, Stream, Device] = None) -> array"));
}
