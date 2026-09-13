import math

import mlx.core as mx
import numpy as np

from mini_sglang_mlx_kernel import paged_prefill_attention, store_kv_cache


HEAD_DIM = 128


def _build_uniform_score_case(q_len, prefix_len, dtype=mx.bfloat16):
    kv_len = prefix_len + q_len
    q = mx.zeros((q_len, 1, HEAD_DIM), dtype=dtype)
    k_cache = mx.zeros((kv_len, 1, HEAD_DIM), dtype=dtype)

    v_np = np.zeros((kv_len, 1, HEAD_DIM), dtype=np.float32)
    for pos in range(kv_len):
        v_np[pos, 0, pos] = 1.0
    v_cache = mx.array(v_np, dtype=dtype)

    qo_indptr = mx.array([0, q_len], dtype=mx.int32)
    kv_indptr = mx.array([0, kv_len], dtype=mx.int32)
    kv_indices = mx.array(np.arange(kv_len, dtype=np.int32))
    prefix_lens = mx.array([prefix_len], dtype=mx.int32)
    mx.eval(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens)
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "prefix_lens": prefix_lens,
        "q_len": q_len,
        "kv_len": kv_len,
        "prefix_len": prefix_len,
    }


def _run_kernel(data, *, is_cross_attention=False, sliding_window_size=0):
    out = paged_prefill_attention(
        data["q"],
        data["k_cache"],
        data["v_cache"],
        data["qo_indptr"],
        data["kv_indptr"],
        data["kv_indices"],
        data["prefix_lens"],
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        max_len_extend=data["q_len"],
        is_cross_attention=is_cross_attention,
        sliding_window_size=sliding_window_size,
    )
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def _expected_weights(q_len, prefix_len, *, is_cross_attention=False, sliding_window_size=0):
    kv_len = prefix_len + q_len
    weights = np.zeros((q_len, kv_len), dtype=np.float32)
    for p in range(q_len):
        pos = prefix_len + p
        attend = []
        for kpos in range(kv_len):
            in_window = sliding_window_size == 0 or kpos >= pos - sliding_window_size + 1
            in_causal = is_cross_attention or kpos <= pos
            if in_window and in_causal:
                attend.append(kpos)
        weights[p, attend] = 1.0 / len(attend)
    return weights


def _assert_attention_weights(
    q_len,
    prefix_len,
    *,
    is_cross_attention=False,
    sliding_window_size=0,
):
    data = _build_uniform_score_case(q_len, prefix_len)
    out = _run_kernel(
        data,
        is_cross_attention=is_cross_attention,
        sliding_window_size=sliding_window_size,
    )
    actual = out[:, 0, : data["kv_len"]]
    expected = _expected_weights(
        q_len,
        prefix_len,
        is_cross_attention=is_cross_attention,
        sliding_window_size=sliding_window_size,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def test_default_prefill_remains_causal():
    _assert_attention_weights(4, 0)


def test_full_attention_is_unmasked():
    _assert_attention_weights(4, 0, is_cross_attention=True)


def test_sliding_causal_window():
    _assert_attention_weights(4, 0, sliding_window_size=2)


def test_sliding_causal_window_crosses_prefix_boundary():
    _assert_attention_weights(4, 4, sliding_window_size=5)


def test_bidirectional_windowed_over_prefix():
    # DFlash2 layers: bidirectional inside the proposal block (DFlash
    # block-diffusion) while still windowed over the cached context.
    _assert_attention_weights(4, 4, is_cross_attention=True, sliding_window_size=5)


def test_bidirectional_window_clips_prefix_only():
    # The window bounds how far back the context reaches but never
    # clips the block itself (a block is at most block_size tokens).
    _assert_attention_weights(2, 6, is_cross_attention=True, sliding_window_size=3)


def test_store_kv_cache_writes_are_reachable_by_prefill():
    q_len = 4
    page_ids = mx.array([3, 1, 4, 2], dtype=mx.int32)
    q = mx.zeros((q_len, 1, HEAD_DIM), dtype=mx.bfloat16)
    k = mx.zeros((q_len, 1, HEAD_DIM), dtype=mx.bfloat16)

    v_np = np.zeros((q_len, 1, HEAD_DIM), dtype=np.float32)
    for pos in range(q_len):
        v_np[pos, 0, pos] = 1.0
    v = mx.array(v_np, dtype=mx.bfloat16)

    k_cache = mx.zeros((5, 1, HEAD_DIM), dtype=mx.bfloat16)
    v_cache = mx.zeros((5, 1, HEAD_DIM), dtype=mx.bfloat16)
    mx.eval(q, k, v, k_cache, v_cache, page_ids)
    store_kv_cache(k_cache, v_cache, page_ids, k, v)

    data = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": mx.array([0, q_len], dtype=mx.int32),
        "kv_indptr": mx.array([0, q_len], dtype=mx.int32),
        "kv_indices": page_ids,
        "prefix_lens": mx.array([0], dtype=mx.int32),
        "q_len": q_len,
        "kv_len": q_len,
        "prefix_len": 0,
    }
    mx.eval(data["qo_indptr"], data["kv_indptr"], data["prefix_lens"])

    out = _run_kernel(data)
    actual = out[:, 0, :q_len]
    expected = _expected_weights(q_len, 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
