import numpy as np
import mlx.core as mx
import math
import time
from mlx_serve_kernel import paged_prefill_attention


def naive_paged_prefill_attention(
    q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices,
    prefix_lens, sm_scale, window_size=-1,
):
    """
    Naive reference implementation of paged prefill (extend) attention.

    q:            (total_q_tokens, num_q_heads, head_dim)
    k_cache:      (num_pages, num_kv_heads, head_dim)
    v_cache:      (num_pages, num_kv_heads, head_dim)
    qo_indptr:    (batch + 1,)
    kv_indptr:    (batch + 1,)
    kv_indices:   (total_kv,)
    prefix_lens:  (batch,)

    Causal mask: for KV positions in extend region (kv_pos >= prefix_len),
        q can attend to k only if q_offset >= k_extend_offset.
    For prefix region, no causal mask.

    Sliding window: kv_pos + window_size > q_abs,
        where q_abs = prefix_len + q_offset.
        Sink token (kv_pos == 0) always valid.
    """
    q_np = np.array(q, dtype=np.float32)
    k_np = np.array(k_cache, dtype=np.float32)
    v_np = np.array(v_cache, dtype=np.float32)
    qo_indptr_np = np.array(qo_indptr, dtype=np.int32)
    kv_indptr_np = np.array(kv_indptr, dtype=np.int32)
    kv_indices_np = np.array(kv_indices, dtype=np.int32)
    prefix_lens_np = np.array(prefix_lens, dtype=np.int32)

    total_q = q_np.shape[0]
    num_q_heads = q_np.shape[1]
    head_dim = q_np.shape[2]
    num_kv_heads = k_np.shape[1]
    kv_group_num = num_q_heads // num_kv_heads

    out_np = np.zeros_like(q_np)
    batch = len(qo_indptr_np) - 1

    for b in range(batch):
        q_start = qo_indptr_np[b]
        q_end = qo_indptr_np[b + 1]
        q_len = q_end - q_start

        kv_start = kv_indptr_np[b]
        kv_end = kv_indptr_np[b + 1]
        kv_len = kv_end - kv_start

        prefix_len = prefix_lens_np[b]

        if q_len == 0 or kv_len == 0:
            continue

        # Gather KV
        page_ids = kv_indices_np[kv_start:kv_end]
        k_gathered = k_np[page_ids]  # (kv_len, num_kv_heads, head_dim)
        v_gathered = v_np[page_ids]

        for qh in range(num_q_heads):
            kv_h = qh // kv_group_num

            for qi in range(q_len):
                q_vec = q_np[q_start + qi, qh, :]  # (head_dim,)
                ki = k_gathered[:, kv_h, :]         # (kv_len, head_dim)
                vi = v_gathered[:, kv_h, :]

                scores = ki @ q_vec * sm_scale  # (kv_len,)

                # Causal mask
                for kv_pos in range(kv_len):
                    if kv_pos >= prefix_len:
                        k_extend_offset = kv_pos - prefix_len
                        if qi < k_extend_offset:
                            scores[kv_pos] = -1e9

                # Sliding window mask
                if window_size > 0:
                    q_abs = prefix_len + qi
                    for kv_pos in range(kv_len):
                        if kv_pos == 0:
                            continue  # sink token always valid
                        if kv_pos + window_size <= q_abs:
                            scores[kv_pos] = -1e9

                # Softmax
                scores_max = np.max(scores)
                scores_exp = np.exp(scores - scores_max)
                scores_sum = np.sum(scores_exp)
                attn_weights = scores_exp / scores_sum

                out_np[q_start + qi, qh, :] = attn_weights @ vi

    return mx.array(out_np, dtype=q.dtype)


def test_correctness(
    batch,
    num_q_heads,
    num_kv_heads,
    head_dim,
    q_lens,
    prefix_lens_list,
    window_size=-1,
    dtype=mx.float16,
):
    """Test kernel correctness against naive implementation."""
    assert len(q_lens) == batch
    assert len(prefix_lens_list) == batch

    # KV len for each seq = prefix_len + q_len (extend = q_len)
    kv_lens = [p + q for p, q in zip(prefix_lens_list, q_lens)]
    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    num_pages = total_kv + 64
    max_len_extend = max(q_lens)

    # Generate data
    q = mx.random.normal((total_q, num_q_heads, head_dim), dtype=dtype)
    k_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)
    v_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)

    # Build qo_indptr
    qo_ind = [0]
    for ql in q_lens:
        qo_ind.append(qo_ind[-1] + ql)
    qo_indptr = mx.array(qo_ind, dtype=mx.int32)

    # Build kv_indptr
    kv_ind = [0]
    for kvl in kv_lens:
        kv_ind.append(kv_ind[-1] + kvl)
    kv_indptr = mx.array(kv_ind, dtype=mx.int32)

    # Random page mapping
    all_pages = np.random.permutation(num_pages)[:total_kv]
    kv_indices = mx.array(all_pages.astype(np.int32))

    prefix_lens = mx.array(prefix_lens_list, dtype=mx.int32)

    sm_scale = 1.0 / math.sqrt(head_dim)

    mx.eval(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens)

    # Kernel output
    out_kernel = paged_prefill_attention(
        q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens,
        sm_scale=sm_scale, max_len_extend=max_len_extend, window_size=window_size,
    )
    mx.eval(out_kernel)

    # Naive output
    out_naive = naive_paged_prefill_attention(
        q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens,
        sm_scale=sm_scale, window_size=window_size,
    )
    mx.eval(out_naive)

    # Compare
    out_k = np.array(out_kernel, dtype=np.float32)
    out_n = np.array(out_naive, dtype=np.float32)
    max_diff = np.max(np.abs(out_k - out_n))
    mean_diff = np.mean(np.abs(out_k - out_n))

    atol = 5e-2 if head_dim == 512 else 2e-2
    is_close = max_diff < atol

    return is_close, max_diff, mean_diff


def bench_kernel(
    batch,
    num_q_heads,
    num_kv_heads,
    head_dim,
    q_len,
    prefix_len,
    window_size=-1,
    dtype=mx.float16,
    warmup=10,
    repeat=100,
):
    """Benchmark kernel latency."""
    q_lens = [q_len] * batch
    prefix_lens_list = [prefix_len] * batch
    kv_lens = [prefix_len + q_len] * batch
    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    num_pages = total_kv + 64
    max_len_extend = q_len

    q = mx.random.normal((total_q, num_q_heads, head_dim), dtype=dtype)
    k_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)
    v_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)

    qo_ind = [0]
    for ql in q_lens:
        qo_ind.append(qo_ind[-1] + ql)
    qo_indptr = mx.array(qo_ind, dtype=mx.int32)

    kv_ind = [0]
    for kvl in kv_lens:
        kv_ind.append(kv_ind[-1] + kvl)
    kv_indptr = mx.array(kv_ind, dtype=mx.int32)

    kv_indices = mx.array(np.arange(total_kv, dtype=np.int32))
    prefix_lens_arr = mx.array(prefix_lens_list, dtype=mx.int32)

    sm_scale = 1.0 / math.sqrt(head_dim)
    mx.eval(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens_arr)

    # Warmup
    for _ in range(warmup):
        out = paged_prefill_attention(
            q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens_arr,
            sm_scale=sm_scale, max_len_extend=max_len_extend, window_size=window_size,
        )
        mx.eval(out)

    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        out = paged_prefill_attention(
            q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens_arr,
            sm_scale=sm_scale, max_len_extend=max_len_extend, window_size=window_size,
        )
        mx.eval(out)
    elapsed = (time.perf_counter() - start) / repeat * 1000  # ms

    return elapsed


if __name__ == "__main__":
    mx.random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Paged Prefill Attention - Correctness Tests")
    print("=" * 70)

    test_configs = [
        # (batch, nqh, nkvh, hd, q_lens, prefix_lens, window_size)
        # Basic: no prefix, no sliding window
        (1, 32, 8, 128, [16], [0], -1),
        (1, 32, 8, 128, [64], [0], -1),
        (1, 32, 8, 128, [128], [0], -1),
        # With prefix
        (1, 32, 8, 128, [32], [64], -1),
        (1, 32, 8, 128, [64], [128], -1),
        # Multiple sequences
        (2, 32, 8, 128, [32, 64], [0, 32], -1),
        (4, 32, 8, 128, [16, 32, 64, 8], [0, 16, 32, 64], -1),
        # Different head dims
        (1, 32, 8, 64, [32], [0], -1),
        (1, 8, 2, 512, [16], [0], -1),
        # MHA (kv_group_num=1)
        (1, 8, 8, 128, [32], [0], -1),
        # Large kv_group_num
        (1, 32, 4, 128, [32], [0], -1),
        # Sliding window
        (1, 32, 8, 128, [64], [128], 64),
        (2, 32, 8, 128, [32, 64], [64, 128], 32),
        # Sliding window, window > total len (effectively no window)
        (1, 32, 8, 128, [32], [0], 1024),
        # Short sequences
        (1, 32, 8, 128, [1], [0], -1),
        (1, 32, 8, 128, [1], [100], -1),
    ]

    all_pass = True
    for batch, nqh, nkvh, hd, qlens, plens, ws in test_configs:
        label = (f"batch={batch}, q_heads={nqh}, kv_heads={nkvh}, "
                 f"head_dim={hd}, q_lens={qlens}, prefix_lens={plens}, window={ws}")
        ok, max_diff, mean_diff = test_correctness(
            batch, nqh, nkvh, hd, qlens, plens, ws,
        )
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}  (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("All correctness tests passed!")
    else:
        print("Some tests FAILED!")

    print()
    print("=" * 70)
    print("Paged Prefill Attention - Benchmarks")
    print("=" * 70)

    bench_configs = [
        # (batch, nqh, nkvh, hd, q_len, prefix_len, window)
        (1, 32, 8, 128, 128, 0, -1),
        (1, 32, 8, 128, 512, 0, -1),
        (1, 32, 8, 128, 128, 512, -1),
        (4, 32, 8, 128, 128, 0, -1),
        (1, 32, 8, 128, 128, 512, 128),
    ]

    for batch, nqh, nkvh, hd, ql, pl, ws in bench_configs:
        elapsed = bench_kernel(batch, nqh, nkvh, hd, ql, pl, ws)
        label = (f"batch={batch}, q_heads={nqh}, kv_heads={nkvh}, "
                 f"head_dim={hd}, q_len={ql}, prefix_len={pl}, window={ws}")
        print(f"  {label}  ->  {elapsed:.3f} ms")
