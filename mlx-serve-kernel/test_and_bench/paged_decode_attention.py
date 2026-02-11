import numpy as np
import mlx.core as mx
import math
import time
from mlx_serve_kernel import paged_decode_attention


def naive_paged_attention(
    q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, window_size=-1
):
    """
    Naive reference implementation of paged decode attention.
    q: (batch, num_q_heads, head_dim)
    k_cache, v_cache: (num_pages, num_kv_heads, head_dim)
    kv_indptr: (batch + 1,)
    kv_indices: (total_kv_tokens,)
    """
    batch = q.shape[0]
    num_q_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k_cache.shape[1]
    kv_group_num = num_q_heads // num_kv_heads

    q_np = np.array(q, dtype=np.float32)
    k_np = np.array(k_cache, dtype=np.float32)
    v_np = np.array(v_cache, dtype=np.float32)
    indptr_np = np.array(kv_indptr, dtype=np.int32)
    indices_np = np.array(kv_indices, dtype=np.int32)

    out_np = np.zeros((batch, num_q_heads, head_dim), dtype=np.float32)

    for b in range(batch):
        kv_start = indptr_np[b]
        kv_end = indptr_np[b + 1]
        kv_len = kv_end - kv_start

        if kv_len == 0:
            continue

        # Gather K, V: (kv_len, num_kv_heads, head_dim)
        page_ids = indices_np[kv_start:kv_end]
        k_gathered = k_np[page_ids]  # (kv_len, num_kv_heads, head_dim)
        v_gathered = v_np[page_ids]

        for qh in range(num_q_heads):
            kv_h = qh // kv_group_num

            qi = q_np[b, qh, :]  # (head_dim,)
            ki = k_gathered[:, kv_h, :]  # (kv_len, head_dim)
            vi = v_gathered[:, kv_h, :]  # (kv_len, head_dim)

            # QK^T
            scores = ki @ qi * sm_scale  # (kv_len,)

            # Sliding window mask
            if window_size > 0:
                win_start = max(1, kv_len - window_size)
                for pos in range(kv_len):
                    if pos != 0 and pos < win_start:
                        scores[pos] = -1e9

            # Softmax
            scores_max = np.max(scores)
            scores_exp = np.exp(scores - scores_max)
            scores_sum = np.sum(scores_exp)
            attn_weights = scores_exp / scores_sum

            # Output
            out_np[b, qh, :] = attn_weights @ vi

    return mx.array(out_np, dtype=q.dtype)


def test_correctness(
    batch,
    num_q_heads,
    num_kv_heads,
    head_dim,
    kv_lens,
    max_kv_splits=8,
    window_size=-1,
    dtype=mx.float16,
):
    """Test kernel correctness against naive implementation."""
    assert len(kv_lens) == batch

    num_pages = sum(kv_lens) + 64  # extra pages
    total_kv = sum(kv_lens)

    # Generate random data
    q = mx.random.normal((batch, num_q_heads, head_dim), dtype=dtype)
    k_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)
    v_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)

    # Build kv_indptr and kv_indices
    indptr = [0]
    for l in kv_lens:
        indptr.append(indptr[-1] + l)
    kv_indptr = mx.array(indptr, dtype=mx.int32)

    # Random page mapping (simulate paged allocation)
    all_pages = np.random.permutation(num_pages)[:total_kv]
    kv_indices = mx.array(all_pages.astype(np.int32))

    # Compute num_kv_splits per request
    splits = []
    for l in kv_lens:
        s = min(max_kv_splits, max(1, (l + 255) // 256))
        splits.append(s)
    num_kv_splits = mx.array(splits, dtype=mx.int32)

    mx.eval(q, k_cache, v_cache, kv_indptr, kv_indices, num_kv_splits)

    # Kernel output
    out_kernel = paged_decode_attention(
        q, k_cache, v_cache, kv_indptr, kv_indices, num_kv_splits,
        sm_scale=1.0 / math.sqrt(head_dim),
        max_kv_splits=max_kv_splits,
        window_size=window_size,
    )
    mx.eval(out_kernel)

    # Naive output
    out_naive = naive_paged_attention(
        q, k_cache, v_cache, kv_indptr, kv_indices,
        sm_scale=1.0 / math.sqrt(head_dim),
        window_size=window_size,
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
    kv_len,
    max_kv_splits=16,
    window_size=-1,
    dtype=mx.float16,
    warmup=10,
    repeat=100,
):
    """Benchmark kernel latency."""
    kv_lens = [kv_len] * batch
    num_pages = sum(kv_lens) + 64
    total_kv = sum(kv_lens)

    q = mx.random.normal((batch, num_q_heads, head_dim), dtype=dtype)
    k_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)
    v_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)

    indptr = [0]
    for l in kv_lens:
        indptr.append(indptr[-1] + l)
    kv_indptr = mx.array(indptr, dtype=mx.int32)

    all_pages = np.arange(total_kv)
    kv_indices = mx.array(all_pages.astype(np.int32))

    splits = []
    for l in kv_lens:
        s = min(max_kv_splits, max(1, (l + 255) // 256))
        splits.append(s)
    num_kv_splits = mx.array(splits, dtype=mx.int32)

    sm_scale = 1.0 / math.sqrt(head_dim)
    mx.eval(q, k_cache, v_cache, kv_indptr, kv_indices, num_kv_splits)

    # Warmup
    for _ in range(warmup):
        out = paged_decode_attention(
            q, k_cache, v_cache, kv_indptr, kv_indices, num_kv_splits,
            sm_scale=sm_scale, max_kv_splits=max_kv_splits, window_size=window_size,
        )
        mx.eval(out)

    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        out = paged_decode_attention(
            q, k_cache, v_cache, kv_indptr, kv_indices, num_kv_splits,
            sm_scale=sm_scale, max_kv_splits=max_kv_splits, window_size=window_size,
        )
        mx.eval(out)
    elapsed = (time.perf_counter() - start) / repeat * 1000  # ms

    return elapsed


if __name__ == "__main__":
    mx.random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Paged Decode Attention - Correctness Tests")
    print("=" * 70)

    test_configs = [
        # (batch, num_q_heads, num_kv_heads, head_dim, kv_lens, max_splits, window)
        # Basic GQA tests
        (1, 32, 8, 128, [128], 8, -1),
        (1, 32, 8, 128, [512], 8, -1),
        (1, 32, 8, 128, [2048], 8, -1),
        (4, 32, 8, 128, [256, 512, 128, 1024], 8, -1),
        # Different head dims
        (1, 32, 8, 64, [512], 8, -1),
        (1, 8, 2, 512, [256], 8, -1),
        # MHA (kv_group_num=1)
        (1, 8, 8, 128, [512], 8, -1),
        # Large kv_group_num
        (1, 32, 4, 128, [512], 8, -1),
        (1, 64, 8, 128, [512], 8, -1),
        # Flash decoding with many splits
        (1, 32, 8, 128, [4096], 16, -1),
        (2, 32, 8, 128, [4096, 2048], 16, -1),
        # Sliding window
        (1, 32, 8, 128, [2048], 8, 512),
        (2, 32, 8, 128, [2048, 1024], 8, 256),
        # Sliding window + short seq (window > seq_len)
        (1, 32, 8, 128, [128], 8, 512),
    ]

    all_pass = True
    for batch, nqh, nkvh, hd, kvlens, ms, ws in test_configs:
        label = (f"batch={batch}, q_heads={nqh}, kv_heads={nkvh}, "
                 f"head_dim={hd}, kv_lens={kvlens}, splits={ms}, window={ws}")
        ok, max_diff, mean_diff = test_correctness(
            batch, nqh, nkvh, hd, kvlens, ms, ws
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
    print("Paged Decode Attention - Benchmarks")
    print("=" * 70)

    bench_configs = [
        # (batch, num_q_heads, num_kv_heads, head_dim, kv_len, max_splits, window)
        (1, 32, 8, 128, 512, 8, -1),
        (1, 32, 8, 128, 2048, 8, -1),
        (1, 32, 8, 128, 8192, 16, -1),
        (4, 32, 8, 128, 2048, 8, -1),
        (16, 32, 8, 128, 512, 8, -1),
    ]

    for batch, nqh, nkvh, hd, kvl, ms, ws in bench_configs:
        elapsed = bench_kernel(batch, nqh, nkvh, hd, kvl, ms, ws)
        label = (f"batch={batch}, q_heads={nqh}, kv_heads={nkvh}, "
                 f"head_dim={hd}, kv_len={kvl}, splits={ms}")
        print(f"  {label}  ->  {elapsed:.3f} ms")
