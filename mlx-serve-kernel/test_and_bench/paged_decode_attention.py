"""
Paged Decode Attention: correctness & performance tests vs MLX SDPA.

Decode attention: each sequence has 1 new query token attending to all
historical KV tokens. Our kernel uses paged KV cache with ragged batching;
MLX SDPA uses dense [1, N_q, 1, D] / [1, N_kv, T_kv, D] per sequence.
"""

import numpy as np
import mlx.core as mx
import math
import time
from mlx_serve_kernel import paged_decode_attention

# ── Helpers ──────────────────────────────────────────────────────────────────

HEAD_DIM = 128


def build_paged_decode_inputs(
    kv_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    dtype=mx.bfloat16,
):
    """
    Build a shared paged KV cache and inputs for both our kernel and MLX SDPA.

    Returns a dict with all arrays needed for both paths.
    """
    batch = len(kv_lens)
    total_kv = sum(kv_lens)
    num_pages = total_kv  # 1:1 page mapping for simplicity

    # Random KV cache (flat paged): (num_pages, num_kv_heads, head_dim)
    k_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)
    v_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)

    # Random queries: (batch, num_q_heads, head_dim)
    q = mx.random.normal((batch, num_q_heads, head_dim), dtype=dtype)

    # Build kv_indptr and kv_indices (identity page mapping)
    indptr = [0]
    for l in kv_lens:
        indptr.append(indptr[-1] + l)
    kv_indptr = mx.array(indptr, dtype=mx.int32)
    kv_indices = mx.array(np.arange(total_kv, dtype=np.int32))

    # num_kv_splits
    max_kv_splits = 32
    splits = []
    for l in kv_lens:
        s = min(max_kv_splits, max(1, (l + 127) // 128))
        splits.append(s)
    num_kv_splits = mx.array(splits, dtype=mx.int32)

    mx.eval(q, k_cache, v_cache, kv_indptr, kv_indices, num_kv_splits)

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        kv_lens=kv_lens,
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def run_our_kernel(data: dict) -> mx.array:
    """Run our paged decode attention kernel."""
    sm_scale = 1.0 / math.sqrt(data["head_dim"])
    out = paged_decode_attention(
        data["q"],
        data["k_cache"],
        data["v_cache"],
        data["kv_indptr"],
        data["kv_indices"],
        data["num_kv_splits"],
        sm_scale=sm_scale,
        max_kv_splits=data["max_kv_splits"],
    )
    return out


def run_mlx_sdpa(data: dict) -> mx.array:
    """
    Run MLX SDPA per-sequence as reference.
    For decode: q is [1, N_q, 1, D], k/v are [1, N_kv, T_kv, D].
    No mask needed since decode query attends to all KV tokens.
    Returns: (batch, N_q, D)
    """
    sm_scale = 1.0 / math.sqrt(data["head_dim"])
    kv_indptr_np = np.array(data["kv_indptr"], dtype=np.int32)
    kv_indices_np = np.array(data["kv_indices"], dtype=np.int32)

    results = []
    for b in range(data["batch"]):
        kv_start = int(kv_indptr_np[b])
        kv_end = int(kv_indptr_np[b + 1])
        page_ids = mx.array(kv_indices_np[kv_start:kv_end].astype(np.int32))

        # Gather dense K, V for this sequence
        k_seq = data["k_cache"][page_ids]  # (T_kv, N_kv, D)
        v_seq = data["v_cache"][page_ids]

        # Reshape for SDPA: [1, N_kv, T_kv, D]
        k_seq = mx.expand_dims(mx.transpose(k_seq, (1, 0, 2)), axis=0)
        v_seq = mx.expand_dims(mx.transpose(v_seq, (1, 0, 2)), axis=0)

        # Query: [1, N_q, 1, D]
        q_seq = data["q"][b:b+1].reshape(1, data["num_q_heads"], 1, data["head_dim"])

        out_seq = mx.fast.scaled_dot_product_attention(
            q_seq, k_seq, v_seq, scale=sm_scale
        )  # [1, N_q, 1, D]
        results.append(out_seq.reshape(data["num_q_heads"], data["head_dim"]))

    return mx.stack(results, axis=0)  # (batch, N_q, D)


# ── Correctness ──────────────────────────────────────────────────────────────

def test_correctness(
    kv_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    dtype=mx.bfloat16,
):
    data = build_paged_decode_inputs(kv_lens, num_q_heads, num_kv_heads, head_dim, dtype)

    out_ours = run_our_kernel(data)
    mx.eval(out_ours)

    out_ref = run_mlx_sdpa(data)
    mx.eval(out_ref)

    out_ours_f32 = np.array(out_ours.astype(mx.float32))
    out_ref_f32 = np.array(out_ref.astype(mx.float32))
    diff = np.abs(out_ours_f32 - out_ref_f32)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    atol = 5e-2
    passed = max_diff < atol
    return passed, max_diff, mean_diff


# ── Benchmark ────────────────────────────────────────────────────────────────

def bench(
    kv_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    dtype=mx.bfloat16,
    warmup: int = 20,
    repeat: int = 100,
):
    data = build_paged_decode_inputs(kv_lens, num_q_heads, num_kv_heads, head_dim, dtype)

    # Warmup + bench our kernel
    for _ in range(warmup):
        mx.eval(run_our_kernel(data))
    t0 = time.perf_counter()
    for _ in range(repeat):
        mx.eval(run_our_kernel(data))
    ours_ms = (time.perf_counter() - t0) / repeat * 1000

    # Warmup + bench MLX SDPA
    for _ in range(warmup):
        mx.eval(run_mlx_sdpa(data))
    t0 = time.perf_counter()
    for _ in range(repeat):
        mx.eval(run_mlx_sdpa(data))
    sdpa_ms = (time.perf_counter() - t0) / repeat * 1000

    return ours_ms, sdpa_ms


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mx.random.seed(42)
    np.random.seed(42)

    # ────── Correctness tests ──────
    print("=" * 80)
    print("Paged Decode Attention — Correctness (vs MLX SDPA)")
    print("=" * 80)

    single_seq_lens = [1, 237, 512, 809, 1024, 2048, 3333, 4096, 8192]
    kv_heads_list = [2]

    # Single-sequence tests
    all_pass = True
    for nkvh in kv_heads_list:
        for kvl in single_seq_lens:
            ok, md, ad = test_correctness([kvl], 16, nkvh)
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] kv_heads={nkvh:>2}, kv_len={kvl:>5}  "
                  f"max_diff={md:.6f}  mean_diff={ad:.6f}")
            if not ok:
                all_pass = False

    # Multi-sequence (ragged batch) tests
    print()
    print("  --- Multi-sequence (ragged batch) ---")
    multi_seq_configs = [
        [1, 237],
        [512, 1024, 2048],
        [237, 809, 3333, 4096],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [4096, 4096, 4096, 4096],
    ]
    for nkvh in kv_heads_list:
        for kv_lens in multi_seq_configs:
            ok, md, ad = test_correctness(kv_lens, 32, nkvh)
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] kv_heads={nkvh:>2}, kv_lens={str(kv_lens):>30}  "
                  f"max_diff={md:.6f}  mean_diff={ad:.6f}")
            if not ok:
                all_pass = False

    print()
    print("ALL PASSED" if all_pass else "SOME TESTS FAILED")

    # ────── Performance tests ──────
    print()
    print("=" * 80)
    print("Paged Decode Attention — Performance (us)")
    print("=" * 80)
    print(f"  {'kv_heads':>8} {'kv_len':>8} {'ours(us)':>10} {'sdpa(us)':>10} {'speedup':>8}")
    print("  " + "-" * 50)

    for nkvh in kv_heads_list:
        for kvl in single_seq_lens:
            ours_ms, sdpa_ms = bench([kvl], 16, nkvh)
            speedup = sdpa_ms / ours_ms if ours_ms > 0 else float("inf")
            print(f"  {nkvh:>8} {kvl:>8} {ours_ms*1000:>10.1f} {sdpa_ms*1000:>10.1f} {speedup:>7.2f}x")
