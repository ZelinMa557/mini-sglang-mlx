"""
Paged Prefill Attention: correctness & performance tests vs MLX SDPA.

Prefill (extend) attention: each sequence has T_q new query tokens attending
to prefix KV + its own causal tokens. Our kernel uses paged KV cache with
ragged batching; MLX SDPA uses dense tensors per sequence.

For the correctness baseline we loop over sequences and call MLX SDPA with a
custom boolean mask that encodes the prefix-extend causal pattern.
"""

import numpy as np
import mlx.core as mx
import math
import time
from mini_sglang_mlx_kernel import paged_prefill_attention

# ── Helpers ──────────────────────────────────────────────────────────────────

HEAD_DIM = 256
HEAD_CONFIGS = [
    (256, 2, 16),  # (head_dim, kv_heads, q_heads)
    (256, 4, 16),
    (256, 4, 24),
    (256, 2, 32),
]


def build_paged_prefill_inputs(
    q_lens: list[int],
    prefix_lens_list: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    dtype=mx.bfloat16,
):
    """
    Build a shared paged KV cache and inputs for both our kernel and MLX SDPA.

    For each sequence i:
      - kv_len_i = prefix_lens_list[i] + q_lens[i]
      - The first prefix_lens_list[i] KV tokens are "prefix" (no causal mask)
      - The remaining q_lens[i] KV tokens are "extend" (causal within this region)

    Returns a dict with all arrays needed for both paths.
    """
    batch = len(q_lens)
    kv_lens = [p + q for p, q in zip(prefix_lens_list, q_lens)]
    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    num_pages = total_kv
    max_len_extend = max(q_lens)

    # Random data
    q = mx.random.normal((total_q, num_q_heads, head_dim), dtype=dtype)
    k_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)
    v_cache = mx.random.normal((num_pages, num_kv_heads, head_dim), dtype=dtype)

    # qo_indptr
    qo_ind = [0]
    for ql in q_lens:
        qo_ind.append(qo_ind[-1] + ql)
    qo_indptr = mx.array(qo_ind, dtype=mx.int32)

    # kv_indptr
    kv_ind = [0]
    for kvl in kv_lens:
        kv_ind.append(kv_ind[-1] + kvl)
    kv_indptr = mx.array(kv_ind, dtype=mx.int32)

    # Identity page mapping
    kv_indices = mx.array(np.arange(total_kv, dtype=np.int32))

    prefix_lens = mx.array(prefix_lens_list, dtype=mx.int32)

    mx.eval(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens)

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        prefix_lens=prefix_lens,
        max_len_extend=max_len_extend,
        q_lens=q_lens,
        prefix_lens_list=prefix_lens_list,
        kv_lens=kv_lens,
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def run_our_kernel(data: dict) -> mx.array:
    """Run our paged prefill attention kernel."""
    sm_scale = 1.0 / math.sqrt(data["head_dim"])
    out = paged_prefill_attention(
        data["q"],
        data["k_cache"],
        data["v_cache"],
        data["qo_indptr"],
        data["kv_indptr"],
        data["kv_indices"],
        data["prefix_lens"],
        sm_scale=sm_scale,
        max_len_extend=data["max_len_extend"],
    )
    return out


def run_mlx_sdpa(data: dict) -> mx.array:
    """
    Run MLX SDPA per-sequence as reference.

    For each sequence we build a boolean mask of shape (T_q, T_kv) where:
      - Prefix region (kv_pos < prefix_len): always True (attend to all prefix)
      - Extend region (kv_pos >= prefix_len): causal within extend,
        i.e. q_idx >= (kv_pos - prefix_len)

    Then call SDPA with q=[1, N_q, T_q, D], k/v=[1, N_kv, T_kv, D].
    Concatenate results.
    """
    sm_scale = 1.0 / math.sqrt(data["head_dim"])
    qo_indptr_np = np.array(data["qo_indptr"], dtype=np.int32)
    kv_indptr_np = np.array(data["kv_indptr"], dtype=np.int32)
    kv_indices_np = np.array(data["kv_indices"], dtype=np.int32)

    results = []
    for b in range(data["batch"]):
        q_start = int(qo_indptr_np[b])
        q_end = int(qo_indptr_np[b + 1])
        q_len = q_end - q_start

        kv_start = int(kv_indptr_np[b])
        kv_end = int(kv_indptr_np[b + 1])
        kv_len = kv_end - kv_start
        prefix_len = data["prefix_lens_list"][b]

        page_ids = mx.array(kv_indices_np[kv_start:kv_end].astype(np.int32))

        # Gather dense K, V
        k_seq = data["k_cache"][page_ids]  # (T_kv, N_kv, D)
        v_seq = data["v_cache"][page_ids]

        # Reshape: [1, N_kv, T_kv, D]
        k_seq = mx.expand_dims(mx.transpose(k_seq, (1, 0, 2)), axis=0)
        v_seq = mx.expand_dims(mx.transpose(v_seq, (1, 0, 2)), axis=0)

        # Query: [1, N_q, T_q, D]
        q_seq = data["q"][q_start:q_end]  # (T_q, N_q, D)
        q_seq = mx.expand_dims(mx.transpose(q_seq, (1, 0, 2)), axis=0)

        # Build boolean mask: (T_q, T_kv)
        # mask[i, j] = True means attend
        mask_np = np.zeros((q_len, kv_len), dtype=bool)
        for i in range(q_len):
            for j in range(kv_len):
                if j < prefix_len:
                    # Prefix: always attend
                    mask_np[i, j] = True
                else:
                    # Extend: causal (q_offset >= k_extend_offset)
                    k_ext = j - prefix_len
                    if i >= k_ext:
                        mask_np[i, j] = True

        mask = mx.array(mask_np)  # (T_q, T_kv), broadcast to [1, 1, T_q, T_kv]

        out_seq = mx.fast.scaled_dot_product_attention(
            q_seq, k_seq, v_seq, scale=sm_scale, mask=mask
        )  # [1, N_q, T_q, D]

        # Back to (T_q, N_q, D)
        out_seq = mx.transpose(out_seq.squeeze(0), (1, 0, 2))
        results.append(out_seq)

    return mx.concatenate(results, axis=0)  # (total_q, N_q, D)


# ── Correctness ──────────────────────────────────────────────────────────────

def test_correctness(
    q_lens: list[int],
    prefix_lens_list: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    dtype=mx.bfloat16,
):
    data = build_paged_prefill_inputs(
        q_lens, prefix_lens_list, num_q_heads, num_kv_heads, head_dim, dtype
    )

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
    q_lens: list[int],
    prefix_lens_list: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    dtype=mx.bfloat16,
    warmup: int = 1,
    repeat: int = 5,
):
    data = build_paged_prefill_inputs(
        q_lens, prefix_lens_list, num_q_heads, num_kv_heads, head_dim, dtype
    )
    q_len = q_lens[0]
    q = mx.random.normal((1, num_q_heads, q_len, head_dim), dtype=dtype)
    k = mx.random.normal((1, num_kv_heads, q_len, head_dim), dtype=dtype)
    v = mx.random.normal((1, num_kv_heads, q_len, head_dim), dtype=dtype)
    mx.eval(q, k, v)
    scale = head_dim ** -0.5

    # Warmup + bench our kernel
    for _ in range(warmup):
        mx.eval(run_our_kernel(data))
    t0 = time.perf_counter()
    for _ in range(repeat):
        mx.eval(run_our_kernel(data))
    ours_ms = (time.perf_counter() - t0) / repeat * 1000

    # Warmup + bench MLX SDPA
    for _ in range(warmup):
        mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal"))
    t0 = time.perf_counter()
    for _ in range(repeat):
        mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal"))
    sdpa_ms = (time.perf_counter() - t0) / repeat * 1000

    return ours_ms, sdpa_ms


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mx.random.seed(42)
    np.random.seed(42)

    # ────── Correctness tests ──────
    print("=" * 80)
    print("Paged Prefill Attention — Correctness (vs MLX SDPA)")
    print("=" * 80)

    # Single-sequence, no prefix (pure prefill)
    single_seq_lens = [1, 237, 512, 809, 1024, 2048, 3333, 4096, 8192]
    all_pass = True

    print("  --- Single sequence, no prefix (pure prefill) ---")
    for head_dim, nkvh, nqh in HEAD_CONFIGS:
        for ql in single_seq_lens:
            ok, md, ad = test_correctness([ql], [0], nqh, nkvh, head_dim=head_dim)
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] head_dim={head_dim:>3}, kv_heads={nkvh:>2}, q_heads={nqh:>2}, q_len={ql:>5}, prefix=0     "
                  f"max_diff={md:.6f}  mean_diff={ad:.6f}")
            if not ok:
                all_pass = False

    # Single-sequence, with prefix (extend)
    print()
    print("  --- Single sequence, with prefix (extend) ---")
    prefix_configs = [
        (237, 100),
        (512, 256),
        (1024, 512),
        (2048, 1024),
    ]
    for head_dim, nkvh, nqh in HEAD_CONFIGS:
        for ql, pl in prefix_configs:
            ok, md, ad = test_correctness([ql], [pl], nqh, nkvh, head_dim=head_dim)
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] head_dim={head_dim:>3}, kv_heads={nkvh:>2}, q_heads={nqh:>2}, q_len={ql:>5}, prefix={pl:>5}  "
                  f"max_diff={md:.6f}  mean_diff={ad:.6f}")
            if not ok:
                all_pass = False

    # Multi-sequence (ragged batch)
    print()
    print("  --- Multi-sequence (ragged batch) ---")
    multi_seq_configs = [
        # (q_lens, prefix_lens)
        ([1, 237], [0, 0]),
        ([512, 1024], [0, 0]),
        ([237, 809, 1024], [0, 100, 512]),
        ([1, 512, 2048, 4096], [0, 0, 0, 0]),
        ([128, 256, 512, 1024], [64, 128, 256, 512]),
    ]
    for head_dim, nkvh, nqh in HEAD_CONFIGS:
        for q_lens, p_lens in multi_seq_configs:
            ok, md, ad = test_correctness(q_lens, p_lens, nqh, nkvh, head_dim=head_dim)
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] head_dim={head_dim:>3}, kv_heads={nkvh:>2}, q_heads={nqh:>2}, q_lens={str(q_lens):>28}, "
                  f"prefix={str(p_lens):>20}  max_diff={md:.6f}  mean_diff={ad:.6f}")
            if not ok:
                all_pass = False

    print()
    print("ALL PASSED" if all_pass else "SOME TESTS FAILED")

    # ────── Performance tests ──────
    print()
    print("=" * 80)
    print("Paged Prefill Attention — Performance (us)")
    print("=" * 80)
    print(f"  {'hdim':>6} {'kv_heads':>8} {'q_heads':>8} {'q_len':>8} {'prefix':>8} {'ours(us)':>10} {'sdpa(us)':>10} {'speedup':>8}")
    print("  " + "-" * 80)

    for head_dim, nkvh, nqh in HEAD_CONFIGS:
        for ql in single_seq_lens:
            ours_ms, sdpa_ms = bench([ql], [0], nqh, nkvh, head_dim=head_dim)
            speedup = sdpa_ms / ours_ms if ours_ms > 0 else float("inf")
            print(f"  {head_dim:>6} {nkvh:>8} {nqh:>8} {ql:>8} {0:>8} "
                  f"{ours_ms*1000:>10.1f} {sdpa_ms*1000:>10.1f} {speedup:>7.2f}x")
