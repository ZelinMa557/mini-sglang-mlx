"""
Test and benchmark: moe_scatter_broadcast kernel

This kernel replaces the dst-first gather in _gather_sort:
  Original:  x_sorted = x[order // K]   (dst reads random src)
  Fused:     x_sorted = moe_scatter_broadcast(x, inv_order, K)  (src writes to K dsts)

Semantics:
  For each token t and expert slot k:
    out[inv_order[t*K + k], :] = x[t, :]
"""

import time
import mlx.core as mx
from mlx_serve_kernel import moe_scatter_broadcast


def benchmark_fn(fn, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()
        mx.eval(mx.zeros(1))
    times = []
    for _ in range(repeats):
        mx.eval(mx.zeros(1))
        start = time.perf_counter()
        fn()
        mx.eval(mx.zeros(1))
        end = time.perf_counter()
        times.append((end - start) * 1000)
    times.sort()
    return sum(times) / len(times)


def naive_gather(x, order, K):
    """Original dst-first: x_sorted = x[order // K]"""
    return x[order // K]


def generate_inputs(M, K, D, E, dtype=mx.float32):
    """Generate test inputs matching MoE _gather_sort."""
    x = mx.random.normal((M, D), dtype=dtype)
    indices = mx.random.randint(0, E, shape=(M * K,))
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    mx.eval(x, indices, order, inv_order)
    return x, indices, order, inv_order.astype(mx.uint32)


# --- Functional test ---
def run_functional_tests():
    print("=" * 60)
    print("Functional Tests")
    print("=" * 60)

    test_cases = [
        (1, 4, 2048, 32),
        (1, 8, 2048, 128),
        (4, 4, 2880, 32),
        (4, 8, 2048, 128),
        (32, 4, 2048, 32),
        (128, 8, 2048, 128),
        (1024, 4, 2880, 32),
        (1024, 8, 2048, 128),
        (2048, 4, 2880, 32),
        (2048, 8, 2048, 128),
    ]

    all_pass = True
    for M, K, D, E in test_cases:
        x, indices, order, inv_order = generate_inputs(M, K, D, E)

        # Naive result
        expected = naive_gather(x, order, K)

        # Fused kernel result
        result = moe_scatter_broadcast(x, inv_order, K)
        mx.eval(expected, result)

        if mx.allclose(expected, result, rtol=1e-5, atol=1e-5):
            print(f"  PASS  M={M:<5} K={K} D={D:<5} E={E:<4}")
        else:
            max_diff = mx.max(mx.abs(expected - result)).item()
            print(f"  FAIL  M={M:<5} K={K} D={D:<5} E={E:<4}  max_diff={max_diff}")
            all_pass = False

    if all_pass:
        print("\nAll functional tests passed!")
    else:
        print("\nSome tests FAILED!")
    return all_pass


# --- Benchmark ---
def run_benchmarks():
    print("\n" + "=" * 60)
    print("Performance Benchmarks")
    print("=" * 60)

    configs = [
        {"name": "Qwen3-30B-A3B", "D": 2048, "E": 128, "K": 8},
        {"name": "GPT-OSS-20B",   "D": 2880, "E": 32,  "K": 4},
    ]
    token_counts = [1, 2, 4, 1024, 2048]

    for cfg in configs:
        name, D, E, K = cfg["name"], cfg["D"], cfg["E"], cfg["K"]
        print(f"\n{'─' * 60}")
        print(f"Model: {name} (D={D}, E={E}, top_k={K})")
        print(f"{'─' * 60}")
        print(f"{'Tokens':<8} {'Naive (ms)':<14} {'Fused (ms)':<14} {'Speedup':<10}")
        print("─" * 46)

        for M in token_counts:
            x, indices, order, inv_order = generate_inputs(M, K, D, E)

            def run_naive():
                out = naive_gather(x, order, K)
                mx.eval(out)

            def run_fused():
                out = moe_scatter_broadcast(x, inv_order, K)
                mx.eval(out)

            t_naive = benchmark_fn(run_naive)
            t_fused = benchmark_fn(run_fused)
            speedup = t_naive / t_fused

            print(f"{M:<8} {t_naive:<14.3f} {t_fused:<14.3f} {speedup:<10.2f}x")

    # Also test with float16 and bfloat16
    print(f"\n{'─' * 60}")
    print("dtype comparison (M=1024, Qwen3 config)")
    print(f"{'─' * 60}")
    D, E, K, M = 2048, 128, 8, 1024
    for dtype, dtype_name in [(mx.float32, "float32"), (mx.float16, "float16"), (mx.bfloat16, "bfloat16")]:
        x, indices, order, inv_order = generate_inputs(M, K, D, E, dtype=dtype)

        def run_naive():
            out = naive_gather(x, order, K)
            mx.eval(out)

        def run_fused():
            out = moe_scatter_broadcast(x, inv_order, K)
            mx.eval(out)

        t_naive = benchmark_fn(run_naive)
        t_fused = benchmark_fn(run_fused)
        speedup = t_naive / t_fused
        print(f"  {dtype_name:<10} Naive={t_naive:.3f}ms  Fused={t_fused:.3f}ms  Speedup={speedup:.2f}x")


if __name__ == "__main__":
    mx.random.seed(42)
    if run_functional_tests():
        run_benchmarks()
