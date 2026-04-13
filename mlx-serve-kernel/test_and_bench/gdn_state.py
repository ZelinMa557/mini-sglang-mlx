"""
Fused GatedDeltaNet state kernel: correctness and performance checks.

This compares the new fused kernels against the current mlx-serve baseline:
- Prefill: Python loops over requests/tokens.
- Decode: gather -> single-step metal kernel -> scatter.

The fused kernels read state directly from slot_ids and write it back in-place.
Inputs q/k/v/g/beta use bfloat16 while recurrent state uses float32.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_ops
from mlx_serve_kernel import gdn_decode_inplace, gdn_prefill_inplace


@dataclass(frozen=True)
class GDNConfig:
    hk: int
    hv: int
    dk: int
    dv: int


CONFIGS = [
    GDNConfig(hk=16, hv=32, dk=128, dv=128),
    GDNConfig(hk=16, hv=48, dk=128, dv=128),
]

PREFILL_LENGTHS = [1, 128, 256, 512, 1024]
DECODE_BATCHES = [1, 2, 4, 8]
INPUT_SCALE = 0.05


def clone(a: mx.array) -> mx.array:
    return mx.array(to_numpy(a)).astype(a.dtype)


def to_numpy(a: mx.array) -> np.ndarray:
    if a.dtype == mx.bfloat16:
        return np.array(a.astype(mx.float32), dtype=np.float32)
    return np.array(a)


def clone_inputs(data: dict[str, mx.array]) -> dict[str, mx.array]:
    return {key: clone(value) for key, value in data.items()}

def numpy_decode_reference(data: dict[str, mx.array]) -> tuple[np.ndarray, np.ndarray]:
    q = to_numpy(data["q"]).astype(np.float32)
    k = to_numpy(data["k"]).astype(np.float32)
    v = to_numpy(data["v"]).astype(np.float32)
    g = to_numpy(data["g"]).astype(np.float32)
    beta = to_numpy(data["beta"]).astype(np.float32)
    state = to_numpy(data["state"]).astype(np.float32)
    slot_ids = to_numpy(data["slot_ids"]).astype(np.int32)

    outputs = []
    for b, slot in enumerate(slot_ids.tolist()):
        hk_per_hv = state[slot].shape[0] // q[b].shape[0]
        q_exp = np.repeat(q[b], hk_per_hv, axis=0) if hk_per_hv > 1 else q[b]
        k_exp = np.repeat(k[b], hk_per_hv, axis=0) if hk_per_hv > 1 else k[b]
        state[slot] = state[slot] * g[b][:, None, None]
        kv_mem = (state[slot] * k_exp[:, None, :]).sum(axis=-1)
        delta = (v[b] - kv_mem) * beta[b][:, None]
        state[slot] = state[slot] + k_exp[:, None, :] * delta[:, :, None]
        outputs.append((state[slot] * q_exp[:, None, :]).sum(axis=-1))

    return np.stack(outputs, axis=0), state


def numpy_prefill_reference(data: dict[str, mx.array]) -> tuple[np.ndarray, np.ndarray]:
    q = to_numpy(data["q"]).astype(np.float32)
    k = to_numpy(data["k"]).astype(np.float32)
    v = to_numpy(data["v"]).astype(np.float32)
    g = to_numpy(data["g"]).astype(np.float32)
    beta = to_numpy(data["beta"]).astype(np.float32)
    state = to_numpy(data["state"]).astype(np.float32)
    slot_ids = to_numpy(data["slot_ids"]).astype(np.int32)
    indptr = to_numpy(data["qo_indptr"]).astype(np.int32)

    outputs = []
    for i, slot in enumerate(slot_ids.tolist()):
        s = state[slot].copy()
        for token in range(int(indptr[i]), int(indptr[i + 1])):
            hk_per_hv = s.shape[0] // q[token].shape[0]
            q_exp = np.repeat(q[token], hk_per_hv, axis=0) if hk_per_hv > 1 else q[token]
            k_exp = np.repeat(k[token], hk_per_hv, axis=0) if hk_per_hv > 1 else k[token]
            s = s * g[token][:, None, None]
            kv_mem = (s * k_exp[:, None, :]).sum(axis=-1)
            delta = (v[token] - kv_mem) * beta[token][:, None]
            s = s + k_exp[:, None, :] * delta[:, :, None]
            outputs.append((s * q_exp[:, None, :]).sum(axis=-1))
        state[slot] = s

    return np.stack(outputs, axis=0), state


def build_decode_inputs(config: GDNConfig, batch: int) -> dict[str, mx.array]:
    num_slots = batch
    slot_ids = mx.array(np.arange(0, batch, dtype=np.int32), dtype=mx.int32)
    q = (INPUT_SCALE * mx.random.normal((batch, config.hk, config.dk), dtype=mx.float32)).astype(mx.bfloat16)
    k = (INPUT_SCALE * mx.random.normal((batch, config.hk, config.dk), dtype=mx.float32)).astype(mx.bfloat16)
    v = (INPUT_SCALE * mx.random.normal((batch, config.hv, config.dv), dtype=mx.float32)).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((batch, config.hv), dtype=mx.float32)).astype(mx.bfloat16)
    beta = mx.sigmoid(mx.random.normal((batch, config.hv), dtype=mx.float32)).astype(mx.bfloat16)
    state = INPUT_SCALE * mx.random.normal(
        (num_slots + 1, config.hv, config.dv, config.dk), dtype=mx.float32
    )
    mx.eval(q, k, v, g, beta, state, slot_ids)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "state": state,
        "slot_ids": slot_ids,
    }


def build_prefill_inputs(config: GDNConfig, lengths: list[int]) -> dict[str, mx.array]:
    batch = len(lengths)
    total = sum(lengths)
    num_slots = batch + 5
    slot_ids = mx.array(np.arange(1, batch + 1, dtype=np.int32), dtype=mx.int32)
    indptr = [0]
    for length in lengths:
        indptr.append(indptr[-1] + length)
    qo_indptr = mx.array(indptr, dtype=mx.int32)
    q = (INPUT_SCALE * mx.random.normal((total, config.hk, config.dk), dtype=mx.float32)).astype(mx.bfloat16)
    k = (INPUT_SCALE * mx.random.normal((total, config.hk, config.dk), dtype=mx.float32)).astype(mx.bfloat16)
    v = (INPUT_SCALE * mx.random.normal((total, config.hv, config.dv), dtype=mx.float32)).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((total, config.hv), dtype=mx.float32)).astype(mx.bfloat16)
    beta = mx.sigmoid(mx.random.normal((total, config.hv), dtype=mx.float32)).astype(mx.bfloat16)
    state = INPUT_SCALE * mx.random.normal(
        (num_slots + 1, config.hv, config.dv, config.dk), dtype=mx.float32
    )
    mx.eval(q, k, v, g, beta, state, slot_ids, qo_indptr)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "state": state,
        "slot_ids": slot_ids,
        "qo_indptr": qo_indptr,
    }


def prepare_baseline_decode_inputs(data: dict[str, mx.array]) -> dict[str, mx.array]:
    prepared = {
        "q": data["q"][:, None, :, :],
        "k": data["k"][:, None, :, :],
        "v": data["v"][:, None, :, :],
        "g": data["g"][:, None, :],
        "beta": data["beta"][:, None, :],
        "state": data["state"][data["slot_ids"]],
    }
    mx.eval(
        prepared["q"],
        prepared["k"],
        prepared["v"],
        prepared["g"],
        prepared["beta"],
        prepared["state"],
    )
    return prepared


def prepare_baseline_prefill_inputs(data: dict[str, mx.array]) -> dict[str, mx.array]:
    slot_ids_np = to_numpy(data["slot_ids"]).astype(np.int32)
    indptr_np = to_numpy(data["qo_indptr"]).astype(np.int32)
    assert len(slot_ids_np) == 1, "benchmark prefill baseline expects a single request"
    slot = int(slot_ids_np[0])
    start = int(indptr_np[0])
    end = int(indptr_np[1])
    prepared = {
        "q": data["q"][start:end][None, :, :, :],
        "k": data["k"][start:end][None, :, :, :],
        "v": data["v"][start:end][None, :, :, :],
        "g": data["g"][start:end][None, :, :],
        "beta": data["beta"][start:end][None, :, :],
        "state": data["state"][slot : slot + 1],
    }
    mx.eval(
        prepared["q"],
        prepared["k"],
        prepared["v"],
        prepared["g"],
        prepared["beta"],
        prepared["state"],
    )
    return prepared


def assert_close(name: str, lhs: mx.array, rhs: mx.array, atol: float = 5e-4) -> None:
    lhs_np = to_numpy(lhs).astype(np.float32)
    rhs_np = to_numpy(rhs).astype(np.float32)
    diff = np.abs(lhs_np - rhs_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    ok = max_diff <= atol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:<42} max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}")
    if not ok:
        raise AssertionError(f"{name} failed: max_diff={max_diff}")


def run_fused_decode_bench(data: dict[str, mx.array]) -> None:
    y = gdn_decode_inplace(
        data["q"], data["k"], data["v"], data["g"], data["beta"],
        data["state"], data["slot_ids"],
    )
    mx.eval(y)


def run_fused_prefill_bench(data: dict[str, mx.array]) -> None:
    y = gdn_prefill_inplace(
        data["q"], data["k"], data["v"], data["g"], data["beta"],
        data["state"], data["slot_ids"], data["qo_indptr"],
    )
    mx.eval(y)


def run_baseline_decode_bench(prepared: dict[str, mx.array]) -> None:
    y, new_state = gated_delta_ops(
        prepared["q"],
        prepared["k"],
        prepared["v"],
        prepared["g"],
        prepared["beta"],
        prepared["state"],
    )
    mx.eval(y, new_state)
    prepared["state"] = new_state


def run_baseline_prefill_bench(prepared: dict[str, mx.array]) -> None:
    y, new_state = gated_delta_ops(
        prepared["q"],
        prepared["k"],
        prepared["v"],
        prepared["g"],
        prepared["beta"],
        prepared["state"],
    )
    mx.eval(y, new_state)
    prepared["state"] = new_state


def run_correctness() -> None:
    print("=" * 88)
    print("Fused GDN state kernel correctness")
    print("=" * 88)

    for config in CONFIGS:
        print(
            f"config hk={config.hk:>2} hv={config.hv:>2} "
            f"dk={config.dk:>3} dv={config.dv:>3}"
        )

        for batch in (1, 4, 8):
            data = build_decode_inputs(config, batch)
            fused_state = clone(data["state"])
            fused_y = gdn_decode_inplace(
                data["q"], data["k"], data["v"], data["g"], data["beta"],
                fused_state, data["slot_ids"],
            )
            ref_y, ref_state = numpy_decode_reference(data)
            assert_close(f"decode batch={batch} output", fused_y, mx.array(ref_y))
            assert_close(f"decode batch={batch} state", fused_state, mx.array(ref_state))

        for lengths in ([1], [7], [128], [33, 65], [17, 129, 257]):
            data = build_prefill_inputs(config, list(lengths))
            fused_state = clone(data["state"])
            fused_y = gdn_prefill_inplace(
                data["q"], data["k"], data["v"], data["g"], data["beta"],
                fused_state, data["slot_ids"], data["qo_indptr"],
            )
            ref_y, ref_state = numpy_prefill_reference(data)
            assert_close(f"prefill lens={list(lengths)} output", fused_y, mx.array(ref_y))
            assert_close(f"prefill lens={list(lengths)} state", fused_state, mx.array(ref_state))
        print()


def bench_once(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat * 1000.0


def run_prefill_perf(warmup: int, repeat: int) -> None:
    print("=" * 88)
    print("Prefill performance vs baseline implementation (ms)")
    print("=" * 88)
    print(f"{'hk':>4} {'hv':>4} {'dk':>5} {'dv':>5} {'len':>8} {'fused':>10} {'baseline':>10} {'speedup':>8}")
    for config in CONFIGS:
        for length in PREFILL_LENGTHS:
            fused_data = build_prefill_inputs(config, [length])
            base_data = clone_inputs(fused_data)
            baseline_prepared = prepare_baseline_prefill_inputs(base_data)

            fused_ms = bench_once(
                lambda: run_fused_prefill_bench(fused_data),
                warmup=warmup,
                repeat=repeat,
            )
            base_ms = bench_once(
                lambda: run_baseline_prefill_bench(baseline_prepared),
                warmup=warmup,
                repeat=repeat,
            )
            speedup = base_ms / fused_ms if fused_ms > 0 else float("inf")
            print(
                f"{config.hk:>4} {config.hv:>4} {config.dk:>5} {config.dv:>5} "
                f"{length:>8} {fused_ms:>10.3f} {base_ms:>10.3f} {speedup:>7.2f}x"
            )


def run_decode_perf(warmup: int, repeat: int) -> None:
    print("=" * 88)
    print("Decode performance vs baseline implementation (ms)")
    print("=" * 88)
    print(f"{'hk':>4} {'hv':>4} {'dk':>5} {'dv':>5} {'batch':>8} {'fused':>10} {'baseline':>10} {'speedup':>8}")
    for config in CONFIGS:
        for batch in DECODE_BATCHES:
            fused_data = build_decode_inputs(config, batch)
            base_data = clone_inputs(fused_data)
            baseline_prepared = prepare_baseline_decode_inputs(base_data)

            fused_ms = bench_once(
                lambda: run_fused_decode_bench(fused_data),
                warmup=warmup,
                repeat=repeat,
            )
            base_ms = bench_once(
                lambda: run_baseline_decode_bench(baseline_prepared),
                warmup=warmup,
                repeat=repeat,
            )
            speedup = base_ms / fused_ms if fused_ms > 0 else float("inf")
            print(
                f"{config.hk:>4} {config.hv:>4} {config.dk:>5} {config.dv:>5} "
                f"{batch:>8} {fused_ms:>10.3f} {base_ms:>10.3f} {speedup:>7.2f}x"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mx.random.seed(42)
    np.random.seed(42)

    if not args.skip_correctness:
        run_correctness()
    if not args.skip_perf:
        run_prefill_perf(args.warmup, args.repeat)
        print()
        run_decode_perf(args.warmup, args.repeat)


if __name__ == "__main__":
    main()
