"""
Fused GatedDeltaNet state kernel: correctness and performance checks.

This compares the new fused kernels against the current mlx-serve baseline:
- Prefill: Python loops over requests/tokens.
- Decode: gather -> single-step metal kernel -> scatter.

The fused kernels read state directly from slot_ids and write it back in-place.
All tests use float32 because the kernel intentionally only supports fp32.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

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

PREFILL_LENGTHS = [1, 128, 256, 512, 1024, 2048, 4096]
DECODE_BATCHES = [1, 2, 4, 8]
INPUT_SCALE = 0.05


def clone(a: mx.array) -> mx.array:
    return mx.array(np.array(a))


def clone_inputs(data: dict[str, mx.array]) -> dict[str, mx.array]:
    return {key: clone(value) for key, value in data.items()}


def _gated_delta_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    hk_per_hv = state.shape[0] // q.shape[0]
    q_exp = mx.repeat(q, hk_per_hv, axis=0) if hk_per_hv > 1 else q
    k_exp = mx.repeat(k, hk_per_hv, axis=0) if hk_per_hv > 1 else k
    state = state * g[:, None, None]
    kv_mem = (state * k_exp[:, None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[:, None]
    state = state + k_exp[:, None, :] * delta[:, :, None]
    y = (state * q_exp[:, None, :]).sum(axis=-1)
    return y, state


@mx.compile
def _gated_delta_step_batched(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    hv = state.shape[1]
    hk = q.shape[1]
    hk_per_hv = hv // hk
    if hk_per_hv > 1:
        q = mx.repeat(q, hk_per_hv, axis=1)
        k = mx.repeat(k, hk_per_hv, axis=1)
    state = state * g[:, :, None, None]
    kv_mem = (state * k[:, :, None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[:, :, None]
    state = state + k[:, :, None, :] * delta[:, :, :, None]
    y = (state * q[:, :, None, :]).sum(axis=-1)
    return y, state


_decode_kernel_cache: dict[tuple[int, int, int, int], object] = {}


def _get_baseline_decode_kernel(hk: int, hv: int, dk: int, dv: int):
    key = (hk, hv, dk, dv)
    if key in _decode_kernel_cache:
        return _decode_kernel_cache[key]

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto q_ = q + b_idx * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * Hv * Dv + hv_idx * Dv;
        y += b_idx * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        auto g_ = g + b_idx * Hv;
        auto beta_ = beta + b_idx * Hv;

        float kv_mem = 0.0f;
        for (int s_idx = dk_idx; s_idx < Dk; s_idx += 32) {{
          auto state = static_cast<float>(i_state[s_idx]) * g_[hv_idx];
          kv_mem += state * k_[s_idx];
          o_state[s_idx] = state;
        }}
        kv_mem = simd_sum(kv_mem);

        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
        float out = 0.0f;
        for (int s_idx = dk_idx; s_idx < Dk; s_idx += 32) {{
          auto state = static_cast<float>(o_state[s_idx]) + k_[s_idx] * delta;
          o_state[s_idx] = state;
          out += state * q_[s_idx];
        }}
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {{
          y[dv_idx] = static_cast<InT>(out);
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=f"gdn_decode_baseline_hk{hk}_hv{hv}_dk{dk}_dv{dv}",
        input_names=["q", "k", "v", "g", "beta", "state_in"],
        output_names=["y", "state_out"],
        source=source,
    )
    _decode_kernel_cache[key] = kernel
    return kernel


def baseline_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    slot_ids: mx.array,
) -> mx.array:
    state_batch = state[slot_ids]
    hk, dk = q.shape[1], q.shape[2]
    hv, dv = v.shape[1], v.shape[2]
    kernel = _get_baseline_decode_kernel(hk, hv, dk, dv)
    y, new_state = kernel(
        inputs=[q, k, v, g, beta, state_batch],
        template=[("InT", q.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)],
        grid=(32, dv, q.shape[0] * hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(q.shape[0], hv, dv), state_batch.shape],
        output_dtypes=[q.dtype, q.dtype],
    )
    state[slot_ids] = new_state
    return y


def baseline_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    slot_ids: mx.array,
    qo_indptr: mx.array,
) -> mx.array:
    output_parts = []
    slot_ids_np = np.array(slot_ids, dtype=np.int32)
    indptr_np = np.array(qo_indptr, dtype=np.int32)

    for req_idx, slot in enumerate(slot_ids_np.tolist()):
        start = int(indptr_np[req_idx])
        end = int(indptr_np[req_idx + 1])
        seg_state = state[slot]
        ys = []
        for token in range(start, end):
            y_t, seg_state = _gated_delta_step(
                q[token], k[token], v[token], g[token], beta[token], seg_state,
            )
            ys.append(y_t)
        state[slot] = seg_state
        output_parts.append(mx.stack(ys, axis=0))

    return mx.concatenate(output_parts, axis=0)


def numpy_decode_reference(data: dict[str, mx.array]) -> tuple[np.ndarray, np.ndarray]:
    q = np.array(data["q"], dtype=np.float32)
    k = np.array(data["k"], dtype=np.float32)
    v = np.array(data["v"], dtype=np.float32)
    g = np.array(data["g"], dtype=np.float32)
    beta = np.array(data["beta"], dtype=np.float32)
    state = np.array(data["state"], dtype=np.float32)
    slot_ids = np.array(data["slot_ids"], dtype=np.int32)

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
    q = np.array(data["q"], dtype=np.float32)
    k = np.array(data["k"], dtype=np.float32)
    v = np.array(data["v"], dtype=np.float32)
    g = np.array(data["g"], dtype=np.float32)
    beta = np.array(data["beta"], dtype=np.float32)
    state = np.array(data["state"], dtype=np.float32)
    slot_ids = np.array(data["slot_ids"], dtype=np.int32)
    indptr = np.array(data["qo_indptr"], dtype=np.int32)

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
    num_slots = batch + 5
    slot_ids = mx.array(np.arange(1, batch + 1, dtype=np.int32), dtype=mx.int32)
    q = INPUT_SCALE * mx.random.normal((batch, config.hk, config.dk), dtype=mx.float32)
    k = INPUT_SCALE * mx.random.normal((batch, config.hk, config.dk), dtype=mx.float32)
    v = INPUT_SCALE * mx.random.normal((batch, config.hv, config.dv), dtype=mx.float32)
    g = mx.sigmoid(mx.random.normal((batch, config.hv), dtype=mx.float32))
    beta = mx.sigmoid(mx.random.normal((batch, config.hv), dtype=mx.float32))
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
    q = INPUT_SCALE * mx.random.normal((total, config.hk, config.dk), dtype=mx.float32)
    k = INPUT_SCALE * mx.random.normal((total, config.hk, config.dk), dtype=mx.float32)
    v = INPUT_SCALE * mx.random.normal((total, config.hv, config.dv), dtype=mx.float32)
    g = mx.sigmoid(mx.random.normal((total, config.hv), dtype=mx.float32))
    beta = mx.sigmoid(mx.random.normal((total, config.hv), dtype=mx.float32))
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


def assert_close(name: str, lhs: mx.array, rhs: mx.array, atol: float = 5e-4) -> None:
    lhs_np = np.array(lhs, dtype=np.float32)
    rhs_np = np.array(rhs, dtype=np.float32)
    diff = np.abs(lhs_np - rhs_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    ok = max_diff <= atol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:<42} max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}")
    if not ok:
        raise AssertionError(f"{name} failed: max_diff={max_diff}")


def sync_token(out: mx.array, state: mx.array, slot_ids: mx.array) -> float:
    token = mx.sum(out.astype(mx.float32))
    token = token + mx.sum(state[slot_ids][:, :1, :1, :1].astype(mx.float32))
    return float(np.array(token))


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
    print("Prefill performance vs current implementation (ms)")
    print("=" * 88)
    print(f"{'hk':>4} {'hv':>4} {'dk':>5} {'dv':>5} {'len':>8} {'fused':>10} {'baseline':>10} {'speedup':>8}")
    for config in CONFIGS:
        for length in PREFILL_LENGTHS:
            fused_data = build_prefill_inputs(config, [length])
            base_data = clone_inputs(fused_data)

            fused_ms = bench_once(
                lambda: sync_token(
                    gdn_prefill_inplace(
                        fused_data["q"], fused_data["k"], fused_data["v"], fused_data["g"],
                        fused_data["beta"], fused_data["state"], fused_data["slot_ids"],
                        fused_data["qo_indptr"],
                    ),
                    fused_data["state"],
                    fused_data["slot_ids"],
                ),
                warmup=warmup,
                repeat=repeat,
            )
            base_ms = bench_once(
                lambda: sync_token(
                    baseline_prefill(
                        base_data["q"], base_data["k"], base_data["v"], base_data["g"],
                        base_data["beta"], base_data["state"], base_data["slot_ids"],
                        base_data["qo_indptr"],
                    ),
                    base_data["state"],
                    base_data["slot_ids"],
                ),
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
    print("Decode performance vs current implementation (ms)")
    print("=" * 88)
    print(f"{'hk':>4} {'hv':>4} {'dk':>5} {'dv':>5} {'batch':>8} {'fused':>10} {'baseline':>10} {'speedup':>8}")
    for config in CONFIGS:
        for batch in DECODE_BATCHES:
            fused_data = build_decode_inputs(config, batch)
            base_data = clone_inputs(fused_data)

            fused_ms = bench_once(
                lambda: sync_token(
                    gdn_decode_inplace(
                        fused_data["q"], fused_data["k"], fused_data["v"], fused_data["g"],
                        fused_data["beta"], fused_data["state"], fused_data["slot_ids"],
                    ),
                    fused_data["state"],
                    fused_data["slot_ids"],
                ),
                warmup=warmup,
                repeat=repeat,
            )
            base_ms = bench_once(
                lambda: sync_token(
                    baseline_decode(
                        base_data["q"], base_data["k"], base_data["v"], base_data["g"],
                        base_data["beta"], base_data["state"], base_data["slot_ids"],
                    ),
                    base_data["state"],
                    base_data["slot_ids"],
                ),
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
