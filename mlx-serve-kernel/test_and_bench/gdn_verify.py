"""
Correctness test for gdn_state_verify (MTP target-verify kernel).

Compares the fused kernel against a naive NumPy reference that processes
tokens one-by-one and saves the state after each token.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx_serve_kernel import gdn_state_verify


INPUT_SCALE = 0.05
ATOL = 5e-4


def to_numpy(a: mx.array) -> np.ndarray:
    if a.dtype == mx.bfloat16:
        return np.array(a.astype(mx.float32), dtype=np.float32)
    return np.array(a)


def clone(a: mx.array) -> mx.array:
    return mx.array(to_numpy(a)).astype(a.dtype)


def numpy_verify_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    slot_ids: mx.array,
) -> tuple[np.ndarray, np.ndarray]:
    """Naive reference: process tokens one-by-one, save state after each step."""
    q = to_numpy(q).astype(np.float32)
    k = to_numpy(k).astype(np.float32)
    v = to_numpy(v).astype(np.float32)
    g = to_numpy(g).astype(np.float32)
    beta = to_numpy(beta).astype(np.float32)
    state = to_numpy(state).astype(np.float32)
    slot_ids = to_numpy(slot_ids).astype(np.int32)

    batch = slot_ids.shape[0]
    num_draft = slot_ids.shape[1]
    hv_per_hk = v.shape[1] // q.shape[1]

    outputs = []
    for b in range(batch):
        base_slot = int(slot_ids[b, 0])
        s = state[base_slot].copy()
        for j in range(num_draft):
            token_idx = b * num_draft + j
            hk_per_hv = s.shape[0] // q[token_idx].shape[0]
            q_exp = np.repeat(q[token_idx], hk_per_hv, axis=0) if hk_per_hv > 1 else q[token_idx]
            k_exp = np.repeat(k[token_idx], hk_per_hv, axis=0) if hk_per_hv > 1 else k[token_idx]
            s = s * g[token_idx][:, None, None]
            kv_mem = (s * k_exp[:, None, :]).sum(axis=-1)
            delta = (v[token_idx] - kv_mem) * beta[token_idx][:, None]
            s = s + k_exp[:, None, :] * delta[:, :, None]
            outputs.append((s * q_exp[:, None, :]).sum(axis=-1))
            # Write checkpoint
            ck_slot = int(slot_ids[b, j])
            state[ck_slot] = s.copy()

    return np.stack(outputs, axis=0), state


def build_verify_inputs(
    batch: int,
    num_draft: int,
    hk: int = 16,
    hv: int = 32,
    dk: int = 128,
    dv: int = 128,
) -> dict[str, mx.array]:
    total_tokens = batch * num_draft
    num_slots = batch * num_draft + 5  # reserve some extra slots

    q = (INPUT_SCALE * mx.random.normal((total_tokens, hk, dk), dtype=mx.float32)).astype(mx.bfloat16)
    k = (INPUT_SCALE * mx.random.normal((total_tokens, hk, dk), dtype=mx.float32)).astype(mx.bfloat16)
    v = (INPUT_SCALE * mx.random.normal((total_tokens, hv, dv), dtype=mx.float32)).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((total_tokens, hv), dtype=mx.float32)).astype(mx.bfloat16)
    beta = mx.sigmoid(mx.random.normal((total_tokens, hv), dtype=mx.float32)).astype(mx.bfloat16)
    state = INPUT_SCALE * mx.random.normal(
        (num_slots + 1, hv, dv, dk), dtype=mx.float32
    )

    # slot_ids[b, 0] is the base slot; slot_ids[b, j] for j>0 are checkpoints.
    slot_ids_np = np.zeros((batch, num_draft), dtype=np.int32)
    for b in range(batch):
        base = b * num_draft + 1
        for j in range(num_draft):
            slot_ids_np[b, j] = base + j
    slot_ids = mx.array(slot_ids_np, dtype=mx.int32)

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


def assert_close(name: str, lhs: mx.array, rhs: mx.array, atol: float = ATOL) -> None:
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


def run_correctness() -> None:
    print("=" * 88)
    print("gdn_state_verify correctness")
    print("=" * 88)

    configs = [
        (16, 32, 128, 128),
        (16, 48, 128, 128),
    ]

    for hk, hv, dk, dv in configs:
        print(f"config hk={hk:>2} hv={hv:>2} dk={dk:>3} dv={dv:>3}")

        for batch, num_draft in [(1, 1), (1, 4), (4, 2), (4, 4), (8, 3)]:
            data = build_verify_inputs(batch, num_draft, hk, hv, dk, dv)
            fused_state = clone(data["state"])
            fused_y = gdn_state_verify(
                data["q"], data["k"], data["v"], data["g"], data["beta"],
                fused_state, data["slot_ids"],
            )
            ref_y, ref_state = numpy_verify_reference(
                data["q"], data["k"], data["v"], data["g"], data["beta"],
                data["state"], data["slot_ids"],
            )
            label = f"verify b={batch} d={num_draft}"
            assert_close(f"{label} output", fused_y, mx.array(ref_y))
            assert_close(f"{label} state", fused_state, mx.array(ref_state))
        print()


def main() -> None:
    mx.random.seed(42)
    np.random.seed(42)
    run_correctness()


if __name__ == "__main__":
    main()
