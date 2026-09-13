from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import mlx.core as mx
from mini_sglang_mlx.utils import cached_load_hf_config

if TYPE_CHECKING:
    from mini_sglang_mlx.models import ModelConfig


# Valid values for :attr:`EngineConfig.spec_algo`.  Kept lowercase
# internally; the CLI also accepts the conventional capitalisation
# ("DFlash") and normalises before constructing the config.
SPEC_ALGOS = ("dflash", "dflash2")


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    dtype: mx.Dtype
    max_running_req: int = 16
    attention_backend: str = "auto"
    page_size: int = 1
    kv_cache_gb: float | None = None
    max_seq_len_override: int | None = None

    # ── Speculative decoding (DFlash / DFlash2) ─────────────────────
    # ``spec_algo`` selects the algorithm; the other two fields
    # configure it:
    #
    # * ``spec_algo="dflash"`` / ``"dflash2"``: ``draft_path`` is a
    #   model directory (or remote repo ID) for a block-diffusion
    #   DFlash draft.  ``num_draft_tokens = K``; the draft block size
    #   at runtime is ``K + 1`` (verify input length per req = K + 1).
    #   DFlash2 differs only in how the block's tokens are chosen
    #   (candidate-selector path walk vs per-slot argmax).
    #
    # ``spec_algo=None`` (default) → no speculative decoding; the
    # other two fields are ignored.
    spec_algo: str | None = None
    draft_path: str | None = None
    num_draft_tokens: int = 0

    # ── Mamba state pool sizing (manual override) ───────────────────
    # When set, ``Engine._create_mamba_pool`` uses this exact slot
    # count instead of computing one from ``max_running_req`` and
    # the spec method's per-req scratch-slot footprint (spec engines
    # reserve 2 + 1 slots per req: main + radix buffer + verify
    # scratch).  Useful for hybrid models (Qwen3.5) where the
    # default heuristic may over- or under-allocate; this knob is
    # exposed for didactic purposes (the project is meant to be a
    # teaching codebase).  Must exceed ``max_running_req``: see
    # ``__post_init__``.
    num_mamba_slots: int | None = None

    def __post_init__(self) -> None:
        if self.spec_algo is not None:
            if self.spec_algo not in SPEC_ALGOS:
                raise ValueError(
                    f"spec_algo must be one of {SPEC_ALGOS} or None, "
                    f"got {self.spec_algo!r}"
                )
            if self.draft_path is None:
                raise ValueError(
                    f"spec_algo={self.spec_algo!r} requires draft_path"
                )
            if self.num_draft_tokens < 1:
                raise ValueError(
                    f"spec_algo={self.spec_algo!r} requires "
                    f"num_draft_tokens >= 1, got {self.num_draft_tokens}"
                )
        if (
            self.num_mamba_slots is not None
            and self.num_mamba_slots <= self.max_running_req
        ):
            # Every in-flight req needs its own main slot, and one more has
            # to survive alongside them: a running req can only reuse a
            # cached prefix if a snapshot is still there to fork.  At
            # exactly one slot per req the pool is always fully consumed by
            # live reqs, so mamba state would never be reused at all.
            raise ValueError(
                f"num_mamba_slots ({self.num_mamba_slots}) must be > "
                f"max_running_req ({self.max_running_req}): each in-flight "
                "req needs a slot, plus at least one for a cached prefix. "
                "Leave it unset to use the default sizing."
            )

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from mini_sglang_mlx.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:23333"
