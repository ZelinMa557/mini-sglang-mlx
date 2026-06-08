from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import mlx.core as mx
from mlx_serve_kernel import paged_decode_attention, paged_prefill_attention

if TYPE_CHECKING:
    from mlx_serve.core import Batch, Req
    from mlx_serve.kvcache import BaseKVCache
    from mlx_serve.models.config import ModelConfig

MAX_KV_SPLITS = 32


@dataclass
class BaseAttnMetadata:
    positions: mx.array

    def get_last_indices(self, bs: int) -> mx.array:
        raise NotImplementedError


@dataclass
class PrefillMetadata(BaseAttnMetadata):
    """Metadata for paged prefill (extend) attention.

    In prefill, each sequence has multiple new query tokens attending to
    prefix KV (cached) + its own causal tokens (extend).
    """

    qo_indptr: mx.array  # (batch + 1,) int32 — CSR row pointers for Q/O
    kv_indptr: mx.array  # (batch + 1,) int32 — CSR row pointers for KV
    kv_indices: mx.array  # (total_kv,) int32 — page indices into the KV cache
    prefix_lens: mx.array  # (batch,) int32 — prefix (cached) length per sequence
    max_len_extend: int

    def get_last_indices(self, bs: int) -> mx.array:
        return self.qo_indptr[1 : 1 + bs] - 1


@dataclass
class DecodeMetadata(BaseAttnMetadata):
    """Metadata for paged decode attention.

    In decode, each sequence has exactly 1 new query token attending to
    all historical KV tokens via split-K reduction.
    """

    kv_indptr: mx.array  # (batch + 1,) int32 — CSR row pointers for KV
    kv_indices: mx.array  # (total_kv,) int32 — page indices into the KV cache
    num_kv_splits: mx.array  # (batch,) int32 — number of KV splits per sequence
    max_kv_splits: int

    def get_last_indices(self, bs: int) -> mx.array:
        return mx.arange(bs, dtype=mx.int32)


AttnMetadata = PrefillMetadata | DecodeMetadata


class AttnBackend:
    """Paged attention backend for Apple GPU using mlx-serve-kernel operators."""

    def __init__(
        self,
        config: ModelConfig,
        kvcache: BaseKVCache,
        page_table: mx.array,
    ) -> None:
        self.config = config
        self.kvcache = kvcache
        self.page_table = page_table
        self.sm_scale = 1.0 / math.sqrt(config.head_dim)

    def forward(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        layer_id: int,
        batch: Batch,
        *,
        is_cross_attention: bool = False,
        sliding_window_size: int = 0,
    ) -> mx.array:
        """Write proposal K/V and run paged attention.

        Args:
            q / k / v: per-token query / key / value tensors
                shaped ``[N, n_heads_or_kv, head_dim]``.
            layer_id: which layer's KV buffer to write into.
            batch: carries ``out_loc`` (where to write K/V) and
                ``attn_metadata`` (the per-batch index tensors).
            is_cross_attention: when ``True``, **drop the causal
                upper bound within the extend window** — query at
                extend offset ``p`` attends to every other extend
                key, in addition to the cached prefix.  Used by
                DFlash full-attention layers, where the masked
                block tokens predict each other bidirectionally.
                Has no effect on the decode path.
            sliding_window_size: when ``> 0``, restrict each query
                to keys whose absolute target-sequence position is
                within ``W`` of (and ≤, when causal) the query's
                position.  Used by DFlash sliding-attention layers
                (``is_cross_attention=False, sliding_window_size=W``).
                ``0`` means "no window".

            The two flags are **mutually exclusive** — either drop
            the causal upper bound (``is_cross_attention=True``) or
            restrict the lower bound to a sliding window
            (``sliding_window_size > 0``), but not both.  The three
            valid combinations are:

            =====================  =====================  =====================
            is_cross_attention     sliding_window_size    mask
            =====================  =====================  =====================
            ``False``              ``0``                  causal (default)
            ``True``               ``0``                  full / bidirectional
            ``False``              ``W > 0``              causal + window
            =====================  =====================  =====================

            Passing both is rejected (the kernel is allowed to
            assume it never happens).

        ``is_cross_attention`` / ``sliding_window_size`` are forwarded
        to :func:`mlx_serve_kernel.paged_prefill_attention` and are
        currently rejected by the decode path (single-query decode
        doesn't need mask variants).
        """
        assert sliding_window_size >= 0, (
            f"sliding_window_size must be >= 0, got {sliding_window_size}"
        )
        assert not (is_cross_attention and sliding_window_size > 0), (
            "is_cross_attention=True and sliding_window_size>0 are "
            "mutually exclusive."
        )

        metadata = batch.attn_metadata
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)

        if isinstance(metadata, PrefillMetadata):
            return paged_prefill_attention(
                q,
                k_cache,
                v_cache,
                metadata.qo_indptr,
                metadata.kv_indptr,
                metadata.kv_indices,
                metadata.prefix_lens,
                sm_scale=self.sm_scale,
                max_len_extend=metadata.max_len_extend,
                is_cross_attention=is_cross_attention,
                sliding_window_size=sliding_window_size,
            )
        else:
            assert isinstance(metadata, DecodeMetadata)
            assert not is_cross_attention and sliding_window_size == 0, (
                "is_cross_attention / sliding_window_size are only "
                "supported on the prefill path."
            )
            return paged_decode_attention(
                q,
                k_cache,
                v_cache,
                metadata.kv_indptr,
                metadata.kv_indices,
                metadata.num_kv_splits,
                sm_scale=self.sm_scale,
                max_kv_splits=metadata.max_kv_splits,
            )

    def store_kv_only(
        self,
        k: mx.array,
        v: mx.array,
        out_loc: mx.array,
        layer_id: int,
    ) -> None:
        """Write K/V into the paged cache without running attention.

        Used by spec engines whose draft layers need to "calibrate"
        cached K/V from freshly-arrived target hidden states (DFlash
        projects target hidden through each draft layer's k_proj /
        v_proj and writes it at the matching page IDs).  Differs from
        :meth:`forward` only in skipping the kernel call — the write
        path is the same.

        Args:
            k: ``[N, num_kv_heads, head_dim]`` keys.
            v: ``[N, num_kv_heads, head_dim]`` values.
            out_loc: ``[N]`` int32 — per-token page IDs to write at.
            layer_id: which layer's KV buffer to write into.
        """
        self.kvcache.store_kv(k, v, out_loc, layer_id)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        if batch.is_prefill:
            batch.attn_metadata = self.build_prefill_metadata(reqs)
        else:
            batch.attn_metadata = self.build_decode_metadata(reqs)

    def build_prefill_metadata(
        self,
        reqs: List[Req],
        *,
        extend_lens: List[int] | None = None,
        kv_lens: List[int] | None = None,
        positions: mx.array | None = None,
    ) -> PrefillMetadata:
        """Build paged prefill metadata.

        Defaults are inferred from each ``req``'s ``cached_len`` /
        ``device_len`` / ``extend_len``.  Callers that need to drive a
        forward pass over a *virtual* extend window (e.g. MTP target
        verify, which evaluates ``K+1`` positions per req while the
        req's own ``extend_len`` is still ``1``) can override
        ``extend_lens`` and ``kv_lens``; ``positions`` falls back to
        ``arange(cached, kv)`` per req where
        ``cached = kv_lens[i] - extend_lens[i]``.
        """
        if extend_lens is None:
            extend_lens = [req.extend_len for req in reqs]
        if kv_lens is None:
            kv_lens = [req.device_len for req in reqs]
        cached_lens = [kv_lens[i] - extend_lens[i] for i in range(len(reqs))]
        if positions is None:
            positions = mx.concatenate(
                [
                    mx.arange(cached_lens[i], kv_lens[i], dtype=mx.int32)
                    for i in range(len(reqs))
                ]
            )

        qo_indptr = _build_indptr(extend_lens)
        kv_indptr = _build_indptr(kv_lens)
        kv_indices = mx.concatenate(
            [
                self.page_table[reqs[i].table_idx, : kv_lens[i]]
                for i in range(len(reqs))
            ]
        )
        prefix_lens = mx.array(cached_lens, dtype=mx.int32)

        return PrefillMetadata(
            positions=positions,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            prefix_lens=prefix_lens,
            max_len_extend=max(extend_lens),
        )

    def build_decode_metadata(
        self,
        reqs: List[Req],
        *,
        kv_lens: List[int] | None = None,
        positions: mx.array | None = None,
    ) -> DecodeMetadata:
        """Build paged decode metadata.

        Defaults are inferred from each ``req`` — ``kv_lens =
        req.device_len`` and ``positions = req.cached_len``.  Callers
        that step the decode loop manually (e.g. MTP draft regular /
        bonus steps, where the req hasn't been advanced yet) can
        override either field; both apply per-req as ``[B]``.
        """
        if kv_lens is None:
            kv_lens = [req.device_len for req in reqs]
        if positions is None:
            positions = mx.array(
                [req.cached_len for req in reqs], dtype=mx.int32
            )

        kv_indptr = _build_indptr(kv_lens)
        kv_indices = mx.concatenate(
            [
                self.page_table[reqs[i].table_idx, : kv_lens[i]]
                for i in range(len(reqs))
            ]
        )
        num_kv_splits = _compute_num_kv_splits(kv_lens)

        return DecodeMetadata(
            positions=positions,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            num_kv_splits=num_kv_splits,
            max_kv_splits=MAX_KV_SPLITS,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_indptr(lens: List[int]) -> mx.array:
    """Build CSR indptr array from a list of lengths."""
    return mx.cumsum(mx.array([0] + lens, dtype=mx.int32))


def _compute_num_kv_splits(kv_lens: List[int]) -> mx.array:
    """Compute per-sequence KV split counts for the decode kernel's split-K reduction."""
    splits = [min(MAX_KV_SPLITS, max(1, (l + 127) // 128)) for l in kv_lens]
    return mx.array(splits, dtype=mx.int32)
