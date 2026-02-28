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
    ) -> mx.array:
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
            )
        else:
            assert isinstance(metadata, DecodeMetadata)
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

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        if batch.is_prefill:
            batch.attn_metadata = self._build_prefill_metadata(reqs)
        else:
            batch.attn_metadata = self._build_decode_metadata(reqs)

    def _build_prefill_metadata(self, reqs: List[Req]) -> PrefillMetadata:
        extend_lens = [req.extend_len for req in reqs]
        kv_lens = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]

        qo_indptr = _build_indptr(extend_lens)
        kv_indptr = _build_indptr(kv_lens)

        kv_indices = mx.concatenate(
            [self.page_table[req.table_idx, : req.device_len] for req in reqs]
        )
        prefix_lens = mx.array(cached_lens, dtype=mx.int32)
        positions = _make_positions(reqs)

        return PrefillMetadata(
            positions=positions,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            prefix_lens=prefix_lens,
            max_len_extend=max(extend_lens),
        )

    def _build_decode_metadata(self, reqs: List[Req]) -> DecodeMetadata:
        kv_lens = [req.device_len for req in reqs]

        kv_indptr = _build_indptr(kv_lens)
        kv_indices = mx.concatenate(
            [self.page_table[req.table_idx, : req.device_len] for req in reqs]
        )
        num_kv_splits = _compute_num_kv_splits(kv_lens)
        positions = mx.array([req.cached_len for req in reqs], dtype=mx.int32)

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


def _make_positions(reqs: List[Req]) -> mx.array:
    """Build position IDs for RoPE: arange(cached_len, device_len) per request."""
    parts = [
        mx.arange(req.cached_len, req.device_len, dtype=mx.int32) for req in reqs
    ]
    return mx.concatenate(parts)


def _compute_num_kv_splits(kv_lens: List[int]) -> mx.array:
    """Compute per-sequence KV split counts for the decode kernel's split-K reduction."""
    splits = [min(MAX_KV_SPLITS, max(1, (l + 127) // 128)) for l in kv_lens]
    return mx.array(splits, dtype=mx.int32)
