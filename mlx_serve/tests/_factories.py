"""Weight-free builders for the mlx_serve test suite.

The pools, trees and managers here are the production classes with toy
shapes, so the tests exercise real code paths without loading a model.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import mlx.core as mx

from mlx_serve.core import SamplingParams
from mlx_serve.kvcache.mamba_pool import MambaStateConfig, MambaStatePool
from mlx_serve.kvcache.radix_manager import RadixCacheManager, RadixTreeNode
from mlx_serve.scheduler.cache import CacheManager, HybridCacheManager
from mlx_serve.scheduler.table import TableManager
from mlx_serve.scheduler.utils import PendingReq

CONV_SHAPE = (3, 4)  # (state_len, d_inner)
TEMPORAL_SHAPE = (2, 3, 4)  # (num_heads, head_dim, state_size)
NUM_PAGES = 64


def make_pool(
    num_slots: int = 4, num_layers: int = 2, verify_width: int = 0
) -> MambaStatePool:
    return MambaStatePool(
        MambaStateConfig(
            num_slots=num_slots,
            num_layers=num_layers,
            conv_shapes=[[CONV_SHAPE] for _ in range(num_layers)],
            temporal_shapes=[TEMPORAL_SHAPE] * num_layers,
            dtype=mx.float32,
            verify_width=verify_width,
        )
    )


def make_hybrid(
    num_pages: int = NUM_PAGES, num_slots: int = 4, num_layers: int = 2
) -> Tuple[HybridCacheManager, MambaStatePool]:
    pool = make_pool(num_slots=num_slots, num_layers=num_layers)
    return HybridCacheManager(None, num_pages, pool), pool


def make_radix(num_pages: int = NUM_PAGES) -> CacheManager:
    return CacheManager(None, num_pages, "radix")


def make_table(max_running_reqs: int = 2, max_seq_len: int = 128) -> TableManager:
    return TableManager(
        max_running_reqs,
        mx.zeros((max_running_reqs, max_seq_len), dtype=mx.int32),
    )


def ids(*tokens: int) -> mx.array:
    return mx.array(list(tokens), dtype=mx.int32)


def pending(
    uid: int, tokens: Sequence[int], max_tokens: int = 3
) -> PendingReq:
    return PendingReq(uid, ids(*tokens), SamplingParams(max_tokens=max_tokens))


def cache_prefix(
    manager: HybridCacheManager,
    tokens: Sequence[int],
    mamba_slot: int | None = None,
) -> mx.array:
    """Mimic a finished request: allocate pages, then hand them to the tree.

    Mirrors ``free_and_cache_finished_req`` (including returning the pages
    of an already-cached prefix), so the page accounting stays exact and
    ``manager.check_integrity()`` remains meaningful afterwards.
    """
    token_ids = ids(*tokens)
    pages = manager.allocate(len(token_ids))
    in_cache = manager.manager.insert_prefix(
        token_ids, pages, mamba_slot=mamba_slot
    )
    manager._free(pages[:in_cache])
    return pages


def node_of(manager: CacheManager, tokens: Sequence[int]) -> RadixTreeNode:
    """The node whose key path ends exactly at ``tokens``.

    Children are keyed by the first token of their key, so the walk has to
    step a whole node at a time -- and refuse a path that ends mid-node.
    """
    node = manager.manager.root_node
    offset = 0
    while offset < len(tokens):
        node = node.children[tokens[offset]]
        assert node._key.tolist() == list(tokens[offset : offset + node.length]), (
            f"{list(tokens)} does not end on a node boundary at {offset}"
        )
        offset += node.length
    return node


def set_lru_order(manager: CacheManager, tokens: Sequence[Sequence[int]]) -> None:
    """Rewrite timestamps so eviction order is the order given here."""
    for order, path in enumerate(tokens):
        node_of(manager, path).timestamp = 1000 + order


def page_balance(manager: CacheManager) -> Tuple[int, int]:
    free_pages = len(manager._free_slots)
    held_pages = manager.manager.size_info.total_size
    assert free_pages + held_pages == manager.num_pages, (
        f"page accounting broken: {free_pages} free + {held_pages} held "
        f"!= {manager.num_pages} pages"
    )
    return free_pages, held_pages


def pool_balance(manager: HybridCacheManager, live_slots: Iterable[int] = ()) -> None:
    """Every slot is free, held by the tree, or held by a listed live req."""
    pool = manager.mamba_pool
    live = list(live_slots)
    assert len(set(live)) == len(live), "a live slot is listed twice"
    owners = set(pool._free_slots) | set(manager.manager.collect_mamba_slots())
    assert not (owners & set(live)), "a live slot is also free or in the tree"
    owned = owners | set(live)
    assert owned == set(range(1, pool.num_slots + 1)), (
        f"unowned mamba slots: {sorted(set(range(1, pool.num_slots + 1)) - owned)}"
    )


def live_slots(reqs: Iterable) -> List[int]:
    return [r.mamba_slot for r in reqs if r.mamba_slot is not None]
