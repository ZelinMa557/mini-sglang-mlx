from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import mlx.core as mx

from mini_sglang_mlx_kernel import fast_compare_key

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo

if TYPE_CHECKING:
    from .mamba_pool import MambaStatePool


class RadixTreeNode:
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # mamba state: slot index into MambaStatePool, or None for
        # pure-attention models / interior nodes after a split ("tombstone").
        self.mamba_slot: int | None = None

        # these fields should be updated later
        self._key: mx.array
        self._value: mx.array
        self._length: int

    def set_key_value(self, key: mx.array, value: mx.array) -> None:
        assert len(key) == len(value)
        self._key = mx.contiguous(key)
        self._value = mx.contiguous(value)
        mx.eval(self._key, self._value)
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[int(self._key[0].item())] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> mx.array:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: mx.array) -> int:
        input_ids = mx.contiguous(input_ids)
        mx.eval(input_ids)
        return fast_compare_key(self._key, input_ids)

    def _split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        # mamba state is not splittable — it stays on the child (self).
        # The new parent node becomes a tombstone (mamba_slot = None).

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode


class RadixCacheManager(BaseCacheManager):
    def __init__(self, device: None = None):
        del device  # MLX runs on Apple Silicon, no explicit device
        self.empty_tensor = mx.array([], dtype=mx.int32)
        super().__init__()
        self.root_node = RadixTreeNode()
        self.root_node.ref_count = 1  # root is always protected
        self.evictable_size = 0
        self.protected_size = 0

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, (RadixCacheHandle, HybridCacheHandle))
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: mx.array) -> Tuple[RadixCacheHandle, mx.array]:
        node, prefix_len = self._walk(input_ids)
        if prefix_len == 0:
            assert node.is_root() and node is self.root_node and prefix_len == 0
            return RadixCacheHandle(prefix_len, node), self.empty_tensor
        value_list: List[mx.array] = []
        matched_node = node
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return RadixCacheHandle(prefix_len, matched_node), mx.concatenate(value_list)

    def insert_prefix(self, input_ids: mx.array, indices: mx.array) -> int:
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        if prefix_len < len(input_ids):
            new_node = RadixTreeNode()
            new_node.set_key_value(
                input_ids[prefix_len:], mx.reshape(indices[prefix_len:], (-1,))
            )
            new_node.set_parent(node)
            self.evictable_size += new_node.length
        return prefix_len

    def _walk(self, input_ids: mx.array) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len

            node = node.children[this_id]

            # NOTE: at least 1 char is matched, so match_len >= 1
            match_len = node.get_match_len(
                mx.reshape(input_ids[prefix_len:], (-1,))
            )
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len

    def _release_node_state(self, node: RadixTreeNode) -> int:
        """Hook: drop whatever the node holds besides its KV pages.

        Subclasses that keep per-node state (mamba snapshots) free it here
        and return how many of the eviction target's units that accounted
        for.  The base tree holds nothing but pages.
        """
        del node
        return 0

    def _evict_leaves(
        self, size: int = 0, num_slots: int = 0
    ) -> Tuple[mx.array, int]:
        """Evict least-recently-used leaves until both targets are met.

        Callers pass whichever target they are short of and 0 for the other.
        Leaves are taken in LRU order regardless of which target is being
        chased: skipping slot-less leaves to reach a slot sooner would let a
        colder node outlive a hotter one.

        Returns the evicted page indices and the number of mamba slots freed.
        """
        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[mx.array] = []
        evicted_size = 0
        freed_slots = 0

        while evicted_size < size or freed_slots < num_slots:
            if not leave_nodes:
                # The page target is backed by `size <= evictable_size`, so
                # only a slot-chasing caller can run the tree dry -- and
                # there it takes whatever was reclaimed.
                assert evicted_size >= size, (
                    f"Cannot evict enough cache, need {size}, "
                    f"only {evicted_size} evicted"
                )
                break
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            freed_slots += self._release_node_state(node)

            parent = node.parent
            del parent.children[int(node._key[0].item())]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        indices = (
            mx.concatenate(evicted_indices)
            if evicted_indices
            else self.empty_tensor
        )
        return indices, freed_slots

    def evict(self, size: int) -> mx.array:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        indices, _ = self._evict_leaves(size=size)
        return indices

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass


# ── Hybrid Mamba-Attention support ──────────────────────────────────────────


@dataclass(frozen=True)
class HybridCacheHandle(BaseCacheHandle):
    """Cache handle for hybrid models.

    ``node`` is the deepest matched node that carries a mamba snapshot, so
    ``node.mamba_slot`` is the state a request admitted on this handle must
    fork before it can run.  ``cached_len > 0`` iff such a snapshot exists
    (``cached_len == 0`` returns the root), which keeps the snapshot a single
    source of truth instead of a copy that can go stale.

    The request's own slot is *not* here: forking it mutates the pool, and a
    query must not.  ``CacheManager.acquire_mamba_slot`` hands it out at
    admission time, where a rejection can still give it back.
    """

    node: RadixTreeNode


class HybridRadixCacheManager(RadixCacheManager):
    """Radix cache that co-manages KV page indices and Mamba state slots.

    For hybrid Mamba-Attention models, a radix node may carry a
    ``mamba_slot`` holding the recurrent state at that node's end boundary.
    A request that matches such a node forks the state (a copy -- the node's
    own slot must stay intact for the next hit) and continues from there.

    Because recurrent state cannot be split or trimmed, a request can only
    reuse a prefix whose *end* has a snapshot: :meth:`_find_mamba_ancestor`
    therefore walks *up* from the deepest KV match, and the length it reuses
    is that ancestor's depth, even when more KV matched below it.
    """

    def __init__(self, mamba_pool: MambaStatePool, device: None = None) -> None:
        super().__init__(device=device)
        self.mamba_pool = mamba_pool

    # ── prefix matching ─────────────────────────────────────────────────

    def match_prefix(
        self, input_ids: mx.array
    ) -> Tuple[HybridCacheHandle, mx.array]:
        """Match KV *and* report which mamba snapshot to fork.

        Deliberately does not touch the mamba pool: forking is a mutation,
        and the base contract says ``match_prefix`` must not modify the
        cache.  Doing it here would also mean every caller that ends up
        rejecting the request has to know it must give a slot back -- which
        is exactly how slots used to leak.
        """
        node, prefix_len = self._walk(input_ids)

        mamba_node, mamba_depth = self._find_mamba_ancestor(node, prefix_len)
        if mamba_node.mamba_slot is None or mamba_depth == 0:
            return HybridCacheHandle(0, self.root_node), self.empty_tensor

        value_list: List[mx.array] = []
        walk = mamba_node
        while not walk.is_root():
            value_list.append(walk.value)
            walk = walk.parent
        value_list.reverse()
        return (
            HybridCacheHandle(mamba_depth, mamba_node),
            mx.concatenate(value_list),
        )

    def _find_mamba_ancestor(
        self, node: RadixTreeNode, depth: int
    ) -> Tuple[RadixTreeNode, int]:
        """Walk up from *node* to find the nearest ancestor with mamba state."""
        cur = node
        cur_depth = depth
        while not cur.is_root():
            if cur.mamba_slot is not None:
                return cur, cur_depth
            cur_depth -= cur.length
            cur = cur.parent
        return self.root_node, 0

    # ── insertion ────────────────────────────────────────────────────────

    def insert_prefix(
        self,
        input_ids: mx.array,
        indices: mx.array,
        mamba_slot: int | None = None,
    ) -> int:
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        if prefix_len < len(input_ids):
            new_node = RadixTreeNode()
            new_node.set_key_value(
                input_ids[prefix_len:], mx.reshape(indices[prefix_len:], (-1,))
            )
            new_node.set_parent(node)
            new_node.mamba_slot = mamba_slot
            self.evictable_size += new_node.length
        else:
            # The whole sequence is already cached.  Adopting the slot here
            # is fine even if the node is locked by a live request: that
            # request forked its own copy at admission and never reads this
            # snapshot again.  The root is the exception -- eviction only
            # ever removes leaves, so a snapshot there could never be
            # reclaimed.
            assert not (node.is_root() and mamba_slot is not None), (
                "cannot store a snapshot on the root"
            )
            if node.mamba_slot is None and mamba_slot is not None:
                node.mamba_slot = mamba_slot
            elif mamba_slot is not None:
                self.mamba_pool.free(mamba_slot)
        return prefix_len

    # ── eviction ─────────────────────────────────────────────────────────

    def _release_node_state(self, node: RadixTreeNode) -> int:
        if node.mamba_slot is None:
            return 0
        self.mamba_pool.free(node.mamba_slot)
        node.mamba_slot = None
        return 1

    def evict_for_mamba(self, count: int) -> Tuple[mx.array, int]:
        """Evict LRU prefixes until *count* mamba slots come back.

        The mamba pool runs dry independently of the KV budget, so without
        this the pool can reach empty while the tree still holds plenty of
        reclaimable snapshots -- and nothing would ever trigger a reclaim.
        Treating "pool dry" as cache pressure makes the pool's capacity an
        upper bound on how many snapshots the tree can hold, which is
        exactly what keeps it from draining permanently.

        Returns the evicted page indices and how many slots were freed; the
        latter is short of *count* once there is nothing left to evict.
        """
        return self._evict_leaves(num_slots=count)

    def collect_mamba_slots(self) -> List[int]:
        """Every mamba slot currently held by a cached prefix."""
        slots: List[int] = []
        stack: List[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            if node.mamba_slot is not None:
                slots.append(node.mamba_slot)
            stack.extend(node.children.values())
        return slots

    def coldest_snapshot_node(
        self, exclude: RadixTreeNode | None = None
    ) -> RadixTreeNode | None:
        """The least-recently-used node holding a mamba snapshot.

        ``exclude`` skips one node -- the one a request is about to fork
        from, which must keep its snapshot until the copy is done.  Its
        children are still searched.
        """
        best: RadixTreeNode | None = None
        stack: List[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            if node.mamba_slot is not None and node is not exclude:
                if best is None or node.timestamp < best.timestamp:
                    best = node
            stack.extend(node.children.values())
        return best

    def check_integrity(self) -> None:
        super().check_integrity()
        # Only meaningful while no request runs: a live request's slot is
        # owned by neither the free list nor the tree, and would look leaked.
        self.mamba_pool.check_ownership(self.collect_mamba_slots())
