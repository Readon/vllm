# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree: Best-first tree construction for speculative decoding.

Ported from lucebox-hub/dflash/test/test_dflash.cpp (build_ddtree, 
follow_verified_tree) which was itself ported from liranringel/ddtree/ddtree.py.

DDTree builds a tree of draft token candidates from the drafter's per-position
top-K log-probability distributions. The tree is then verified in a single
target forward pass using tree attention, and the accepted path is extracted
by walking the tree following the target model's posterior.

Key parameters:
  - K: top-K candidates per position (default 5)
  - budget: maximum number of non-root tree nodes (default 10)
  - chain_seed: if True, pre-seed the top-1 chain to guarantee DDTree
    acceptance >= chain mode acceptance (default True)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import torch


@dataclass
class DDTree:
    """Flat DFS-ordered tree from draft top-K distributions.

    Slot 0 is the tree root (the bonus token from the previous spec round);
    slots 1..n_nodes are the DFS-ordered tree nodes.

    Attributes:
        n_nodes: Number of non-root tree nodes.
        token_ids: [n_nodes] token ID at each tree node.
        depths: [n_nodes] depth within the tree (1..L), root is at depth 0.
        parents: [n_nodes + 1] parent index for each slot (parents[0] = -1).
        child_maps: list of dicts mapping token_id -> child flat index.
                    Length = n_nodes + 1 (one per slot including root).
        visibility: [(1+n_nodes)^2] boolean ancestor mask (row-major).
                    visibility[i * N + j] is True iff j is an ancestor of i.
    """
    n_nodes: int = 0
    token_ids: list[int] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    child_maps: list[dict[int, int]] = field(default_factory=list)
    visibility: list[bool] = field(default_factory=list)

    @property
    def total_slots(self) -> int:
        """Total slots including root."""
        return 1 + self.n_nodes

    def to_tree_choices(self) -> list[tuple[int, ...]]:
        """Convert DDTree to vLLM's tree_choices format.

        tree_choices: list of tuples, each tuple is a path from root
        represented as child-index at each level. E.g. for a tree:
            root
           / |  \\
          A  B   C    (depth 1)
         /|  |
        D E  F        (depth 2)

        tree_choices = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0)]

        The tuples are sorted by (length, tuple) to match vLLM convention.
        """
        if self.n_nodes == 0:
            return []

        # Build a mapping: flat_index -> child_rank among siblings
        # For each node, find its rank among its parent's children
        child_ranks: dict[int, int] = {}
        for slot in range(self.total_slots):
            children = self.child_maps[slot]
            # Sort children by token_id for deterministic ordering
            sorted_tokens = sorted(children.keys())
            for rank, tok in enumerate(sorted_tokens):
                child_idx = children[tok]
                child_ranks[child_idx] = rank

        # For each non-root node, build the path from root
        tree_choices: list[tuple[int, ...]] = []
        for slot in range(1, self.total_slots):
            path: list[int] = []
            current = slot
            while current != 0:
                path.append(child_ranks[current])
                current = self.parents[current]
            path.reverse()
            tree_choices.append(tuple(path))

        # Sort by (length, tuple) to match vLLM convention
        tree_choices.sort(key=lambda t: (len(t), t))
        return tree_choices


@dataclass(order=False)
class _HeapEntry:
    """Entry in the best-first heap."""
    neg_logw: float          # negated cumulative log-prob (for max-heap via min neg)
    parent_index: int        # flat tree index of parent
    depth: int               # 1..L
    rank: int                # rank within top-K at this depth (0-indexed)
    logw: float              # actual cumulative log-prob sum

    def __lt__(self, other: _HeapEntry) -> bool:
        # Python heapq is a min-heap; we want highest logw first,
        # so compare by neg_logw ascending (= logw descending).
        return self.neg_logw < other.neg_logw


def build_ddtree(
    top_log_probs: torch.Tensor,   # [L, K] float32 (log-probabilities, sorted desc)
    top_token_ids: torch.Tensor,   # [L, K] int32 (matching token IDs)
    budget: int = 10,
    chain_seed: bool = True,
) -> DDTree:
    """Build a DDTree from per-position top-K log-prob distributions.

    Uses a best-first heap over prefixes of the per-position top-K
    distributions. Pops until `budget` nodes are accumulated.

    Args:
        top_log_probs: [L, K] log-probabilities per position, sorted descending.
        top_token_ids: [L, K] matching token IDs per position.
        budget: Maximum number of non-root tree nodes.
        chain_seed: If True, pre-seed full top-1 chain (defensive — guarantees
                    DDTree acceptance >= chain mode even with flat softmax).

    Returns:
        DDTree with the constructed tree structure.
    """
    L = top_log_probs.shape[0]
    K = top_log_probs.shape[1]

    # Work on CPU for small tensors
    if top_log_probs.is_cuda:
        top_log_probs = top_log_probs.cpu()
    if top_token_ids.is_cuda:
        top_token_ids = top_token_ids.cpu()

    log_probs = top_log_probs.tolist()
    token_ids = top_token_ids.tolist()

    tree = DDTree()
    if budget <= 0 or L <= 0:
        tree.parents = [-1]
        tree.child_maps = [{}]
        tree.visibility = [True]
        return tree

    tree.parents = [-1]          # root
    tree.child_maps = [{}]       # root's children

    heap: list[_HeapEntry] = []

    if chain_seed:
        # Pre-seed full top-1 chain: guarantees AL >= chain mode.
        chain_depth = min(L, budget)
        cum_logw = 0.0
        prev_idx = 0
        for d in range(1, chain_depth + 1):
            tok_id = token_ids[d - 1][0]
            cum_logw += log_probs[d - 1][0]

            cur_idx = tree.n_nodes + 1
            tree.token_ids.append(tok_id)
            tree.depths.append(d)
            tree.parents.append(prev_idx)
            tree.child_maps.append({})
            tree.child_maps[prev_idx][tok_id] = cur_idx
            tree.n_nodes += 1

            # Push next-best sibling at this depth into the heap.
            if K > 1:
                sibling_logw = (cum_logw
                                - log_probs[d - 1][0]
                                + log_probs[d - 1][1])
                heapq.heappush(heap, _HeapEntry(
                    neg_logw=-sibling_logw,
                    parent_index=prev_idx,
                    depth=d,
                    rank=1,
                    logw=sibling_logw,
                ))
            prev_idx = cur_idx
    else:
        # Paper-style pure best-first: seed with depth-1 top-1.
        root_logw = log_probs[0][0]
        heapq.heappush(heap, _HeapEntry(
            neg_logw=-root_logw,
            parent_index=0,
            depth=1,
            rank=0,
            logw=root_logw,
        ))

    while heap and tree.n_nodes < budget:
        top = heapq.heappop(heap)

        depth_minus_1 = top.depth - 1
        rank = top.rank
        tok_id = token_ids[depth_minus_1][rank]

        current_index = tree.n_nodes + 1
        tree.token_ids.append(tok_id)
        tree.depths.append(top.depth)
        tree.parents.append(top.parent_index)
        tree.child_maps.append({})
        tree.child_maps[top.parent_index][tok_id] = current_index
        tree.n_nodes += 1

        # Push next sibling (same depth, next-best rank at this depth).
        if rank + 1 < K:
            sibling_logw = (top.logw
                            - log_probs[depth_minus_1][rank]
                            + log_probs[depth_minus_1][rank + 1])
            heapq.heappush(heap, _HeapEntry(
                neg_logw=-sibling_logw,
                parent_index=top.parent_index,
                depth=top.depth,
                rank=rank + 1,
                logw=sibling_logw,
            ))

        # Push first child (next depth, top-1 rank under this node).
        if top.depth < L:
            child_logw = top.logw + log_probs[top.depth][0]
            heapq.heappush(heap, _HeapEntry(
                neg_logw=-child_logw,
                parent_index=current_index,
                depth=top.depth + 1,
                rank=0,
                logw=child_logw,
            ))

    # Build ancestor-only visibility mask (flat row-major, (1+n)^2).
    N = 1 + tree.n_nodes
    tree.visibility = [False] * (N * N)
    tree.visibility[0] = True   # root sees itself
    for i in range(1, N):
        p = tree.parents[i]
        # Inherit parent's visibility, then mark self.
        for j in range(i):
            tree.visibility[i * N + j] = tree.visibility[p * N + j]
        tree.visibility[i * N + i] = True

    return tree


def follow_verified_tree(
    tree: DDTree,
    posterior: torch.Tensor,  # [1+n_nodes] target argmax at each tree slot
) -> tuple[list[int], int]:
    """Walk the verified tree following the target's argmax (posterior).

    At each node, look up the target model's argmax → next_token.
    If next_token matches a child of the current node, move to that child.
    If no child matches, stop — the accepted path + bonus token are returned.

    Args:
        tree: The DDTree structure.
        posterior: [total_slots] tensor of target model's argmax token at each slot.

    Returns:
        accepted_indices: flat tree indices of the accepted path (root first).
        bonus_token: target's argmax at the deepest accepted node (next token
                     to emit that wasn't in the tree).
    """
    posterior_list = posterior.tolist()
    accepted = [0]

    current_index = 0
    next_token = int(posterior_list[0])
    while True:
        children = tree.child_maps[current_index]
        if next_token not in children:
            break
        current_index = children[next_token]
        accepted.append(current_index)
        next_token = int(posterior_list[current_index])

    return accepted, next_token


def ddtree_to_qq_bias(
    tree: DDTree,
    device: str = "cpu",
) -> torch.Tensor:
    """Build the qq_bias tensor for tree attention.

    Creates a [total_slots, total_slots] float tensor where:
      - qq_bias[i, j] = 0.0 if tree slot j is an ancestor of slot i (or j == i)
      - qq_bias[i, j] = -inf otherwise

    This is the query-to-query portion of the tree attention mask.
    When passed as qq_bias to unified_attention, it restricts each tree
    position to attend only to its ancestor nodes among the new tokens,
    while the causal mask handles past KV cache positions normally.

    Args:
        tree: The DDTree structure.
        device: Device for the output tensor.

    Returns:
        [total_slots, total_slots] float tensor.
    """
    N = tree.total_slots
    qq_bias = torch.full((N, N), float("-inf"), device=device, dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            if tree.visibility[i * N + j]:
                qq_bias[i, j] = 0.0
    return qq_bias


def ddtree_to_attn_mask(
    tree: DDTree,
    past_length: int = 0,
) -> torch.Tensor:
    """Build a tree attention mask for target model verification.

    Creates a [total_slots, past_length + total_slots] mask where:
      - Past KV cache positions are always attended (0.0)
      - Tree positions use the ancestor-only visibility mask
      - Non-ancestor tree positions are masked (-inf)

    Args:
        tree: The DDTree structure.
        past_length: Number of past KV cache positions.

    Returns:
        [total_slots, past_length + total_slots] float mask.
    """
    N = tree.total_slots
    total_kv = past_length + N
    device = "cpu"

    # Initialize all to -inf (masked out)
    mask = torch.full((N, total_kv), float("-inf"), device=device, dtype=torch.float32)

    # Past KV cache: all tree nodes can attend freely
    if past_length > 0:
        mask[:, :past_length] = 0.0

    # Tree positions: use ancestor-only visibility
    for i in range(N):
        for j in range(N):
            if tree.visibility[i * N + j]:
                mask[i, past_length + j] = 0.0

    return mask
