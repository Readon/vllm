# MTP + DDTree Design for Qwen3.5-27B-AWQ on SM75

Date: 2025-05-03
Status: Design Phase
Hardware: 2× RTX 2080 Ti (sm_75, 22 GiB each)

---

## 1. Problem Statement

Current MTP chain verification achieves 76.1 tok/s (1.83× over 41.6 tok/s baseline).
Chain verification accepts ~2.17 tokens per step; when a draft token is rejected,
**all subsequent draft tokens are discarded**.

DDTree builds a tree of draft candidates and verifies the entire tree in a single
target forward pass. Even though some branches are rejected, other branches may
still be accepted, increasing average tokens accepted per step.

### Expected improvement

- Chain verify: ~2.17 accepted/step (72.4% acceptance rate)
- DDTree (budget=10): ~2.8-3.2 accepted/step (85-90% acceptance rate)
- Estimated throughput: 85-95 tok/s (~2.0-2.3× baseline)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: MTP Draft (modified)                             │
│                                                          │
│  Target forward → hidden_states + logits                 │
│  MTP layer 0 forward → logits_0 (top-K: K=5)            │
│  MTP layer 1 forward → logits_1 (top-K: K=5)            │
│  MTP layer 2 forward → logits_2 (top-K: K=5)            │
│                                                          │
│  For each MTP step, extract top-K log-probs + token_ids  │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: DDTree Builder (new)                             │
│                                                          │
│  Input: top_log_probs [L × K], top_token_ids [L × K]    │
│  Algorithm: best-first heap, budget=10 nodes              │
│  Output: DDTree struct (token_ids, parents, depths,      │
│          visibility mask, child_maps)                    │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Tree Verification (modified)                     │
│                                                          │
│  Flatten tree tokens → sequence                          │
│  Build tree attention mask (ancestor-only)               │
│  Target model forward (1 pass, tree-structured attn)     │
│  Walk tree following target's argmax at each node        │
│  Extract accepted path + bonus token                     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 DDTree Builder

**New file**: `vllm/v1/spec_decode/ddtree.py`

Pure-Python port of `lucebox-hub/dflash/test/test_dflash.cpp` lines 143-420.

```python
@dataclass
class DDTree:
    """Flat DFS-ordered tree from draft top-K distributions."""
    n_nodes: int                              # excludes root
    token_ids: torch.Tensor                   # [n_nodes] int32
    depths: torch.Tensor                      # [n_nodes] int32
    parents: torch.Tensor                     # [n_nodes+1] int32 (root at index 0)
    child_maps: list[dict[int, int]]          # token_id → child flat index
    visibility: torch.Tensor                  # [(1+n_nodes)^2] bool (ancestor mask)


def build_ddtree(
    top_log_probs: torch.Tensor,  # [L, K] float32
    top_token_ids: torch.Tensor,  # [L, K] int32
    budget: int = 10,
    chain_seed: bool = True,
) -> DDTree:
    """
    Best-first heap construction of DDTree.
    
    Algorithm:
    1. If chain_seed=True: pre-seed the top-1 chain (guarantees ≥ chain mode)
    2. Push siblings and children into max-heap keyed by cumulative log-prob
    3. Pop highest-prob entry, add to tree, push its sibling + first child
    4. Repeat until budget reached
    5. Build ancestor visibility mask for tree attention
    
    Runs on CPU (small tensors, L≤5, K≤5, budget≤10).
    """
```

Key parameters:
- **K=5**: top-5 per position (enough diversity without excessive tree width)
- **budget=10**: max tree nodes (fits in 22 GiB with room for KV cache)
- **chain_seed=True**: defensive — guarantees DDTree ≥ chain mode performance
- **temperature=1.0**: no softmax sharpening needed (MTP logits are full precision, unlike Q4_K_M GGUF)

### 3.2 MTP Top-K Logit Extraction

**Modified file**: `vllm/model_executor/models/qwen3_5_mtp.py`

Currently MTP runs L sequential forward passes through MTP layers, each producing
hidden states that feed the next layer. We need to also extract logits at each step.

```python
class Qwen3_5MultiTokenPredictor:
    def forward(self, ...):
        # Existing code runs MTP layers sequentially:
        #   hidden → MTP layer 0 → hidden_0 → MTP layer 1 → hidden_1 → ...
        #
        # NEW: At each step, compute logits and extract top-K
        # Store (top_log_probs, top_token_ids) for DDTree builder
        
        top_log_probs_list = []
        top_token_ids_list = []
        
        for step_idx in range(self.num_next_token_predict):
            # ... existing MTP layer forward ...
            logits = self.lm_head(hidden)  # [batch, vocab]
            
            # Extract top-K log-probabilities
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            topk_vals, topk_ids = torch.topk(log_probs, k=K, dim=-1)
            # topk_vals: [batch, K], topk_ids: [batch, K]
            
            top_log_probs_list.append(topk_vals)
            top_token_ids_list.append(topk_ids)
        
        return draft_token_ids, (top_log_probs, top_token_ids)
```

Design notes:
- Top-K extraction adds minimal overhead (log_softmax + topk on small batch)
- K=5 means 5 float values per position, negligible memory
- The argmax (top-1) path is the chain mode fallback

### 3.3 Dynamic Tree Attention

**Challenge**: vLLM's existing `TreeAttentionBackend` expects a **static** tree
defined by `speculative_token_tree` at initialization time. DDTree generates a
**dynamic** tree per speculative step.

**Solution**: Create a `DynamicTreeAttentionMetadataBuilder` that accepts a
per-step tree structure.

**New file**: `vllm/v1/attention/backends/dynamic_tree_attn.py`

Or alternatively, modify the existing `TreeAttentionMetadataBuilder` to support
runtime tree updates.

```python
class DynamicTreeAttentionMetadataBuilder(TreeAttentionMetadataBuilder):
    """Extended tree attention builder that supports per-step tree structures."""
    
    def update_tree(self, tree: DDTree):
        """Update the tree attention bias for a new DDTree structure."""
        # Convert DDTree to tree_choices format
        tree_choices = ddtree_to_tree_choices(tree)
        depth_counts = _get_depth_counts(tree_choices)
        self.tree_attn_bias = _prepare_tree_attn_bias(
            tree_choices, depth_counts, 
            dtype=torch.float32, device=self.device
        )
        self.reorder_batch_threshold = self.tree_attn_bias.shape[0]
```

Conversion function:
```python
def ddtree_to_tree_choices(tree: DDTree) -> list[tuple[int, ...]]:
    """Convert DDTree flat structure to tree_choices format.
    
    tree_choices: list of paths from root, e.g. [(0,), (0,0), (0,1), (0,0,0)]
    Each tuple represents the child indices at each level.
    """
    # DFS traversal of DDTree, building path tuples
    # ...
```

### 3.4 Tree Verification (Rejection Sampler Extension)

**Modified file**: `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`

Current chain verification:
```python
# Sequential: accept until first rejection
for i in range(num_tokens - 1):
    if target[i] != draft[i+1]:
        rejected = True  # stop here
```

DDTree verification:
```python
def ddtree_verify(
    tree: DDTree,
    target_posterior: torch.Tensor,  # [1+n_nodes] argmax at each tree node
) -> tuple[list[int], int]:
    """
    Walk verified tree following target's argmax.
    
    Algorithm:
    1. Start at root (index 0)
    2. Look up target's argmax at current node → next_token
    3. Check if next_token matches any child → move to that child
    4. If no child matches → stop, return accepted path + bonus token
    
    Returns:
      accepted_indices: list of flat tree indices (root first)
      bonus_token: target's argmax at deepest accepted node
    """
    accepted = [0]
    current = 0
    next_token = target_posterior[0].item()
    
    while True:
        children = tree.child_maps[current]
        if next_token not in children:
            break
        current = children[next_token]
        accepted.append(current)
        next_token = target_posterior[current].item()
    
    return accepted, next_token
```

### 3.5 Integration with MTP Speculative Decoding

**Key insight**: MTP in vLLM currently uses `method="eagle"` internally when
tree attention is needed, or sequential passes when not. The DDTree integration
hooks into the existing speculative decoding pipeline.

**Configuration**:
```python
speculative_config = {
    'method': 'mtp',
    'num_speculative_tokens': 3,
    'ddtree_budget': 10,       # new param
    'ddtree_topk': 5,           # new param  
    'ddtree_chain_seed': True,  # new param
}
```

**Modified file**: `vllm/config/speculative.py`
- Add `ddtree_budget`, `ddtree_topk`, `ddtree_chain_seed` fields

---

## 4. Execution Flow

### 4.1 Draft Phase

```
1. Target model decode forward → logits, hidden_states
2. Sample token from logits (normal decode)
3. For each MTP step (0..L-1):
   a. Run MTP layer forward → hidden_states_i
   b. Compute logits_i = lm_head(hidden_states_i)
   c. Extract top-K: (top_log_probs[i], top_token_ids[i])
   d. Chain draft token = argmax(logits_i)
4. Return: chain_draft_tokens (for fallback), top_K_distributions (for DDTree)
```

### 4.2 Tree Build Phase

```
1. Collect (top_log_probs, top_token_ids) from all MTP steps
2. Call DDTree builder on CPU (fast, small tensors):
   tree = build_ddtree(top_log_probs, top_token_ids, budget=10)
3. Convert DDTree → tree_choices format for vLLM tree attention
4. Flatten tree tokens into sequence for target model forward
5. Update tree attention bias
```

### 4.3 Verify Phase

```
1. Target model forward with tree attention mask:
   - Input: flattened tree tokens [root, node_1, node_2, ...]
   - Attention: ancestor-only mask (tree structure)
   - Output: logits at each tree node position
2. Compute argmax at each tree node → posterior[i]
3. Walk tree: follow_verified_tree(tree, posterior)
   → accepted_indices, bonus_token
4. Return: accepted tokens + bonus token
```

---

## 5. Memory Budget Analysis

Current MTP spec=3 with chain verification:
- Model weights: ~10 GiB/GPU (AWQ 4-bit)
- KV cache (gpu_mem=0.85): ~8.5 GiB/GPU
- CUDA graphs + workspace: ~2.5 GiB/GPU
- Total: ~21 GiB/GPU ✓ (fits in 22 GiB)

DDTree budget=10 adds:
- Tree tokens: 10 × 2 bytes (fp16) = 20 bytes (negligible)
- Tree attention mask: 11² × 4 bytes = 484 bytes (negligible)
- Extra KV cache slots for tree tokens: 10 × (hidden_size/head_dim × kv_heads × 2)
  = 10 × 128 × 4 × 2 = 10 KiB per layer × 64 layers = 640 KiB (negligible)
- Target forward for 11 tokens (vs 4 for chain): slightly more activation memory
  but still within existing workspace allocation

**Conclusion**: DDTree adds negligible memory overhead. The tree verification
uses the same target forward pass — just with 11 tokens instead of 4.

---

## 6. Implementation Plan

### Phase 1: DDTree Builder (1-2 hours)
- [ ] Create `vllm/v1/spec_decode/ddtree.py` with `build_ddtree()` and `DDTree` dataclass
- [ ] Port from C++ (lucebox-hub) to Python/PyTorch
- [ ] Unit test with synthetic log-prob distributions
- [ ] Verify tree structure matches C++ implementation

### Phase 2: MTP Logit Extraction (1-2 hours)
- [ ] Modify `qwen3_5_mtp.py` to optionally return top-K log-probs
- [ ] Add config flags for DDTree mode
- [ ] Test that logit extraction doesn't break existing MTP chain mode

### Phase 3: Tree Verification Integration (2-3 hours)
- [ ] Create `ddtree_verify()` in rejection_sampler or new file
- [ ] Hook into speculative decoding pipeline
- [ ] Convert DDTree → tree_choices for tree attention
- [ ] Dynamic tree attention bias update

### Phase 4: End-to-End Testing (2-3 hours)
- [ ] Test with `enforce_eager=True` first (debugging)
- [ ] Test with CUDA graphs
- [ ] Benchmark: compare DDTree vs chain on MTP spec=3
- [ ] Tune budget (5, 10, 15, 20) and K (3, 5, 7)

### Phase 5: Optimization (if needed)
- [ ] GPU-parallelize DDTree builder if CPU becomes bottleneck
- [ ] Cache tree structure patterns (common trees)
- [ ] Tune temperature parameter

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DDTree builder too slow on CPU | Low | L=3, K=5, budget=10 → ~50 iterations, <1ms |
| Tree attention mask incompatible with TRITON_ATTN | Medium | May need to fall back to manual mask construction |
| CUDA graph capture fails with dynamic tree | High | **Key risk** — cudagraphs require static shapes. Need fixed max tree size with padding. |
| Acceptance rate doesn't improve enough | Low | chain_seed=True guarantees ≥ chain mode performance |

### Critical Risk: CUDA Graphs with Dynamic Trees

CUDA graphs require **static input shapes**. DDTree generates a different tree
structure each step (different number of tokens, different attention mask).

**Solution**: Fixed-size padded tree.
- Pad tree to `budget` nodes + 1 root = budget+1 total tokens
- Use fixed-size attention mask (budget+1)²
- Unused positions get masked out with -inf in attention mask
- This makes the tree verification CUDA-graph compatible

Example with budget=10:
```
Tree slot count: always 11 (root + 10 potential nodes)
Actual nodes: 7 → positions 8,9,10 are padding (masked out)
Attention mask: 11×11, padding rows/cols = -inf
```

This matches how EAGLE's static tree already works in vLLM — fixed shape,
variable content.

---

## 8. Success Criteria

- [ ] DDTree with MTP spec=3 ≥ 76.1 tok/s (current chain mode)
- [ ] Target: ≥ 85 tok/s (>10% improvement over chain mode)
- [ ] No regression in baseline (41.6 tok/s without speculation)
- [ ] Works with CUDA graphs (no enforce_eager)
- [ ] Works with TRITON_ATTN backend on sm_75
