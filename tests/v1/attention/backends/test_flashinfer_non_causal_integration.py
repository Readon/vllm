# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test for FlashInfer non-causal attention support (DFlash scenario).

This test verifies that the FlashInfer backend correctly handles non-causal
attention end-to-end, which is required for DFlash speculative decoding.
"""
import torch

from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    FlashInferBackend,
    FlashInferMetadata,
)


def test_backend_declares_non_causal_support():
    """FlashInferBackend must declare supports_non_causal() == True."""
    assert FlashInferBackend.supports_non_causal() is True, (
        "FlashInferBackend must declare non-causal support"
    )


def test_metadata_causal_field_defaults_to_true():
    """FlashInferMetadata.causal should default to True for backward compat."""
    metadata = FlashInferMetadata(
        num_actual_tokens=10,
        slot_mapping=torch.arange(10),
        q_data_type=torch.bfloat16,
        num_decodes=5,
        num_decode_tokens=5,
        num_prefills=2,
        num_prefill_tokens=5,
        use_cascade=False,
        prefill=None,
        decode=None,
        cascade_wrapper=None,
    )
    assert hasattr(metadata, "causal"), "Missing causal field"
    assert metadata.causal is True, "Default must be True"


def test_metadata_causal_can_be_set_to_false():
    """FlashInferMetadata.causal must accept False for DFlash."""
    metadata = FlashInferMetadata(
        num_actual_tokens=10,
        slot_mapping=torch.arange(10),
        q_data_type=torch.bfloat16,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=10,
        use_cascade=False,
        prefill=None,
        decode=None,
        cascade_wrapper=None,
        causal=False,
    )
    assert metadata.causal is False


def test_dflash_getattr_pattern():
    """Verify the getattr pattern used in dflash.py works correctly.

    dflash.py:268 uses:
        assert getattr(attn_metadata, "causal", None) is False
    This must pass when causal=False and fail when causal=True.
    """
    # Non-causal (DFlash case) - assertion should pass
    nc = FlashInferMetadata(
        num_actual_tokens=10,
        slot_mapping=torch.arange(10),
        q_data_type=torch.bfloat16,
        num_decodes=0, num_decode_tokens=0,
        num_prefills=1, num_prefill_tokens=10,
        use_cascade=False, prefill=None, decode=None,
        cascade_wrapper=None, causal=False,
    )
    assert getattr(nc, "causal", None) is False, (
        "DFlash assertion 'getattr(attn_metadata, causal, None) is False' "
        "must pass for non-causal metadata"
    )

    # Causal (normal case) - assertion should fail
    c = FlashInferMetadata(
        num_actual_tokens=10,
        slot_mapping=torch.arange(10),
        q_data_type=torch.bfloat16,
        num_decodes=0, num_decode_tokens=0,
        num_prefills=1, num_prefill_tokens=10,
        use_cascade=False, prefill=None, decode=None,
        cascade_wrapper=None, causal=True,
    )
    assert getattr(c, "causal", None) is not False, (
        "DFlash assertion should NOT pass for causal metadata"
    )


def test_wrapper_plan_respects_causal_false():
    """Verify BatchPrefillWithPagedKVCacheWrapper.plan() respects causal=False."""
    if not torch.cuda.is_available():
        return  # Skip if no GPU

    workspace = torch.zeros(2048 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "HND")

    # Minimal plan with causal=False
    qo_indptr = torch.tensor([0, 4], dtype=torch.int32, device="cpu")
    paged_kv_indptr = torch.tensor([0, 2], dtype=torch.int32, device="cpu")
    paged_kv_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor([4], dtype=torch.int32, device="cpu")

    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=64,
        page_size=4,
        causal=False,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        o_data_type=torch.float16,
    )

    assert wrapper._causal is False, (
        f"Wrapper should have _causal=False after plan(causal=False), "
        f"got {wrapper._causal}"
    )

    # Verify the actual forward works with non-causal mask
    q = torch.randn(4, 1, 64, dtype=torch.float16, device="cuda")
    kv_cache = torch.randn(2, 2, 1, 4, 64, dtype=torch.float16, device="cuda")
    out = torch.empty(4, 1, 64, dtype=torch.float16, device="cuda")

    wrapper.run(q, kv_cache, out=out)
    assert out.shape == (4, 1, 64)
    assert not torch.isnan(out).any(), "Output contains NaN"


def test_wrapper_plan_respects_causal_true():
    """Verify wrapper also works correctly with causal=True."""
    if not torch.cuda.is_available():
        return

    workspace = torch.zeros(2048 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "HND")

    qo_indptr = torch.tensor([0, 4], dtype=torch.int32, device="cpu")
    paged_kv_indptr = torch.tensor([0, 2], dtype=torch.int32, device="cpu")
    paged_kv_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor([4], dtype=torch.int32, device="cpu")

    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=64,
        page_size=4,
        causal=True,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        o_data_type=torch.float16,
    )

    assert wrapper._causal is True, (
        f"Wrapper should have _causal=True after plan(causal=True), "
        f"got {wrapper._causal}"
    )


def test_non_causal_vs_causal_produces_different_output():
    """Non-causal and causal attention should produce different results."""
    if not torch.cuda.is_available():
        return

    workspace = torch.zeros(2048 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    qo_indptr = torch.tensor([0, 4], dtype=torch.int32, device="cpu")
    paged_kv_indptr = torch.tensor([0, 2], dtype=torch.int32, device="cpu")
    paged_kv_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor([4], dtype=torch.int32, device="cpu")

    # Run with causal=True
    wrapper_causal = BatchPrefillWithPagedKVCacheWrapper(workspace, "HND")
    wrapper_causal.plan(
        qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=1, num_kv_heads=1, head_dim_qk=64, page_size=4,
        causal=True, q_data_type=torch.float16,
        kv_data_type=torch.float16, o_data_type=torch.float16,
    )

    # Run with causal=False
    wrapper_non_causal = BatchPrefillWithPagedKVCacheWrapper(workspace, "HND")
    wrapper_non_causal.plan(
        qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=1, num_kv_heads=1, head_dim_qk=64, page_size=4,
        causal=False, q_data_type=torch.float16,
        kv_data_type=torch.float16, o_data_type=torch.float16,
    )

    # Same inputs
    q = torch.randn(4, 1, 64, dtype=torch.float16, device="cuda")
    kv_cache = torch.randn(2, 2, 1, 4, 64, dtype=torch.float16, device="cuda")

    out_causal = torch.empty(4, 1, 64, dtype=torch.float16, device="cuda")
    out_non_causal = torch.empty(4, 1, 64, dtype=torch.float16, device="cuda")

    wrapper_causal.run(q.clone(), kv_cache, out=out_causal)
    wrapper_non_causal.run(q.clone(), kv_cache, out=out_non_causal)

    # They should differ (non-causal sees all tokens, causal masks future)
    diff = (out_causal - out_non_causal).abs().max().item()
    assert diff > 1e-4, (
        f"Causal and non-causal outputs should differ significantly, "
        f"but max diff is only {diff:.6f}. "
        "This suggests causal masking is not being applied correctly."
    )
    print(f"Max difference between causal and non-causal: {diff:.6f}")
