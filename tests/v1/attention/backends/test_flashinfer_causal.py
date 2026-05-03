# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test causal attribute support in FlashInferMetadata."""
import torch

from vllm.v1.attention.backends.flashinfer import FlashInferMetadata


def test_flashinfer_metadata_has_causal_attribute():
    """Verify FlashInferMetadata has causal attribute with correct default."""
    # Create minimal FlashInferMetadata instance
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

    # Check causal attribute exists and defaults to True
    assert hasattr(metadata, 'causal'), \
        "FlashInferMetadata should have 'causal' attribute"
    assert metadata.causal is True, \
        f"FlashInferMetadata.causal should default to True, got {metadata.causal}"


def test_flashinfer_metadata_causal_can_be_false():
    """Verify FlashInferMetadata.causal can be explicitly set to False for DFlash."""
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
        causal=False,  # Explicitly set for non-causal attention (DFlash use case)
    )

    assert metadata.causal is False, \
        f"FlashInferMetadata.causal should be False when explicitly set, got {metadata.causal}"


def test_flashinfer_metadata_getattr_causal_pattern():
    """Verify getattr pattern used in dflash.py assertion works correctly.

    This test simulates the check in dflash.py:267:
        assert getattr(attn_metadata, "causal", None) is False
    """
    # Test with causal=True (default)
    metadata_causal = FlashInferMetadata(
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
        causal=True,
    )
    result = getattr(metadata_causal, "causal", None)
    assert result is True, \
        f"getattr with causal=True should return True, got {result}"
    assert (result is not False), \
        "DFlash assertion would fail for causal=True (expected behavior)"

    # Test with causal=False (DFlash use case)
    metadata_non_causal = FlashInferMetadata(
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
        causal=False,
    )
    result = getattr(metadata_non_causal, "causal", None)
    assert result is False, \
        f"getattr with causal=False should return False, got {result}"
    # This is the key: DFlash assertion should PASS here
    assert (result is False), \
        "DFlash assertion 'getattr(attn_metadata, \"causal\", None) is False' should pass"
