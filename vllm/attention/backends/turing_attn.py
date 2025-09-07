# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2024 Jules, an AI-powered software engineer.
"""
Attention layer with a custom Triton kernel for Turing GPUs (Compute Capability 7.5).

This backend is designed to provide a memory-efficient attention mechanism for
NVIDIA's Turing architecture, which does not support modern FlashAttention-2.

NOTE: This implementation accelerates the **prefill phase** of attention by using
a custom Triton kernel. The **decode phase** reuses the existing, highly-optimized
PagedAttention CUDA kernel from vLLM, as this is typically faster for single-
token decoding than a generic Triton implementation.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

import torch
import triton
import triton.language as tl

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionMetadata,
                                              AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger

# We need to set this environment variable to allow Triton to use the legacy PTX assembler,
# which is necessary for older compute capabilities like 7.5 (Turing).
os.environ['TRITON_USE_LEGACY_PTX_ASSEMBLER'] = '1'

logger = init_logger(__name__)


class TuringAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "Turing"

    @staticmethod
    def get_impl_cls() -> Type["TuringAttentionImpl"]:
        return TuringAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return TuringAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["TuringAttentionMetadataBuilder"]:
        return TuringAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [16, 32, 64, 128]

    @staticmethod
    def get_supported_dtypes() -> List[torch.dtype]:
        return [torch.float16]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> Tuple[int, ...]:
        # Use the same stride order as PagedAttention for compatibility
        return (0, 1, 2)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class TuringAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """
    Metadata for the TuringAttentionBackend.
    This is largely based on XFormersMetadata, as the metadata needed is
    determined by the PagedAttention mechanism rather than the core kernel.
    """
    # Additional fields specific to Turing backend
    seq_lens: Optional[List[int]]
    max_query_len: int
    max_prefill_seq_len: int
    use_cuda_graph: bool
    seq_start_loc: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    query_start_loc: Optional[torch.Tensor]
    max_decode_query_len: Optional[int] = None

    _cached_prefill_metadata: Optional["TuringAttentionMetadata"] = field(default=None)
    _cached_decode_metadata: Optional["TuringAttentionMetadata"] = field(default=None)

    @property
    def prefill_metadata(self) -> Optional["TuringAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        self._cached_prefill_metadata = TuringAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills] if self.seq_lens_tensor is not None else None,
            max_decode_seq_len=0,
            block_tables=self.block_tables[:self.num_prefills] if self.block_tables is not None else None,
            seq_lens=self.seq_lens[:self.num_prefills] if self.seq_lens else None,
            max_query_len=self.max_query_len,
            max_decode_query_len=getattr(self, 'max_decode_query_len', None),
            max_prefill_seq_len=self.max_prefill_seq_len,
            use_cuda_graph=False,
            seq_start_loc=None,
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills] if self.context_lens_tensor is not None else None,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1] if self.query_start_loc is not None else None,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["TuringAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata

        self._cached_decode_metadata = TuringAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:] if self.seq_lens_tensor is not None else None,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=self.block_tables[self.num_prefills:] if self.block_tables is not None else None,
            seq_lens=None,
            max_query_len=1,  # For decode, query length is always 1
            max_decode_query_len=getattr(self, 'max_decode_query_len', 1),
            max_prefill_seq_len=0,
            use_cuda_graph=self.use_cuda_graph,
            seq_start_loc=None,
            context_lens_tensor=None,
            query_start_loc=None,
        )
        return self._cached_decode_metadata

class TuringAttentionMetadataBuilder(CommonMetadataBuilder[TuringAttentionMetadata]):
    _metadata_cls = TuringAttentionMetadata

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int) -> TuringAttentionMetadata:
        """Build attention metadata with on-device tensors."""
        # Call parent build method to get base metadata
        base_metadata = super().build(seq_lens, query_lens, cuda_graph_pad_size, batch_size)

        # Create TuringAttentionMetadata with additional fields
        return TuringAttentionMetadata(
            num_prefills=base_metadata.num_prefills,
            num_prefill_tokens=base_metadata.num_prefill_tokens,
            num_decode_tokens=base_metadata.num_decode_tokens,
            slot_mapping=base_metadata.slot_mapping,
            multi_modal_placeholder_index_maps=base_metadata.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=base_metadata.enable_kv_scales_calculation,
            seq_lens_tensor=base_metadata.seq_lens_tensor,
            max_decode_seq_len=base_metadata.max_decode_seq_len,
            block_tables=getattr(base_metadata, 'block_tables', None),
            seq_lens=getattr(base_metadata, 'seq_lens', None),
            max_query_len=getattr(base_metadata, 'max_query_len', 0),
            max_prefill_seq_len=getattr(base_metadata, 'max_prefill_seq_len', 0),
            use_cuda_graph=getattr(base_metadata, 'use_cuda_graph', False),
            seq_start_loc=getattr(base_metadata, 'seq_start_loc', None),
            context_lens_tensor=getattr(base_metadata, 'context_lens_tensor', None),
            query_start_loc=getattr(base_metadata, 'query_start_loc', None),
        )


class TuringAttentionImpl(AttentionImpl[TuringAttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes are not supported by the Turing attention backend.")
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TuringAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Turing-optimized attention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # Get k_scale and v_scale from layer if available, otherwise use default
        # Use existing tensors to avoid creating new ones during CUDA graph capture
        if hasattr(layer, '_k_scale'):
            k_scale = layer._k_scale
        else:
            k_scale = torch.ones(1, dtype=query.dtype, device=query.device)

        if hasattr(layer, '_v_scale'):
            v_scale = layer._v_scale
        else:
            v_scale = torch.ones(1, dtype=query.dtype, device=query.device)

        # Reshape input tensors to proper format for attention computation
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        # Only update KV cache for decode operations when kv_cache is available
        if kv_cache.numel() > 0 and key is not None and value is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Write keys and values to cache
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        # Prefill phase
        if prefill_meta := attn_metadata.prefill_metadata:
            # We only handle the case where there is no prefix caching.
            # If there is prefix caching, PagedAttention.forward_prefix should be used,
            # but that is a more complex kernel. This implementation focuses on the
            # common case of a new prompt.
            if (kv_cache.numel() == 0 or
                prefill_meta.block_tables is None or
                prefill_meta.block_tables.numel() == 0):
                # Use Turing-optimized kernel for prefill
                prefill_query = query[:num_prefill_tokens]
                prefill_key = key[:num_prefill_tokens] if key is not None else None
                prefill_value = value[:num_prefill_tokens] if value is not None else None

                out = _run_turing_flash_attention_forward(
                    prefill_query,
                    prefill_key,
                    prefill_value,
                    prefill_meta,
                    self.num_heads,
                    self.scale
                )
                # The Turing kernel returns [num_tokens, num_heads * head_size], reshape to [num_tokens, num_heads, head_size]
                output[:num_prefill_tokens] = out.view(num_prefill_tokens, self.num_heads, self.head_size)
            else:
                # Fallback for prefix caching
                logger.warning_once("TuringAttentionBackend using PagedAttention for prefix caching.")

                prefill_query = query[:num_prefill_tokens]
                prefill_key = key[:num_prefill_tokens] if key is not None else None
                prefill_value = value[:num_prefill_tokens] if value is not None else None

                out = PagedAttention.forward_prefix(
                    prefill_query,
                    prefill_key,
                    prefill_value,
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.max_prefill_seq_len,
                    None,  # alibi_slopes
                    self.sliding_window,
                    k_scale,
                    v_scale,
                )
                # PagedAttention returns [num_tokens, num_heads, head_size], which is what output buffer expects
                output[:num_prefill_tokens] = out

        # Decode phase - always use PagedAttention for decode as it's optimized for single-token generation
        if decode_meta := attn_metadata.decode_metadata:
            decode_query = query[num_prefill_tokens:]

            decode_output = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.seq_lens_tensor,
                decode_meta.max_decode_seq_len,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                alibi_slopes=None,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            # PagedAttention.forward_decode returns [num_tokens, num_heads, head_size], which is what output buffer expects
            output[num_prefill_tokens:] = decode_output

        # Reshape the output tensor back to the expected format
        return output.view(-1, self.num_heads * self.head_size)


def _run_turing_flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: TuringAttentionMetadata,
    num_heads: int,
    scale: float,
) -> torch.Tensor:
    """
    Wrapper to call the Triton kernel for the prefill phase.

    Args:
        query: [num_tokens, num_heads, head_size]
        key: [num_tokens, num_kv_heads, head_size]
        value: [num_tokens, num_kv_heads, head_size]
    """
    if attn_metadata.seq_lens is None or len(attn_metadata.seq_lens) == 0:
        raise ValueError("seq_lens is required for Turing attention")

    batch_size = len(attn_metadata.seq_lens)
    seq_lens = attn_metadata.seq_lens
    max_seq_len = max(seq_lens)

    # Input tensors are already in [num_tokens, num_heads, head_size] format
    _, _, head_size = query.shape
    _, num_kv_heads, _ = key.shape

    # Handle GQA/MQA by repeating KV heads if needed
    if num_kv_heads != num_heads:
        num_queries_per_kv = num_heads // num_kv_heads
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    if head_size not in [16, 32, 64, 128]:
        raise ValueError(f"Head size {head_size} not supported by Turing backend. Supported sizes: [16, 32, 64, 128]")

    # Create padded tensors for batch processing
    q_padded = torch.zeros(batch_size, max_seq_len, num_heads, head_size,
                          dtype=query.dtype, device=query.device)
    k_padded = torch.zeros(batch_size, max_seq_len, num_heads, head_size,
                          dtype=key.dtype, device=key.device)
    v_padded = torch.zeros(batch_size, max_seq_len, num_heads, head_size,
                          dtype=value.dtype, device=value.device)

    # Fill padded tensors
    start_idx = 0
    for i, seq_len in enumerate(seq_lens):
        end_idx = start_idx + seq_len
        # Tensors are already in the correct shape [seq_len, num_heads, head_size]
        q_padded[i, :seq_len] = query[start_idx:end_idx]
        k_padded[i, :seq_len] = key[start_idx:end_idx]
        v_padded[i, :seq_len] = value[start_idx:end_idx]
        start_idx = end_idx

    # Transpose to [batch_size, num_heads, seq_len, head_size]
    q_padded = q_padded.transpose(1, 2)
    k_padded = k_padded.transpose(1, 2)
    v_padded = v_padded.transpose(1, 2)

    # Run attention kernel
    is_causal = True
    output_padded = _turing_attention_kernel(q_padded, k_padded, v_padded, is_causal, scale)

    # Unpad and reshape output back to [num_tokens, num_heads * head_size]
    output_padded = output_padded.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_size]
    output_list = []
    for i, seq_len in enumerate(seq_lens):
        out_seq = output_padded[i, :seq_len]  # [seq_len, num_heads, head_size]
        # Reshape to [seq_len, num_heads * head_size] to match expected output format
        out_seq = out_seq.reshape(seq_len, num_heads * head_size)
        output_list.append(out_seq)

    return torch.cat(output_list, dim=0)  # [num_tokens, num_heads * head_size]


@triton.jit
def _turing_attention_kernel_forward(
    Q, K, V, Out,
    q_stride_z, q_stride_h, q_stride_m, q_stride_k,
    k_stride_z, k_stride_h, k_stride_n, k_stride_k,
    v_stride_z, v_stride_h, v_stride_n, v_stride_k,
    out_stride_z, out_stride_h, out_stride_m, out_stride_k,
    Z, H, N_CTX,
    SCALE: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)

    # Bounds checking for batch and head dimensions
    if off_z >= Z or off_h >= H:
        return

    # Check if we're within sequence bounds
    if start_m * BLOCK_M >= N_CTX:
        return

    q_offset = off_z * q_stride_z + off_h * q_stride_h
    k_offset = off_z * k_stride_z + off_h * k_stride_h
    v_offset = off_z * v_stride_z + off_h * v_stride_h
    out_offset = off_z * out_stride_z + off_h * out_stride_h

    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX, D_HEAD), strides=(q_stride_m, q_stride_k), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, D_HEAD), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(D_HEAD, N_CTX), strides=(k_stride_k, k_stride_n), offsets=(0, 0), block_shape=(D_HEAD, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(N_CTX, D_HEAD), strides=(v_stride_n, v_stride_k), offsets=(0, 0), block_shape=(BLOCK_N, D_HEAD), order=(1, 0))
    Out_block_ptr = tl.make_block_ptr(base=Out + out_offset, shape=(N_CTX, D_HEAD), strides=(out_stride_m, out_stride_k), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, D_HEAD), order=(1, 0))

    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    m_i = tl.full([BLOCK_M, 1], value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    q = (q * SCALE).to(Q.dtype.element_ty)

    # For causal attention, only process up to the current position
    loop_end = N_CTX if not IS_CAUSAL else min(N_CTX, (start_m + 1) * BLOCK_M)
    for start_n in range(0, loop_end, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        s_ij = tl.dot(q, k, out_dtype=tl.float32)

        if IS_CAUSAL:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # Ensure we don't go out of bounds
            offs_m = tl.where(offs_m < N_CTX, offs_m, N_CTX - 1)
            offs_n = tl.where(offs_n < N_CTX, offs_n, N_CTX - 1)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(causal_mask, s_ij, float('-inf'))

        m_ij = tl.max(s_ij, 1)[:, None]
        m_i_new = tl.maximum(m_i, m_ij)
        exp_diff = tl.exp(m_i - m_i_new)
        acc = acc * exp_diff
        l_i = l_i * exp_diff
        p_ij = tl.exp(s_ij - m_i_new)
        l_i += tl.sum(p_ij, 1)[:, None]
        p_ij = p_ij.to(V.dtype.element_ty)
        acc += tl.dot(p_ij, v)
        m_i = m_i_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    out = acc / l_i_safe
    tl.store(Out_block_ptr, out.to(Out.dtype.element_ty), boundary_check=(0, 1))


def _turing_attention_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, scale: float) -> torch.Tensor:
    shape = q.shape
    Z, H, N_CTX, D_HEAD = shape

    o = torch.empty_like(q)

    # Use smaller block sizes for Turing architecture to fit in shared memory
    BLOCK_M = 64
    BLOCK_N = 32

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z, H)

    _turing_attention_kernel_forward[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        SCALE=scale,
        D_HEAD=D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
    )
    return o
