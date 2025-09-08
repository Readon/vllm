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
import itertools
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
                # Use the new prefix-aware Triton kernel for chunked prefill
                prefill_query = query[:num_prefill_tokens]
                prefill_key = key[:num_prefill_tokens] if key is not None else None
                prefill_value = value[:num_prefill_tokens] if value is not None else None

                out = _prefix_attention(
                    prefill_query,
                    prefill_key,
                    prefill_value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.max_query_len,
                    self.scale,
                    self.num_kv_heads,
                    self.kv_cache_dtype,
                    k_scale,
                    v_scale,
                )
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
    This version avoids manual padding and uses cumulative sequence lengths.
    """
    if attn_metadata.seq_lens is None or not attn_metadata.seq_lens:
        raise ValueError("seq_lens is required for Turing attention")

    # Input tensors are already in [num_tokens, num_heads, head_size] format
    num_tokens, _, head_size = query.shape
    _, num_kv_heads, _ = key.shape

    # Handle GQA/MQA by repeating KV heads if needed
    if num_kv_heads != num_heads:
        num_queries_per_kv = num_heads // num_kv_heads
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    if head_size not in [16, 32, 64, 128]:
        raise ValueError(f"Head size {head_size} not supported by Turing backend. Supported sizes: [16, 32, 64, 128]")

    # Create cumulative sequence lengths tensor
    cu_seqlens = torch.tensor(
        list(itertools.accumulate([0] + attn_metadata.seq_lens)),
        device=query.device,
        dtype=torch.int32
    )

    is_causal = True
    output = _turing_attention_kernel(
        query,
        key,
        value,
        cu_seqlens,
        attn_metadata.max_prefill_seq_len,
        is_causal,
        scale
    )

    return output.view(num_tokens, num_heads * head_size)


@triton.jit
def _prefix_fwd_kernel(
    Q, K, V, K_cache, V_cache, block_tables,
    k_scale, v_scale, sm_scale, B_Start_Loc, B_Seqlen, Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_block_tables_b, stride_block_tables_s,
    stride_k_cache_b, stride_k_cache_h, stride_k_cache_d, stride_k_cache_bl, stride_k_cache_x,
    stride_v_cache_b, stride_v_cache_h, stride_v_cache_d, stride_v_cache_bl,
    num_queries_per_kv: tl.constexpr, x: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    DEQUANTIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_query_len, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Attention with KV Cache
    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    for start_n in range(0, cur_batch_ctx_len, BLOCK_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE)
        bn = tl.load(block_tables + cur_batch * stride_block_tables_b + (start_n // BLOCK_SIZE) * stride_block_tables_s)
        off_k = (bn[None, :] * stride_k_cache_b + cur_kv_head * stride_k_cache_h +
                 (offs_d[:, None] // x) * stride_k_cache_d +
                 ((start_n + offs_bs_n[None, :]) % BLOCK_SIZE) * stride_k_cache_bl +
                 (offs_d[:, None] % x) * stride_k_cache_x)
        off_v = (bn[:, None] * stride_v_cache_b + cur_kv_head * stride_v_cache_h +
                 offs_d[None, :] * stride_v_cache_d +
                 offs_bs_n[:, None] * stride_v_cache_bl)

        k_loaded = tl.load(K_cache + off_k, mask=(start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, other=0.0)
        if DEQUANTIZE:
            k = (k_loaded.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_loaded

        qk = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        qk *= sm_scale
        qk = tl.where((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        v_loaded = tl.load(V_cache + off_v, mask=(start_n + offs_bs_n[:, None]) < cur_batch_ctx_len, other=0.0)
        if DEQUANTIZE:
            v = (v_loaded.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_loaded
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Self-attention for the new tokens
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        off_k = (cur_batch_in_all_start_index + start_n + offs_n[None, :]) * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        off_v = (cur_batch_in_all_start_index + start_n + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < cur_batch_query_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < cur_batch_query_len, other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_query_len)


def _prefix_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
    num_kv_heads: int,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(query)
    batch_size = block_tables.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (batch_size, num_heads, triton.cdiv(max_query_len, BLOCK_M))

    x = 16 // key_cache.element_size()

    dequantize = "int8" in kv_cache_dtype or "fp8" in kv_cache_dtype

    _prefix_fwd_kernel[grid](
        query, key, value, key_cache, value_cache, block_tables,
        k_scale, v_scale, sm_scale, query_start_loc, seq_lens_tensor, output,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3), key_cache.stride(4),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        num_queries_per_kv=num_heads // num_kv_heads, x=x,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=head_size, BLOCK_N=BLOCK_N,
        BLOCK_SIZE=value_cache.shape[3],
        DEQUANTIZE=dequantize,
    )
    return output


turing_autotune_configs = [
    # Start with smaller block sizes and vary warps/stages
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
    # Include some larger block sizes as well
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
]


@triton.autotune(
    configs=turing_autotune_configs,
    key=['D_HEAD'],
)
@triton.jit
def _turing_attention_kernel_forward(
    Q, K, V, Out,
    cu_seqlens_q,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
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

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
    seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start

    if start_m * BLOCK_M >= seqlen_q:
        return

    q_offset = off_h * stride_qh + cu_seqlens_q_start * stride_qm
    k_offset = off_h * stride_kh + cu_seqlens_q_start * stride_kn
    v_offset = off_h * stride_vh + cu_seqlens_q_start * stride_vn
    out_offset = off_h * stride_oh + cu_seqlens_q_start * stride_om

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset, shape=(seqlen_q, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset, shape=(D_HEAD, seqlen_q),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(D_HEAD, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset, shape=(seqlen_q, D_HEAD),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, D_HEAD),
        order=(1, 0)
    )
    Out_block_ptr = tl.make_block_ptr(
        base=Out + out_offset, shape=(seqlen_q, D_HEAD),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )

    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    m_i = tl.full([BLOCK_M, 1], value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    q = (q * SCALE).to(Q.dtype.element_ty)

    loop_end = seqlen_q if not IS_CAUSAL else (start_m + 1) * BLOCK_M
    for start_n in range(0, loop_end, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        s_ij = tl.dot(q, k, out_dtype=tl.float32)

        if IS_CAUSAL:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
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


def _turing_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    total_tokens, num_heads, head_size = q.shape
    batch_size = len(cu_seqlens) - 1
    o = torch.empty_like(q)

    grid = lambda META: (triton.cdiv(max_seq_len, META['BLOCK_M']), batch_size, num_heads)

    # Strides for 3D tensors, interpreted as 4D by the kernel
    stride_qz, stride_qh, stride_qm, stride_qk = 0, q.stride(1), q.stride(0), q.stride(2)
    stride_kz, stride_kh, stride_kn, stride_kk = 0, k.stride(1), k.stride(0), k.stride(2)
    stride_vz, stride_vh, stride_vn, stride_vk = 0, v.stride(1), v.stride(0), v.stride(2)
    stride_oz, stride_oh, stride_om, stride_ok = 0, o.stride(1), o.stride(0), o.stride(2)


    _turing_attention_kernel_forward[grid](
        q, k, v, o,
        cu_seqlens,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        batch_size, num_heads, max_seq_len,
        SCALE=scale,
        D_HEAD=head_size,
        IS_CAUSAL=is_causal,
    )
    return o
