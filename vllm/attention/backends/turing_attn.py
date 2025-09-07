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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import torch
import triton
import triton.language as tl

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
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
    seq_lens: Optional[List[int]] = None
    seq_lens_tensor: Optional[torch.Tensor] = None
    max_prefill_seq_len: int = 0
    max_decode_seq_len: int = 0
    use_cuda_graph: bool = False
    seq_start_loc: Optional[torch.Tensor] = None
    context_lens_tensor: Optional[torch.Tensor] = None
    query_start_loc: Optional[torch.Tensor] = None

    _cached_prefill_metadata: Optional["TuringAttentionMetadata"] = None
    _cached_decode_metadata: Optional["TuringAttentionMetadata"] = None

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
            seq_lens=self.seq_lens[:self.num_prefills] if self.seq_lens else None,
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills] if self.seq_lens_tensor is not None else None,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            block_tables=self.block_tables[:self.num_prefills] if self.block_tables is not None else None,
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills] if self.context_lens_tensor is not None else None,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1] if self.query_start_loc is not None else None,
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
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
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:] if self.seq_lens_tensor is not None else None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=self.block_tables[self.num_prefills:] if self.block_tables is not None else None,
            context_lens_tensor=None,
            query_start_loc=None,
            use_cuda_graph=self.use_cuda_graph,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
        )
        return self._cached_decode_metadata

class TuringAttentionMetadataBuilder(CommonMetadataBuilder[TuringAttentionMetadata]):
    _metadata_cls = TuringAttentionMetadata


class TuringAttentionImpl(AttentionImpl[TuringAttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes are not supported by the Turing attention backend.")
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

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
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        output = torch.empty_like(query)

        # Prefill phase
        if prefill_meta := attn_metadata.prefill_metadata:
            # We only handle the case where there is no prefix caching.
            # If there is prefix caching, PagedAttention.forward_prefix should be used,
            # but that is a more complex kernel. This implementation focuses on the
            # common case of a new prompt.
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                out = _run_turing_flash_attention_forward(
                    query[:num_prefill_tokens],
                    key[:num_prefill_tokens],
                    value[:num_prefill_tokens],
                    prefill_meta,
                    self.num_heads,
                    self.scale
                )
                output[:num_prefill_tokens] = out
            else:
                # Fallback for prefix caching, which this simple backend doesn't support.
                # A more robust implementation would use PagedAttention.forward_prefix here.
                logger.warning_once("TuringAttentionBackend does not support prefix caching, falling back to PagedAttention for prefill.")
                key_cache, value_cache = PagedAttention.split_kv_cache(
                    kv_cache, self.num_kv_heads, self.head_size)
                out = PagedAttention.forward_prefix(
                    query[:num_prefill_tokens],
                    key[:num_prefill_tokens],
                    value[:num_prefill_tokens],
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.max_prefill_seq_len,
                    alibi_slopes=None,
                    sliding_window=self.sliding_window,
                )
                output[:num_prefill_tokens] = out

        # Decode phase
        if decode_meta := attn_metadata.decode_metadata:
            decode_query = query[num_prefill_tokens:]
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            output[num_prefill_tokens:] = PagedAttention.forward_decode(
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
            )

        return output


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
    """
    # Reshape tensors to be per-head.
    Z = len(attn_metadata.seq_lens)
    H = num_heads
    N_CTX = query.shape[0] // Z
    D_HEAD = query.shape[-1] // H

    q = query.view(Z, N_CTX, H, D_HEAD).transpose(1, 2)
    k = key.view(Z, N_CTX, H, D_HEAD).transpose(1, 2)
    v = value.view(Z, N_CTX, H, D_HEAD).transpose(1, 2)

    # The Triton kernel is causal by default in this simple integration.
    # A more complete version would get this from a config.
    is_causal = True
    output = _turing_attention_kernel(q, k, v, is_causal, scale)

    return output.transpose(1, 2).reshape(-1, H * D_HEAD)


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

    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        s_ij = tl.dot(q, k, out_dtype=tl.float32)

        if IS_CAUSAL:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            causal_mask = offs_m[:, None] < offs_n[None, :]
            s_ij = tl.where(causal_mask, float('-inf'), s_ij)

        m_ij = tl.max(s_ij, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        exp_diff = tl.exp(m_i - m_i_new)
        acc = acc * exp_diff
        l_i = l_i * exp_diff
        p_ij = tl.exp(s_ij - m_i_new[:, None])
        l_i += tl.sum(p_ij, 1)
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

    grid = (triton.cdiv(N_CTX, 128), Z, H)

    BLOCK_M = 128
    BLOCK_N = 64

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
