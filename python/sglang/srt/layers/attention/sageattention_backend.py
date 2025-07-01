from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_ops.sparse_int8_attn import (
    forward as sparse_sageattn_fwd,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class SageAttentionMetadata:
    tensor_layout: str = "HND"
    mask_id: Optional[torch.Tensor] = None
    cache_seqlens_int32: torch.Tensor = None
    page_table: torch.Tensor = None
    max_seq_len_q: int = 0
    max_seq_len_k: int = 0
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    window_size: Tuple[int, int] = (-1, -1)


class SageAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata: SageAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = SageAttentionMetadata()

        extend_seq_lens = forward_batch.extend_seq_lens
        seqlens_in_batch = forward_batch.seq_lens

        metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)

        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )

        metadata.max_seq_len_k = seqlens_in_batch.max().item()
        # Precompute page table
        metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, : metadata.max_seq_len_k
        ]

        if forward_batch.forward_mode == ForwardMode.DECODE:
            # Precompute cumulative sequence lengths
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
        else:
            extend_no_prefix = not any(forward_batch.extend_prefix_lens)
            # Precompute cumulative sequence lengths
            if not extend_no_prefix:
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.cu_seqlens_q = metadata.cu_seqlens_k
            metadata.max_seq_len_q = seqlens_in_batch.max().item()
        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        metadata = self.forward_metadata
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache, value_cache = kv_cache[0], kv_cache[1]

        o = sparse_sageattn(q, key_cache, value_cache, k, v, metadata.tensor_layout)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


def sparse_sageattn(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    mask_id=None,
    is_causal=False,
    tensor_layout="HND",
):
    if mask_id is None:
        mask_id = torch.ones(
            (
                q.shape[0],
                q.shape[1],
                (q.shape[2] + 128 - 1) // 128,
                (q.shape[3] + 64 - 1) // 64,
            ),
            dtype=torch.int8,
            device=q.device,
        )  # TODO

    output_dtype = q.dtype
    if output_dtype == torch.bfloat16 or output_dtype == torch.float32:
        v = v.to(torch.float16)

    seq_dim = 1 if tensor_layout == "NHD" else 2
    km = k.mean(dim=seq_dim, keepdim=True)
    # km = torch.zeros((k.size(0), k.size(1), 1, k.size(3)), dtype=torch.float16, device=k.device)  # Placeholder for mean, not used in quantization

    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, km=km, tensor_layout=tensor_layout
    )

    o = sparse_sageattn_fwd(
        q_int8,
        k_int8,
        mask_id,
        v,
        q_scale,
        k_scale,
        is_causal=is_causal,
        tensor_layout=tensor_layout,
        output_dtype=output_dtype,
    )
    return o


@triton.jit
def quant_per_block_int8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    sm_scale,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = (
        Input
        + off_b * stride_iz
        + off_h * stride_ih
        + offs_n[:, None] * stride_in
        + offs_k[None, :]
    )
    output_ptrs = (
        Output
        + off_b * stride_oz
        + off_h * stride_oh
        + offs_n[:, None] * stride_on
        + offs_k[None, :]
    )
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def per_block_int8(
    q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = (
            q_int8.stride(0),
            q_int8.stride(1),
            q_int8.stride(2),
        )
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = (
            k_int8.stride(0),
            k_int8.stride(1),
            k_int8.stride(2),
        )
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = (
            q_int8.stride(0),
            q_int8.stride(2),
            q_int8.stride(1),
        )
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = (
            k_int8.stride(0),
            k_int8.stride(2),
            k_int8.stride(1),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32
    )
    k_scale = torch.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32
    )

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_qo,
        stride_h_qo,
        stride_seq_qo,
        q_scale.stride(0),
        q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim,
        BLK=BLKQ,
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k,
        k_int8,
        k_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        k_scale.stride(0),
        k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim,
        BLK=BLKK,
    )

    return q_int8, q_scale, k_int8, k_scale
