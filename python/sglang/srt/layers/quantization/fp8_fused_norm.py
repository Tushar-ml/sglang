"""Fused RMSNorm + per-block FP8 quant for DeepGEMM (vLLM rms_norm_per_block_quant analog)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.utils import get_bool_env_var, is_cuda

_GROUP_SIZE = 128


def _alloc_outputs(
    input: torch.Tensor, group_size: int, scale_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    output_q = torch.empty(input.shape, device=input.device, dtype=fp8_dtype)
    output_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=output_q.shape,
        device=input.device,
        group_size=group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    return output_q, output_s


def _try_cuda_rmsnorm_per_block_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor],
    group_size: int,
    scale_ue8m0: bool,
    gemma_style: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        if residual is None:
            op_name = (
                "gemma_rmsnorm_per_block_fp8_quant"
                if gemma_style
                else "rmsnorm_per_block_fp8_quant"
            )
        else:
            op_name = (
                "gemma_fused_add_rmsnorm_per_block_fp8_quant"
                if gemma_style
                else "fused_add_rmsnorm_per_block_fp8_quant"
            )
        if not hasattr(torch.ops.sgl_kernel, op_name):
            return None

        output_q, output_s = _alloc_outputs(input, group_size, scale_ue8m0)
        op = getattr(torch.ops.sgl_kernel, op_name)
        if residual is None:
            op(output_q, output_s, input, weight, eps, group_size, scale_ue8m0)
        else:
            op(
                output_q,
                output_s,
                input,
                residual,
                weight,
                eps,
                group_size,
                scale_ue8m0,
            )
        return output_q, output_s
    except Exception:
        return None


def _fallback_rmsnorm_per_block_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor],
    group_size: int,
    scale_ue8m0: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import fused_add_rmsnorm, rmsnorm

    normed = torch.empty_like(input)
    if residual is None:
        rmsnorm(input, weight, eps, normed)
    else:
        normed = input.clone()
        res = residual.clone()
        fused_add_rmsnorm(normed, res, weight, eps)

    return sglang_per_token_group_quant_fp8(
        normed,
        group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )


def rmsnorm_per_block_fp8_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int = _GROUP_SIZE,
    scale_ue8m0: bool = True,
    gemma_style: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm fused with per-block FP8 quant for DeepGEMM (vLLM analog)."""
    cuda_out = _try_cuda_rmsnorm_per_block_fp8(
        input, weight, eps, None, group_size, scale_ue8m0, gemma_style
    )
    if cuda_out is not None:
        return cuda_out
    return _fallback_rmsnorm_per_block_fp8(
        input, weight, eps, None, group_size, scale_ue8m0
    )


def fused_add_rmsnorm_per_block_fp8_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int = _GROUP_SIZE,
    scale_ue8m0: bool = True,
    gemma_style: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused (input+residual) + RMSNorm + per-block FP8 quant."""
    cuda_out = _try_cuda_rmsnorm_per_block_fp8(
        input, weight, eps, residual, group_size, scale_ue8m0, gemma_style
    )
    if cuda_out is not None:
        return cuda_out
    return _fallback_rmsnorm_per_block_fp8(
        input, weight, eps, residual, group_size, scale_ue8m0
    )


def gemma_rmsnorm_per_block_fp8_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int = _GROUP_SIZE,
    scale_ue8m0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return rmsnorm_per_block_fp8_quant(
        input, weight, eps, group_size, scale_ue8m0, gemma_style=True
    )


def gemma_fused_add_rmsnorm_per_block_fp8_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int = _GROUP_SIZE,
    scale_ue8m0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fused_add_rmsnorm_per_block_fp8_quant(
        input, residual, weight, eps, group_size, scale_ue8m0, gemma_style=True
    )


def use_gemma4_fused_norm_fp8_quant(quant_config) -> bool:
    if not is_cuda() or quant_config is None:
        return False
    if not get_bool_env_var("SGLANG_GEMMA4_FUSED_NORM_FP8", "1"):
        return False
    block_size = getattr(quant_config, "weight_block_size", None)
    if block_size is None:
        return False
    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return False
    return block_size[1] == _GROUP_SIZE
