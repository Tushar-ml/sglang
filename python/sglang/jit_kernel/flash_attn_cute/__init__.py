# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang,
#   Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Adapted from vllm-project/vllm (vllm/vllm_flash_attn/cute/) with import paths
# rewritten from vllm.vllm_flash_attn.cute → sglang.jit_kernel.flash_attn_cute.
# Requires: pip install quack-kernels nvidia-cutlass-dsl
"""Flash Attention CUTE (CuTe DSL) kernels — FA4 forward pass for SM90/SM100/SM120."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
