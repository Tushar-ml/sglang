"""Benchmark Triton kernels for Gemma4 fused ops.

Run:
    python sgl-kernel/benchmark/bench_gemma4_fused_ops.py
"""

import itertools
import os
import sys

import torch
import triton
import triton.testing

IS_CI = os.environ.get("SGLANG_IS_CI", "0") == "1"

sys.path.insert(0, "python")
from sglang.srt.layers.gemma4_fused_ops import (  # noqa: E402
    _triton_fused_routing,
    _triton_qkv_rmsnorm,
    _triton_rmsnorm_residual_scalar,
    _triton_routing_post_topk,
)

# ---------------------------------------------------------------------------
# Benchmark 1: gemma4_rmsnorm_residual_scalar
# ---------------------------------------------------------------------------
RMSNORM_CONFIGS = (
    [(1, 2560), (4, 2560), (32, 2560), (128, 2560)]
    if IS_CI
    else list(itertools.product([1, 4, 16, 64, 256, 512, 1024], [2560, 4096]))
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=RMSNORM_CONFIGS,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton (autotuned)"],
        styles=[("blue", "-")],
        ylabel="µs (median)",
        plot_name="gemma4-rmsnorm-residual-scalar",
        args={},
    )
)
def bench_rmsnorm(M, N, provider):
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    w = torch.randn(N, dtype=dtype, device="cuda")
    r = torch.randn(M, N, dtype=dtype, device="cuda")
    scalar = torch.tensor([0.7], dtype=torch.float32, device="cuda")
    eps = 1e-6
    fn = lambda: _triton_rmsnorm_residual_scalar(x, w, r, scalar, eps)
    ms = triton.testing.do_bench_cudagraph(fn)
    return ms * 1e3  # µs


# ---------------------------------------------------------------------------
# Benchmark 2: gemma4_qkv_rmsnorm
# ---------------------------------------------------------------------------
QKV_CONFIGS = (
    [(4, 8, 4, 128), (32, 8, 4, 128)]
    if IS_CI
    else list(itertools.product([1, 4, 32, 128, 512, 2048], [(8, 4, 128), (16, 8, 64)]))
)

QKV_FLAT = (
    [(m, nq, nkv, hd) for m, (nq, nkv, hd) in QKV_CONFIGS] if not IS_CI else QKV_CONFIGS
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "num_q", "num_kv", "head_dim"],
        x_vals=QKV_FLAT,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton (num_warps=head_dim//32, 2D grid)"],
        styles=[("blue", "-")],
        ylabel="µs (median)",
        plot_name="gemma4-qkv-rmsnorm",
        args={},
    )
)
def bench_qkv(M, num_q, num_kv, head_dim, provider):
    dtype = torch.bfloat16
    q = torch.randn(M, num_q * head_dim, dtype=dtype, device="cuda")
    k = torch.randn(M, num_kv * head_dim, dtype=dtype, device="cuda")
    v = torch.randn(M, num_kv * head_dim, dtype=dtype, device="cuda")
    q_w = torch.randn(head_dim, dtype=dtype, device="cuda")
    k_w = torch.randn(head_dim, dtype=dtype, device="cuda")

    def fn():
        qc, kc, vc = q.clone(), k.clone(), v.clone()
        _triton_qkv_rmsnorm(qc, kc, vc, q_w, k_w, num_q, num_kv, head_dim, 1e-6)

    ms = triton.testing.do_bench_cudagraph(fn)
    return ms * 1e3


# ---------------------------------------------------------------------------
# Benchmark 3: gemma4_routing_post_topk
# ---------------------------------------------------------------------------
ROUTING_TOPK_CONFIGS = (
    [(32, 8), (256, 8)]
    if IS_CI
    else list(itertools.product([1, 8, 32, 128, 512, 2048], [2, 4, 8]))
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "K"],
        x_vals=ROUTING_TOPK_CONFIGS,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("blue", "-")],
        ylabel="µs (median)",
        plot_name="gemma4-routing-post-topk",
        args={},
    )
)
def bench_routing_post_topk(B, K, provider):
    E = 256
    dtype = torch.bfloat16
    logits = torch.randn(B, K, dtype=dtype, device="cuda")
    ids = torch.randint(0, E, (B, K), dtype=torch.int64, device="cuda")
    scale = torch.rand(E, dtype=torch.float32, device="cuda") + 0.5
    fn = lambda: _triton_routing_post_topk(logits, ids, scale)
    ms = triton.testing.do_bench_cudagraph(fn)
    return ms * 1e3


# ---------------------------------------------------------------------------
# Benchmark 4: gemma4_fused_routing
# ---------------------------------------------------------------------------
FUSED_ROUTING_CONFIGS = (
    [(32, 256, 8), (256, 256, 8)]
    if IS_CI
    else list(itertools.product([1, 8, 32, 128, 512, 2048], [64, 128, 256], [4, 8]))
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T", "E", "K"],
        x_vals=FUSED_ROUTING_CONFIGS,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton (num_warps=1)"],
        styles=[("blue", "-")],
        ylabel="µs (median)",
        plot_name="gemma4-fused-routing",
        args={},
    )
)
def bench_fused_routing(T, E, K, provider):
    dtype = torch.bfloat16
    gating = torch.randn(T, E, dtype=dtype, device="cuda")
    scale = torch.rand(E, dtype=torch.float32, device="cuda") + 0.5
    fn = lambda: _triton_fused_routing(gating, scale, K)
    ms = triton.testing.do_bench_cudagraph(fn)
    return ms * 1e3


if __name__ == "__main__":
    print("=" * 60)
    print("Gemma4 fused ops: Triton benchmark")
    print("=" * 60)
    bench_rmsnorm.run(print_data=True)
    bench_qkv.run(print_data=True)
    bench_routing_post_topk.run(print_data=True)
    bench_fused_routing.run(print_data=True)
