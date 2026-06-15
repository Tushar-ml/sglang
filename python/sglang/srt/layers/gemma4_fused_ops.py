"""Fused ops for Gemma4 decoder layer operations.

All five operations are implemented as autotuned Triton kernels.
The QKV and routing kernels use a launcher-level num_warps heuristic
(autotune on constexpr-headed kernels is not possible via @triton.autotune).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------


def gemma_rmsnorm_residual_scalar(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(x) + residual) * scalar."""
    return _triton_rmsnorm_residual_scalar(x, weight, residual, scalar, eps)


def gemma_dual_rmsnorm_residual_scalar(
    x1: torch.Tensor,
    weight1: torch.Tensor,
    x2: torch.Tensor,
    weight2: torch.Tensor,
    weight3: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
    eps3: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar."""
    return _triton_dual_rmsnorm_residual_scalar(
        x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
    )


def gemma_qkv_rmsnorm(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> None:
    """In-place fused RMSNorm on Q, K, V for Gemma4 attention."""
    _triton_qkv_rmsnorm(
        q, k, v, q_weight, k_weight, num_q_heads, num_kv_heads, head_dim, eps
    )


def gemma_routing_post_topk(
    topk_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax + scale-gather + casts for Gemma4 routing."""
    return _triton_routing_post_topk(topk_logits, topk_ids, per_expert_scale)


def gemma4_fused_routing(
    gating_output: torch.Tensor,
    per_expert_scale: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-launch Gemma4 router."""
    return _triton_fused_routing(gating_output, per_expert_scale, topk)


# ---------------------------------------------------------------------------
# Triton kernel launchers and definitions
# ---------------------------------------------------------------------------


def _m_bucket(M: int) -> int:
    """Coarse-bucket M for autotune key — avoids per-row retuning."""
    if M <= 1:
        return 1
    elif M <= 8:
        return 8
    elif M <= 64:
        return 64
    elif M <= 512:
        return 512
    else:
        return 4096


def _triton_rmsnorm_residual_scalar(x, weight, residual, scalar, eps):
    assert x.dim() == 2 and x.stride(-1) == 1, "Expected contiguous 2D input"
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x)
    _gemma_rmsnorm_residual_kernel[(M,)](
        x,
        weight,
        residual,
        scalar,
        out,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        _m_bucket(M),
        N,
        eps,
        HAS_SCALAR=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _triton_dual_rmsnorm_residual_scalar(
    x1, w1, x2, w2, w3, r, scalar, eps1, eps2, eps3
):
    assert x1.dim() == 2 and x1.stride(-1) == 1
    M, N = x1.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x1)
    _gemma_dual_rmsnorm_residual_kernel[(M,)](
        x1,
        w1,
        x2,
        w2,
        w3,
        r,
        scalar,
        out,
        x1.stride(0),
        x2.stride(0),
        r.stride(0),
        out.stride(0),
        _m_bucket(M),
        N,
        eps1,
        eps2,
        eps3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _triton_qkv_rmsnorm(
    q, k, v, q_weight, k_weight, num_q_heads, num_kv_heads, head_dim, eps
):
    assert q.stride(-1) == 1
    M = q.shape[0] if q.dim() >= 2 else 1
    BLOCK = triton.next_power_of_2(head_dim)
    # One warp per 32 elements gives good occupancy without excess sync.
    num_warps = min(max(head_dim // 32, 1), 32)
    has_kv = k is not None and v is not None
    if has_kv:
        assert k.stride(-1) == 1 and v.stride(-1) == 1
        assert k_weight is not None and k_weight.shape[-1] == head_dim
    if M <= 256:
        total_heads = num_q_heads + (2 * num_kv_heads if has_kv else 0)
        _gemma_qkv_rmsnorm_kernel[(M, total_heads)](
            q,
            k if has_kv else q,
            v if has_kv else q,
            q_weight,
            k_weight if has_kv else q_weight,
            q.stride(0),
            k.stride(0) if has_kv else 0,
            v.stride(0) if has_kv else 0,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads if has_kv else 0,
            HEAD_DIM=head_dim,
            eps=eps,
            HAS_KV=has_kv,
            BY_HEAD=True,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return
    _gemma_qkv_rmsnorm_kernel[(M,)](
        q,
        k if has_kv else q,
        v if has_kv else q,
        q_weight,
        k_weight if has_kv else q_weight,
        q.stride(0),
        k.stride(0) if has_kv else 0,
        v.stride(0) if has_kv else 0,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads if has_kv else 0,
        HEAD_DIM=head_dim,
        eps=eps,
        HAS_KV=has_kv,
        BY_HEAD=False,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )


def _triton_routing_post_topk(topk_logits, topk_ids, per_expert_scale):
    B, K = topk_logits.shape
    BLOCK_K = triton.next_power_of_2(K)
    out_weights = torch.empty((B, K), dtype=torch.float32, device=topk_logits.device)
    out_ids = torch.empty((B, K), dtype=torch.int32, device=topk_logits.device)
    _gemma_routing_post_topk_kernel[(B,)](
        topk_logits,
        topk_ids,
        per_expert_scale,
        out_weights,
        out_ids,
        topk_logits.stride(0),
        out_weights.stride(0),
        out_ids.stride(0),
        K=K,
        BLOCK_K=BLOCK_K,
    )
    return out_weights, out_ids


def _triton_fused_routing(gating_output, per_expert_scale, topk):
    T, E = gating_output.shape
    assert topk <= E
    assert E <= 1024
    BLOCK_E = triton.next_power_of_2(E)
    topk_weights = torch.empty(
        (T, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty((T, topk), dtype=torch.int32, device=gating_output.device)
    if T == 0:
        return topk_weights, topk_ids
    # tl.sort is most efficient with few warps: 1 warp per 256 experts.
    num_warps = max(1, BLOCK_E // 256)
    _gemma4_routing_kernel[(T,)](
        gating_output,
        per_expert_scale,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        E=E,
        K=topk,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps,
    )
    return topk_weights, topk_ids


# -----------------------------------------------------------------------
# Kernel: _gemma_rmsnorm_residual_kernel
# Autotuned on num_warps/num_stages keyed by (M_BUCKET, N).
# M_BUCKET discretises the batch size so decode (M≤1) and large-prefill
# (M≥512) each get their own optimal warp count.
# BLOCK_SIZE is caller-computed as next_power_of_2(N).
# -----------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [1, 2, 4, 8, 16, 32]
        for ns in [1, 2, 3]
    ],
    key=["M_BUCKET", "N"],
)
@triton.jit
def _gemma_rmsnorm_residual_kernel(
    X_ptr,
    W_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x,
    stride_r,
    stride_o,
    M_BUCKET,  # coarse M bucket — autotune key only, not used in body
    N,
    eps,
    HAS_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    var = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    out = x * rrms * w + r
    if HAS_SCALAR:
        scalar = tl.load(Scalar_ptr).to(tl.float32)
        out = out * scalar
    tl.store(Out_ptr + row * stride_o + cols, out.to(x.dtype), mask=mask)


# -----------------------------------------------------------------------
# Kernel: _gemma_dual_rmsnorm_residual_kernel
# -----------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [1, 2, 4, 8, 16, 32]
        for ns in [1, 2, 3]
    ],
    key=["M_BUCKET", "N"],
)
@triton.jit
def _gemma_dual_rmsnorm_residual_kernel(
    X1_ptr,
    W1_ptr,
    X2_ptr,
    W2_ptr,
    W3_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x1,
    stride_x2,
    stride_r,
    stride_o,
    M_BUCKET,  # autotune key only
    N,
    eps1,
    eps2,
    eps3,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x1 = tl.load(X1_ptr + row * stride_x1 + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X2_ptr + row * stride_x2 + cols, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    var1 = tl.sum(x1 * x1, axis=0) / N
    norm1 = x1 * tl.rsqrt(var1 + eps1) * w1
    var2 = tl.sum(x2 * x2, axis=0) / N
    norm2 = x2 * tl.rsqrt(var2 + eps2) * w2
    combined = norm1 + norm2
    var3 = tl.sum(combined * combined, axis=0) / N
    norm3 = combined * tl.rsqrt(var3 + eps3) * w3
    scalar = tl.load(Scalar_ptr).to(tl.float32)
    out = (norm3 + r) * scalar
    tl.store(Out_ptr + row * stride_o + cols, out.to(x1.dtype), mask=mask)


# -----------------------------------------------------------------------
# Kernel: _gemma_qkv_rmsnorm_kernel
# num_warps set by launcher heuristic (head_dim // 32).
# No autotune: NUM_Q/KV_HEADS are tl.constexpr required by tl.static_range.
# -----------------------------------------------------------------------


@triton.jit
def _gemma_qkv_rmsnorm_store(
    X_ptr,
    W_ptr,
    stride_m,
    m,
    h,
    cols,
    mask,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_WEIGHT: tl.constexpr,
):
    off = m * stride_m + h * HEAD_DIM + cols
    x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
    out = x * rrms
    if HAS_WEIGHT:
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = out * w
    tl.store(X_ptr + off, out.to(X_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gemma_qkv_rmsnorm_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_w_ptr,
    K_w_ptr,
    stride_q_m,
    stride_k_m,
    stride_v_m,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_KV: tl.constexpr,
    BY_HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM
    if BY_HEAD:
        h_all = tl.program_id(1)
        if h_all < NUM_Q_HEADS:
            _gemma_qkv_rmsnorm_store(
                Q_ptr,
                Q_w_ptr,
                stride_q_m,
                m,
                h_all,
                cols,
                mask,
                HEAD_DIM,
                eps,
                True,
            )
        elif HAS_KV and h_all < NUM_Q_HEADS + NUM_KV_HEADS:
            h = h_all - NUM_Q_HEADS
            _gemma_qkv_rmsnorm_store(
                K_ptr, K_w_ptr, stride_k_m, m, h, cols, mask, HEAD_DIM, eps, True
            )
        elif HAS_KV:
            h = h_all - NUM_Q_HEADS - NUM_KV_HEADS
            _gemma_qkv_rmsnorm_store(
                V_ptr, Q_w_ptr, stride_v_m, m, h, cols, mask, HEAD_DIM, eps, False
            )
    else:
        for h in tl.static_range(NUM_Q_HEADS):
            _gemma_qkv_rmsnorm_store(
                Q_ptr, Q_w_ptr, stride_q_m, m, h, cols, mask, HEAD_DIM, eps, True
            )
        if HAS_KV:
            for h in tl.static_range(NUM_KV_HEADS):
                _gemma_qkv_rmsnorm_store(
                    K_ptr,
                    K_w_ptr,
                    stride_k_m,
                    m,
                    h,
                    cols,
                    mask,
                    HEAD_DIM,
                    eps,
                    True,
                )
            for h in tl.static_range(NUM_KV_HEADS):
                _gemma_qkv_rmsnorm_store(
                    V_ptr,
                    Q_w_ptr,
                    stride_v_m,
                    m,
                    h,
                    cols,
                    mask,
                    HEAD_DIM,
                    eps,
                    False,
                )


# -----------------------------------------------------------------------
# Kernel: _gemma_routing_post_topk_kernel
# -----------------------------------------------------------------------


@triton.jit
def _gemma_routing_post_topk_kernel(
    Logits_ptr,
    Ids_ptr,
    Scale_ptr,
    Out_weights_ptr,
    Out_ids_ptr,
    stride_l,
    stride_ow,
    stride_oi,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    logits = tl.load(
        Logits_ptr + row * stride_l + cols, mask=mask, other=float("-inf")
    ).to(tl.float32)
    ids_i64 = tl.load(Ids_ptr + row * stride_l + cols, mask=mask, other=0)
    max_val = tl.max(logits, axis=0)
    exp_val = tl.exp(logits - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    weights = exp_val / sum_exp
    scale = tl.load(Scale_ptr + ids_i64, mask=mask, other=1.0).to(tl.float32)
    weights = weights * scale
    tl.store(Out_weights_ptr + row * stride_ow + cols, weights, mask=mask)
    tl.store(Out_ids_ptr + row * stride_oi + cols, ids_i64.to(tl.int32), mask=mask)


# -----------------------------------------------------------------------
# Kernel: _gemma4_routing_kernel
# num_warps set by launcher heuristic (BLOCK_E // 256, min 1).
# E and K are runtime args (non-constexpr) — BLOCK_E stays constexpr
# because it is used in tl.arange.
# -----------------------------------------------------------------------


@triton.jit
def _gemma4_routing_kernel(
    gating_ptr,
    per_expert_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    stride_g_t,
    E,  # runtime — used for valid mask
    K,  # runtime — used for top-K mask and store offset
    BLOCK_E: tl.constexpr,  # caller-computed next_power_of_2(E)
):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E
    logits = tl.load(
        gating_ptr + pid * stride_g_t + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    MIN32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign = logit_bits >> 31
    key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ MIN32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    sk64 = key.to(tl.int64) & 0x00000000FFFFFFFF
    packed = (sk64 << 32) | offs_e.to(tl.int64)
    sorted_p = tl.sort(packed, descending=False)
    all_keys = ((sorted_p >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_p & 0x00000000FFFFFFFF).to(tl.int32)
    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ MIN32)
    all_logits = all_bits.to(tl.float32, bitcast=True)
    top_mask = offs_e < K
    max_l = tl.max(tl.where(top_mask, all_logits, -float("inf")), axis=0)
    raw_exp = tl.where(top_mask, tl.exp(all_logits - max_l), 0.0)
    denom = tl.sum(raw_exp, axis=0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    weights = raw_exp / denom
    scales = tl.load(
        per_expert_scale_ptr + all_ids.to(tl.int64),
        mask=top_mask,
        other=1.0,
    ).to(tl.float32)
    weights = weights * scales
    base_off = pid * K + offs_e
    tl.store(topk_weights_ptr + base_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + base_off, all_ids, mask=top_mask)
