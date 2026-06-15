"""Tests for gemma4_fused_ops CUDA kernels vs Triton reference."""

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip the whole module if CUDA is unavailable or sgl_kernel is not built
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

try:
    import sgl_kernel  # noqa: F401 — triggers op registration
except ImportError:
    pytest.skip("sgl_kernel not installed", allow_module_level=True)


DTYPES = [torch.float16, torch.bfloat16, torch.float32]
EPS = 1e-6


def rtol_atol(dtype):
    if dtype == torch.float32:
        return (1e-5, 1e-5)
    if dtype == torch.bfloat16:
        # bf16 has only 7 mantissa bits; one ULP at magnitude ~2 is ~0.004
        return (2e-3, 5e-3)
    return (1e-3, 1e-3)


# ---------------------------------------------------------------------------
# gemma4_rmsnorm_residual_scalar
# ---------------------------------------------------------------------------
class TestRMSNormResidualScalar:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("M,N", [(1, 512), (4, 2560), (32, 4096), (128, 2048)])
    def test_correctness(self, dtype, M, N):
        x = torch.randn(M, N, dtype=dtype, device="cuda")
        w = torch.randn(N, dtype=dtype, device="cuda")
        r = torch.randn(M, N, dtype=dtype, device="cuda")
        scalar = torch.tensor([0.7], dtype=torch.float32, device="cuda")

        # Reference: pure torch
        x_f = x.float()
        w_f = w.float()
        r_f = r.float()
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(var + EPS)
        ref = ((x_f * rrms * w_f + r_f) * 0.7).to(dtype)

        out = torch.ops.sgl_kernel.gemma4_rmsnorm_residual_scalar.default(
            x, w, r, scalar, EPS
        )
        torch.testing.assert_close(
            out, ref, **dict(zip(["rtol", "atol"], rtol_atol(dtype)))
        )


# ---------------------------------------------------------------------------
# gemma4_dual_rmsnorm_residual_scalar
# ---------------------------------------------------------------------------
class TestDualRMSNormResidualScalar:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("M,N", [(1, 512), (4, 2560), (16, 4096)])
    def test_correctness(self, dtype, M, N):
        x1 = torch.randn(M, N, dtype=dtype, device="cuda")
        w1 = torch.randn(N, dtype=dtype, device="cuda")
        x2 = torch.randn(M, N, dtype=dtype, device="cuda")
        w2 = torch.randn(N, dtype=dtype, device="cuda")
        w3 = torch.randn(N, dtype=dtype, device="cuda")
        r = torch.randn(M, N, dtype=dtype, device="cuda")
        scalar = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        def ref_dual(x1, w1, x2, w2, w3, r, sc, eps):
            x1f, w1f = x1.float(), w1.float()
            x2f, w2f = x2.float(), w2.float()
            w3f, rf = w3.float(), r.float()
            n1 = x1f * torch.rsqrt((x1f * x1f).mean(-1, keepdim=True) + eps) * w1f
            n2 = x2f * torch.rsqrt((x2f * x2f).mean(-1, keepdim=True) + eps) * w2f
            comb = n1 + n2
            n3 = comb * torch.rsqrt((comb * comb).mean(-1, keepdim=True) + eps) * w3f
            return ((n3 + rf) * sc).to(x1.dtype)

        ref = ref_dual(x1, w1, x2, w2, w3, r, 0.5, EPS)
        out = torch.ops.sgl_kernel.gemma4_dual_rmsnorm_residual_scalar.default(
            x1,
            w1,
            x2,
            w2,
            w3,
            r,
            torch.tensor([0.5], dtype=torch.float32, device="cuda"),
            EPS,
            EPS,
            EPS,
        )
        torch.testing.assert_close(
            out, ref, **dict(zip(["rtol", "atol"], rtol_atol(dtype)))
        )


# ---------------------------------------------------------------------------
# gemma4_qkv_rmsnorm
# ---------------------------------------------------------------------------
class TestQKVRMSNorm:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "M,num_q,num_kv,head_dim",
        [
            (1, 8, 4, 128),
            (4, 8, 4, 128),
            (256, 8, 4, 128),
            (512, 16, 8, 64),
        ],
    )
    def test_correctness_with_kv(self, dtype, M, num_q, num_kv, head_dim):
        q = torch.randn(M, num_q * head_dim, dtype=dtype, device="cuda")
        k = torch.randn(M, num_kv * head_dim, dtype=dtype, device="cuda")
        v = torch.randn(M, num_kv * head_dim, dtype=dtype, device="cuda")
        q_w = torch.randn(head_dim, dtype=dtype, device="cuda")
        k_w = torch.randn(head_dim, dtype=dtype, device="cuda")

        def ref_head_norm(x, w, has_w, eps=EPS):
            xf = x.float()
            var = (xf * xf).mean(-1, keepdim=True)
            rrms = torch.rsqrt(var + eps)
            out = xf * rrms
            if has_w:
                out = out * w.float()
            return out.to(x.dtype)

        # Build reference using view/unflatten
        q_ref = q.clone().unflatten(-1, (num_q, head_dim))
        q_ref = ref_head_norm(q_ref, q_w, True).flatten(-2)
        k_ref = k.clone().unflatten(-1, (num_kv, head_dim))
        k_ref = ref_head_norm(k_ref, k_w, True).flatten(-2)
        v_ref = v.clone().unflatten(-1, (num_kv, head_dim))
        v_ref = ref_head_norm(v_ref, None, False).flatten(-2)

        q_out = q.clone()
        k_out = k.clone()
        v_out = v.clone()
        torch.ops.sgl_kernel.gemma4_qkv_rmsnorm.default(
            q_out, k_out, v_out, q_w, k_w, num_q, num_kv, head_dim, EPS
        )
        rtol, atol = rtol_atol(dtype)
        torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(v_out, v_ref, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_q_only(self, dtype):
        M, num_q, head_dim = 4, 8, 128
        q = torch.randn(M, num_q * head_dim, dtype=dtype, device="cuda")
        q_w = torch.randn(head_dim, dtype=dtype, device="cuda")

        def ref_head_norm(x, w, eps=EPS):
            xf = x.float()
            var = (xf * xf).mean(-1, keepdim=True)
            return (xf * torch.rsqrt(var + eps) * w.float()).to(x.dtype)

        q_ref = ref_head_norm(q.clone().unflatten(-1, (num_q, head_dim)), q_w).flatten(
            -2
        )
        q_out = q.clone()
        torch.ops.sgl_kernel.gemma4_qkv_rmsnorm.default(
            q_out, None, None, q_w, None, num_q, 0, head_dim, EPS
        )
        torch.testing.assert_close(
            q_out, q_ref, **dict(zip(["rtol", "atol"], rtol_atol(dtype)))
        )


# ---------------------------------------------------------------------------
# gemma4_routing_post_topk
# ---------------------------------------------------------------------------
class TestRoutingPostTopK:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("B,K,E", [(1, 2, 64), (8, 8, 256), (32, 4, 128)])
    def test_correctness(self, dtype, B, K, E):
        logits = torch.randn(B, K, dtype=dtype, device="cuda")
        ids = torch.randint(0, E, (B, K), dtype=torch.int64, device="cuda")
        scale = torch.rand(E, dtype=torch.float32, device="cuda") + 0.5

        # Reference
        logits_f = logits.float()
        softmax = torch.softmax(logits_f, dim=-1)
        sc = scale[ids.long()]
        ref_w = (softmax * sc).float()
        ref_ids = ids.int()

        w, out_ids = torch.ops.sgl_kernel.gemma4_routing_post_topk.default(
            logits, ids, scale
        )
        torch.testing.assert_close(w, ref_w, rtol=1e-3, atol=1e-4)
        assert (out_ids == ref_ids).all()


# ---------------------------------------------------------------------------
# gemma4_fused_routing
# ---------------------------------------------------------------------------
class TestFusedRouting:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "T,E,K",
        [
            (1, 64, 2),
            (4, 256, 8),
            (32, 256, 4),
            (1, 128, 4),
            (8, 512, 8),
        ],
    )
    def test_correctness(self, dtype, T, E, K):
        gating = torch.randn(T, E, dtype=dtype, device="cuda")
        scale = torch.rand(E, dtype=torch.float32, device="cuda") + 0.5

        # Reference: topk → softmax → scale gather
        logits_f = gating.float()
        topk_logits, topk_ids = torch.topk(logits_f, k=K, dim=-1)
        softmax = torch.softmax(topk_logits, dim=-1)
        ref_w = (softmax * scale[topk_ids]).float()
        ref_ids = topk_ids.int()

        w, out_ids = torch.ops.sgl_kernel.gemma4_fused_routing.default(gating, scale, K)

        # Sort both by expert_id before comparing to avoid tie-breaking differences.
        # When two experts have equal bf16 logits their order is implementation-defined,
        # so we compare the SET of selected experts and their corresponding weights.
        ref_order = ref_ids.argsort(dim=-1)
        out_order = out_ids.argsort(dim=-1)
        ref_ids_sorted = ref_ids.gather(1, ref_order)
        out_ids_sorted = out_ids.gather(1, out_order)
        ref_w_sorted = ref_w.gather(1, ref_order)
        out_w_sorted = w.gather(1, out_order)
        assert (
            out_ids_sorted == ref_ids_sorted
        ).all(), (
            f"Top-K expert sets differ:\nref={ref_ids_sorted}\nout={out_ids_sorted}"
        )
        torch.testing.assert_close(out_w_sorted, ref_w_sorted, rtol=1e-3, atol=1e-4)

    def test_empty_tokens(self):
        gating = torch.empty(0, 256, dtype=torch.bfloat16, device="cuda")
        scale = torch.ones(256, dtype=torch.float32, device="cuda")
        w, ids = torch.ops.sgl_kernel.gemma4_fused_routing.default(gating, scale, 8)
        assert w.shape == (0, 8)
        assert ids.shape == (0, 8)

    def test_large_e(self):
        T, E, K = 4, 1024, 8
        gating = torch.randn(T, E, dtype=torch.float16, device="cuda")
        scale = torch.rand(E, dtype=torch.float32, device="cuda") + 0.5
        logits_f = gating.float()
        topk_logits, topk_ids = torch.topk(logits_f, k=K, dim=-1)
        ref_w = (torch.softmax(topk_logits, dim=-1) * scale[topk_ids]).float()
        ref_ids = topk_ids.int()
        w, out_ids = torch.ops.sgl_kernel.gemma4_fused_routing.default(gating, scale, K)
        # Sort by expert_id for tie-agnostic comparison
        ref_order = ref_ids.argsort(dim=-1)
        out_order = out_ids.argsort(dim=-1)
        assert (ref_ids.gather(1, ref_order) == out_ids.gather(1, out_order)).all()
        torch.testing.assert_close(
            ref_w.gather(1, ref_order), w.gather(1, out_order), rtol=1e-3, atol=1e-4
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
