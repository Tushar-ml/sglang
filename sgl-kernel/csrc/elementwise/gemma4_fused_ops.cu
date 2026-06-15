/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Fused CUDA kernels for Gemma4 decoder layer operations.
// Replaces gemma4_fused_ops.py Triton kernels with AOT CUDA equivalents.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "utils.h"

// ---------------------------------------------------------------------------
// Block-level sum reduction. Assumes blockDim.x == 256 (8 warps).
// Returns the total sum on every thread (broadcast via shared mem).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum_256(float val, float* smem) {
  // Warp reduce
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask);
  // Lane 0 of each warp writes partial sum
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = val;
  __syncthreads();
  // Thread 0 accumulates 8 warp sums and broadcasts
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int w = 0; w < 8; w++)
      t += smem[w];
    smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// Kernel 1: gemma_rmsnorm_residual_scalar
//   out[row] = (rmsnorm(x[row], w) + r[row]) * scalar
// blockDim = 256, grid = (M,)
// ---------------------------------------------------------------------------
template <typename T, bool HAS_SCALAR>
__global__ void gemma_rmsnorm_residual_scalar_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w,
    const T* __restrict__ r,
    const float* __restrict__ scalar,
    T* __restrict__ out,
    int64_t stride_x,
    int64_t stride_r,
    int64_t stride_o,
    int N,
    float eps) {
  __shared__ float s_warp[8];
  int tid = threadIdx.x;
  int row = blockIdx.x;

  float sum_sq = 0.0f;
  for (int i = tid; i < N; i += 256) {
    float v = (float)x[row * stride_x + i];
    sum_sq += v * v;
  }
  float rrms = rsqrtf(block_reduce_sum_256(sum_sq, s_warp) / N + eps);
  float sc = HAS_SCALAR ? *scalar : 1.0f;

  for (int i = tid; i < N; i += 256) {
    float xv = (float)x[row * stride_x + i];
    float wv = (float)w[i];
    float rv = (float)r[row * stride_r + i];
    out[row * stride_o + i] = (T)((xv * rrms * wv + rv) * sc);
  }
}

torch::Tensor
gemma4_rmsnorm_residual_scalar(torch::Tensor x, torch::Tensor w, torch::Tensor r, torch::Tensor scalar, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2 && x.stride(-1) == 1, "x must be contiguous 2D");
  int M = x.size(0), N = x.size(1);
  auto out = torch::empty_like(x);
  const auto stream = at::cuda::getCurrentCUDAStream();
  // scalar must be float32 (callers should ensure this)
  auto scalar_f32 = scalar.to(torch::kFloat32);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(x.scalar_type(), c_type, [&] {
    gemma_rmsnorm_residual_scalar_kernel<c_type, true><<<M, 256, 0, stream>>>(
        (const c_type*)x.data_ptr(),
        (const c_type*)w.data_ptr(),
        (const c_type*)r.data_ptr(),
        (const float*)scalar_f32.data_ptr(),
        (c_type*)out.data_ptr(),
        x.stride(0),
        r.stride(0),
        out.stride(0),
        N,
        (float)eps);
    return true;
  });
  return out;
}

// ---------------------------------------------------------------------------
// Kernel 2: gemma_dual_rmsnorm_residual_scalar
//   out = (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + r) * scalar
// Three separate block reductions; x1/x2/w1/w2 are read twice (no temp buffer).
// blockDim = 256, grid = (M,)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gemma_dual_rmsnorm_residual_scalar_kernel(
    const T* __restrict__ x1,
    const T* __restrict__ w1,
    const T* __restrict__ x2,
    const T* __restrict__ w2,
    const T* __restrict__ w3,
    const T* __restrict__ r,
    const float* __restrict__ scalar,
    T* __restrict__ out,
    int64_t sx1,
    int64_t sx2,
    int64_t sr,
    int64_t so,
    int N,
    float eps1,
    float eps2,
    float eps3) {
  __shared__ float s_warp[8];
  int tid = threadIdx.x;
  int row = blockIdx.x;

  // --- variance of x1 ---
  float sum = 0.0f;
  for (int i = tid; i < N; i += 256) {
    float v = (float)x1[row * sx1 + i];
    sum += v * v;
  }
  float rrms1 = rsqrtf(block_reduce_sum_256(sum, s_warp) / N + eps1);

  // --- variance of x2 ---
  sum = 0.0f;
  for (int i = tid; i < N; i += 256) {
    float v = (float)x2[row * sx2 + i];
    sum += v * v;
  }
  float rrms2 = rsqrtf(block_reduce_sum_256(sum, s_warp) / N + eps2);

  // --- variance of combined = norm1 + norm2 ---
  sum = 0.0f;
  for (int i = tid; i < N; i += 256) {
    float n1 = (float)x1[row * sx1 + i] * rrms1 * (float)w1[i];
    float n2 = (float)x2[row * sx2 + i] * rrms2 * (float)w2[i];
    float c = n1 + n2;
    sum += c * c;
  }
  float rrms3 = rsqrtf(block_reduce_sum_256(sum, s_warp) / N + eps3);

  // --- compute and store output ---
  float sc = *scalar;
  for (int i = tid; i < N; i += 256) {
    float n1 = (float)x1[row * sx1 + i] * rrms1 * (float)w1[i];
    float n2 = (float)x2[row * sx2 + i] * rrms2 * (float)w2[i];
    float comb = n1 + n2;
    float n3 = comb * rrms3 * (float)w3[i];
    float rv = (float)r[row * sr + i];
    out[row * so + i] = (T)((n3 + rv) * sc);
  }
}

torch::Tensor gemma4_dual_rmsnorm_residual_scalar(
    torch::Tensor x1,
    torch::Tensor w1,
    torch::Tensor x2,
    torch::Tensor w2,
    torch::Tensor w3,
    torch::Tensor r,
    torch::Tensor scalar,
    double eps1,
    double eps2,
    double eps3) {
  TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
  TORCH_CHECK(x1.dim() == 2 && x1.stride(-1) == 1, "x1 must be contiguous 2D");
  int M = x1.size(0), N = x1.size(1);
  auto out = torch::empty_like(x1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_f32 = scalar.to(torch::kFloat32);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(x1.scalar_type(), c_type, [&] {
    gemma_dual_rmsnorm_residual_scalar_kernel<c_type><<<M, 256, 0, stream>>>(
        (const c_type*)x1.data_ptr(),
        (const c_type*)w1.data_ptr(),
        (const c_type*)x2.data_ptr(),
        (const c_type*)w2.data_ptr(),
        (const c_type*)w3.data_ptr(),
        (const c_type*)r.data_ptr(),
        (const float*)scalar_f32.data_ptr(),
        (c_type*)out.data_ptr(),
        x1.stride(0),
        x2.stride(0),
        r.stride(0),
        out.stride(0),
        N,
        (float)eps1,
        (float)eps2,
        (float)eps3);
    return true;
  });
  return out;
}

// ---------------------------------------------------------------------------
// Kernel 3: gemma_qkv_rmsnorm
// In-place per-head RMSNorm on Q (weight), K (weight), V (no weight).
// One warp (32 threads) per head.
// ---------------------------------------------------------------------------

// Normalize one head in-place using a single warp.
template <typename T>
__device__ __forceinline__ void
rmsnorm_head_inplace(T* x_ptr, const T* w_ptr, bool has_weight, int head_dim, float eps) {
  int lane = threadIdx.x & 31;
  float sum_sq = 0.0f;
  for (int i = lane; i < head_dim; i += 32) {
    float v = (float)x_ptr[i];
    sum_sq += v * v;
  }
  for (int mask = 16; mask > 0; mask >>= 1)
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
  float rrms = rsqrtf(sum_sq / head_dim + eps);
  for (int i = lane; i < head_dim; i += 32) {
    float v = (float)x_ptr[i] * rrms;
    if (has_weight) v *= (float)w_ptr[i];
    x_ptr[i] = (T)v;
  }
}

// BY_HEAD=true:  grid=(M, total_heads), blockDim=32  [one warp per (token,head)]
// BY_HEAD=false: grid=(M,),             blockDim=32  [one warp per token, loops over heads]
template <typename T, bool BY_HEAD>
__global__ void gemma_qkv_rmsnorm_kernel(
    T* __restrict__ q,
    const T* __restrict__ q_w,
    T* __restrict__ k,
    const T* __restrict__ k_w,
    T* __restrict__ v,
    int64_t sq,
    int64_t sk,
    int64_t sv,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float eps,
    bool has_kv) {
  int m = blockIdx.x;

  if constexpr (BY_HEAD) {
    int h_all = blockIdx.y;
    if (h_all < num_q_heads) {
      rmsnorm_head_inplace<T>(q + m * sq + h_all * head_dim, q_w, true, head_dim, eps);
    } else if (has_kv && h_all < num_q_heads + num_kv_heads) {
      int h = h_all - num_q_heads;
      rmsnorm_head_inplace<T>(k + m * sk + h * head_dim, k_w, true, head_dim, eps);
    } else if (has_kv) {
      int h = h_all - num_q_heads - num_kv_heads;
      rmsnorm_head_inplace<T>(v + m * sv + h * head_dim, nullptr, false, head_dim, eps);
    }
  } else {
    for (int h = 0; h < num_q_heads; h++)
      rmsnorm_head_inplace<T>(q + m * sq + h * head_dim, q_w, true, head_dim, eps);
    if (has_kv) {
      for (int h = 0; h < num_kv_heads; h++)
        rmsnorm_head_inplace<T>(k + m * sk + h * head_dim, k_w, true, head_dim, eps);
      for (int h = 0; h < num_kv_heads; h++)
        rmsnorm_head_inplace<T>(v + m * sv + h * head_dim, nullptr, false, head_dim, eps);
    }
  }
}

void gemma4_qkv_rmsnorm(
    at::Tensor& q,
    const std::optional<at::Tensor>& k,
    const std::optional<at::Tensor>& v,
    const at::Tensor& q_w,
    const std::optional<at::Tensor>& k_w,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    double eps) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  int M = q.dim() >= 2 ? (int)q.size(0) : 1;
  bool has_kv = k.has_value() && v.has_value();
  int total_heads = (int)num_q_heads + (has_kv ? 2 * (int)num_kv_heads : 0);
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Gather raw pointers; use q as dummy when has_kv=false (kernel checks has_kv)
  void* k_ptr = has_kv ? k->data_ptr() : q.data_ptr();
  void* v_ptr = has_kv ? v->data_ptr() : q.data_ptr();
  void* k_w_ptr = (has_kv && k_w.has_value()) ? k_w->data_ptr() : q_w.data_ptr();
  int64_t sk = has_kv ? k->stride(0) : 0;
  int64_t sv = has_kv ? v->stride(0) : 0;

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(q.scalar_type(), c_type, [&] {
    if (M <= 256) {
      gemma_qkv_rmsnorm_kernel<c_type, true><<<dim3(M, total_heads), 32, 0, stream>>>(
          (c_type*)q.data_ptr(),
          (const c_type*)q_w.data_ptr(),
          (c_type*)k_ptr,
          (const c_type*)k_w_ptr,
          (c_type*)v_ptr,
          q.stride(0),
          sk,
          sv,
          (int)num_q_heads,
          (int)num_kv_heads,
          (int)head_dim,
          (float)eps,
          has_kv);
    } else {
      gemma_qkv_rmsnorm_kernel<c_type, false><<<M, 32, 0, stream>>>(
          (c_type*)q.data_ptr(),
          (const c_type*)q_w.data_ptr(),
          (c_type*)k_ptr,
          (const c_type*)k_w_ptr,
          (c_type*)v_ptr,
          q.stride(0),
          sk,
          sv,
          (int)num_q_heads,
          (int)num_kv_heads,
          (int)head_dim,
          (float)eps,
          has_kv);
    }
    return true;
  });
}
