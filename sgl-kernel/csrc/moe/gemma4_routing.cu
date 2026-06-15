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

// CUDA kernels for Gemma4 MoE routing.
// Replaces gemma4_fused_ops.py Triton routing kernels.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cfloat>
#include <tuple>
#include <vector>

#ifndef USE_ROCM
#include <cub/cub.cuh>
#else
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "utils.h"

// ---------------------------------------------------------------------------
// Kernel 4: gemma_routing_post_topk
// Fused softmax(topk_logits) * per_expert_scale[topk_ids] → (f32, i32).
// One block per token; one warp (blockDim=32); K ≤ 32.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gemma_routing_post_topk_kernel(
    const T* __restrict__ logits_ptr,
    const int64_t* __restrict__ ids_ptr,
    const float* __restrict__ scale_ptr,
    float* __restrict__ out_weights,
    int32_t* __restrict__ out_ids,
    int stride_l,
    int stride_ow,
    int stride_oi,
    int K) {
  int row = blockIdx.x;
  int tid = threadIdx.x;  // 0..31

  float logit = (tid < K) ? (float)logits_ptr[row * stride_l + tid] : -FLT_MAX;
  int64_t id_val = (tid < K) ? ids_ptr[row * stride_l + tid] : 0;

  // Stable warp-level softmax
  float max_l = logit;
  for (int mask = 16; mask > 0; mask >>= 1)
    max_l = fmaxf(max_l, __shfl_xor_sync(0xffffffff, max_l, mask));

  float exp_v = (tid < K) ? expf(logit - max_l) : 0.0f;
  float sum_exp = exp_v;
  for (int mask = 16; mask > 0; mask >>= 1)
    sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, mask);

  if (tid < K) {
    float weight = exp_v / sum_exp;
    weight *= scale_ptr[id_val];
    out_weights[row * stride_ow + tid] = weight;
    out_ids[row * stride_oi + tid] = (int32_t)id_val;
  }
}

std::tuple<at::Tensor, at::Tensor>
gemma4_routing_post_topk(torch::Tensor topk_logits, torch::Tensor topk_ids, torch::Tensor per_expert_scale) {
  TORCH_CHECK(topk_logits.is_cuda(), "topk_logits must be a CUDA tensor");
  int B = (int)topk_logits.size(0);
  int K = (int)topk_logits.size(1);
  TORCH_CHECK(K <= 32, "K must be <= 32 for gemma4_routing_post_topk CUDA kernel");

  auto out_weights = torch::empty({B, K}, topk_logits.options().dtype(torch::kFloat32));
  auto out_ids = torch::empty({B, K}, topk_logits.options().dtype(torch::kInt32));
  auto scale_f32 = per_expert_scale.to(torch::kFloat32).contiguous();
  const auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(topk_logits.scalar_type(), c_type, [&] {
    gemma_routing_post_topk_kernel<c_type><<<B, 32, 0, stream>>>(
        (const c_type*)topk_logits.data_ptr(),
        (const int64_t*)topk_ids.data_ptr(),
        (const float*)scale_f32.data_ptr(),
        (float*)out_weights.data_ptr(),
        (int32_t*)out_ids.data_ptr(),
        (int)topk_logits.stride(0),
        (int)out_weights.stride(0),
        (int)out_ids.stride(0),
        K);
    return true;
  });
  return {out_weights, out_ids};  // implicitly constructs std::tuple
}

// ---------------------------------------------------------------------------
// Kernel 5: gemma4_fused_routing
// One-pass router: topk(gating[T,E]) + softmax(top-K logits) + scale.
//
// Sort trick (matching Triton implementation):
//   key = (bits & 0x80000000) ? (bits ^ 0x80000000) : ~bits
//   packed = (zero_extend_key << 32) | expert_id
// Sorted ascending as signed int64 → descending float order.
//
// blockDim = BLOCK_E (next power of 2 ≥ E), K ≤ 32.
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_E>
__launch_bounds__(BLOCK_E) __global__ void gemma4_fused_routing_kernel(
    const T* __restrict__ gating_ptr,
    const float* __restrict__ scale_ptr,
    float* __restrict__ topk_weights_ptr,
    int32_t* __restrict__ topk_ids_ptr,
    int stride_g_t,
    int E,
    int K) {
  using BlockSort = cub::BlockRadixSort<int64_t, BLOCK_E, 1>;
  __shared__ typename BlockSort::TempStorage sort_storage;
  __shared__ int s_ids[32];  // K <= 32

  int pid = blockIdx.x;
  int tid = threadIdx.x;

  // Pack (sort_key, expert_id) into int64.
  // Key bijection: ascending int64 sort → descending float logit order.
  // Positive floats (bits MSB=0): key = ~bits → upper 32 bits of packed are
  //   in [0x80000000, 0xFFFFFFFF], making packed a negative int64 (sorts before
  //   positive packed values from negative floats).
  // Negative floats (bits MSB=1): key = bits ^ 0x80000000 → upper 32 bits in
  //   [0x00000000, 0x7FFFFFFF], making packed a positive int64 (sorts after).
  int64_t packed;
  if (tid < E) {
    float logit = (float)gating_ptr[pid * stride_g_t + tid];
    uint32_t bits = __float_as_uint(logit);
    uint32_t key = (bits & 0x80000000u) ? (bits ^ 0x80000000u) : (~bits);
    // Zero-extend key to int64 (avoid sign extension) then shift to upper 32 bits
    packed = ((int64_t)(uint64_t)key << 32) | (int64_t)tid;
  } else {
    packed = (int64_t)0x7FFFFFFFFFFFFFFF;  // max int64 sentinel → sorts last
  }

  // Ascending CUB sort: thread 0 ends up with largest logit (smallest key)
  // BlockRadixSort requires array references even for ITEMS_PER_THREAD=1
  int64_t keys[1] = {packed};
  BlockSort(sort_storage).Sort(keys);
  packed = keys[0];

  // Store top-K expert IDs to shared memory
  if (tid < K) {
    s_ids[tid] = (int)(packed & 0xFFFFFFFF);
  }
  __syncthreads();

  // First warp (threads 0..31) computes softmax + scale for top-K
  if (tid < 32) {
    float logit = -FLT_MAX;
    int expert_id = 0;
    if (tid < K) {
      expert_id = s_ids[tid];
      logit = (float)gating_ptr[pid * stride_g_t + expert_id];
    }

    // Stable warp-level softmax over top-K
    float max_l = logit;
    for (int mask = 16; mask > 0; mask >>= 1)
      max_l = fmaxf(max_l, __shfl_xor_sync(0xffffffff, max_l, mask));

    float exp_v = (tid < K) ? expf(logit - max_l) : 0.0f;
    float sum_exp = exp_v;
    for (int mask = 16; mask > 0; mask >>= 1)
      sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, mask);

    if (tid < K) {
      float weight = exp_v / sum_exp;
      weight *= scale_ptr[expert_id];
      topk_weights_ptr[pid * K + tid] = weight;
      topk_ids_ptr[pid * K + tid] = (int32_t)expert_id;
    }
  }
}

std::tuple<at::Tensor, at::Tensor>
gemma4_fused_routing(torch::Tensor gating_output, torch::Tensor per_expert_scale, int64_t topk) {
  TORCH_CHECK(gating_output.is_cuda(), "gating_output must be a CUDA tensor");
  TORCH_CHECK(gating_output.dim() == 2, "gating_output must be 2D [T, E]");
  int T = (int)gating_output.size(0);
  int E = (int)gating_output.size(1);
  int K = (int)topk;
  TORCH_CHECK(E <= 1024, "E must be <= 1024, got ", E);
  TORCH_CHECK(K <= 32, "K must be <= 32 for gemma4_fused_routing CUDA kernel, got ", K);
  TORCH_CHECK(K <= E, "K must be <= E");

  auto out_weights = torch::empty({T, K}, gating_output.options().dtype(torch::kFloat32));
  auto out_ids = torch::empty({T, K}, gating_output.options().dtype(torch::kInt32));

  if (T == 0) return {out_weights, out_ids};

  auto scale_f32 = per_expert_scale.to(torch::kFloat32).contiguous();
  auto gating_contig = gating_output.contiguous();
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Compute next power of 2 >= E for BLOCK_E
  int block_e = 1;
  while (block_e < E)
    block_e <<= 1;

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(gating_output.scalar_type(), c_type, [&] {
    auto* g = (const c_type*)gating_contig.data_ptr();
    auto* sc = (const float*)scale_f32.data_ptr();
    auto* w = (float*)out_weights.data_ptr();
    auto* ids = (int32_t*)out_ids.data_ptr();
    int st = (int)gating_contig.stride(0);

    if (block_e <= 64) {
      gemma4_fused_routing_kernel<c_type, 64><<<T, 64, 0, stream>>>(g, sc, w, ids, st, E, K);
    } else if (block_e <= 128) {
      gemma4_fused_routing_kernel<c_type, 128><<<T, 128, 0, stream>>>(g, sc, w, ids, st, E, K);
    } else if (block_e <= 256) {
      gemma4_fused_routing_kernel<c_type, 256><<<T, 256, 0, stream>>>(g, sc, w, ids, st, E, K);
    } else if (block_e <= 512) {
      gemma4_fused_routing_kernel<c_type, 512><<<T, 512, 0, stream>>>(g, sc, w, ids, st, E, K);
    } else {
      gemma4_fused_routing_kernel<c_type, 1024><<<T, 1024, 0, stream>>>(g, sc, w, ids, st, E, K);
    }
    return true;
  });
  return {out_weights, out_ids};
}
