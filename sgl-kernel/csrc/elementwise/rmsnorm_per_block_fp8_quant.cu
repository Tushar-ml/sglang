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

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

constexpr int kThreadsPerGroup = 16;
constexpr int kBlockThreads = 256;

template <int THREADS_PER_GROUP>
__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffffffff;
  static_assert(
      (THREADS_PER_GROUP & (THREADS_PER_GROUP - 1)) == 0 && THREADS_PER_GROUP <= 16 && THREADS_PER_GROUP >= 1,
      "THREADS_PER_GROUP must be 1, 2, 4, 8, or 16");
  if constexpr (THREADS_PER_GROUP >= 16) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  }
  if constexpr (THREADS_PER_GROUP >= 8) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  }
  if constexpr (THREADS_PER_GROUP >= 4) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  }
  if constexpr (THREADS_PER_GROUP >= 2) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  }
  return val;
}

template <
    typename T,
    typename DST_DTYPE,
    bool HAS_RESIDUAL,
    bool GEMMA_STYLE,
    bool SCALE_UE8M0,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void rmsnorm_per_block_fp8_quant_kernel(
    const T* __restrict__ input,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s,
    const int input_stride,
    const int residual_stride,
    const int hidden_dim,
    const int num_groups_per_row,
    const int group_size,
    const int scale_stride,
    const float eps,
    const float min_8bit,
    const float max_8bit) {
  const int row = blockIdx.x;
  const T* row_input = input + static_cast<int64_t>(row) * input_stride;

  float sum_sq = 0.f;
  for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    float x = static_cast<float>(row_input[col]);
    if constexpr (HAS_RESIDUAL) {
      x += static_cast<float>(residual[static_cast<int64_t>(row) * residual_stride + col]);
    }
    sum_sq += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, kBlockThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_tmp;
  sum_sq = BlockReduce(reduce_tmp).Sum(sum_sq);

  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf(sum_sq / hidden_dim + eps);
  }
  __syncthreads();

  const float inv_rms = s_inv_rms;
  const int lane_id = threadIdx.x % kThreadsPerGroup;
  const int local_group_slot = threadIdx.x / kThreadsPerGroup;
  const int groups_per_wave = blockDim.x / kThreadsPerGroup;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;
  const int32_t num_vec_elems = group_size / vec_size;

  for (int g_base = 0; g_base < num_groups_per_row; g_base += groups_per_wave) {
    const int g = g_base + local_group_slot;
    if (g >= num_groups_per_row) {
      continue;
    }

    const T* group_input = row_input + g * group_size;
    DST_DTYPE* group_output =
        static_cast<DST_DTYPE*>(output_q) + static_cast<int64_t>(row) * hidden_dim + g * group_size;

    scale_element_t* scale_output;
    if constexpr (SCALE_UE8M0) {
      const int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
      const int64_t global_group_id = static_cast<int64_t>(row) * num_groups_per_row + g;
      const int64_t row_idx = global_group_id / num_groups_per_row;
      const int col_idx_unpacked = global_group_id % num_groups_per_row;
      const int col_idx = col_idx_unpacked / num_elems_per_pack;
      const int pack_idx = col_idx_unpacked % num_elems_per_pack;
      scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                     (col_idx * scale_stride * num_elems_per_pack + row_idx * num_elems_per_pack + pack_idx);
    } else {
      const int64_t global_group_id = static_cast<int64_t>(row) * num_groups_per_row + g;
      scale_output = reinterpret_cast<scale_element_t*>(output_s) + global_group_id;
    }

    float local_absmax = eps;
    for (int32_t i = lane_id; i < num_vec_elems; i += kThreadsPerGroup) {
      vec_t input_vec;
      vec_t weight_vec;
      input_vec.cast_load(group_input + i * vec_size);
      weight_vec.cast_load(weight + g * group_size + i * vec_size);
#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float x = static_cast<float>(input_vec[j]);
        if constexpr (HAS_RESIDUAL) {
          x += static_cast<float>(
              residual[static_cast<int64_t>(row) * residual_stride + g * group_size + i * vec_size + j]);
        }
        const float w = static_cast<float>(weight_vec[j]);
        float y;
        if constexpr (GEMMA_STYLE) {
          y = x * inv_rms * (1.f + w);
        } else {
          y = x * inv_rms * w;
        }
        local_absmax = fmaxf(local_absmax, fabsf(y));
      }
    }

    local_absmax = GroupReduceMax<kThreadsPerGroup>(local_absmax, lane_id);

    float y_s = local_absmax / max_8bit;
    if constexpr (SCALE_UE8M0) {
      y_s = exp2f(ceilf(log2f(fmaxf(y_s, 1e-10f))));
    }

    scale_element_t y_s_quant;
    if constexpr (SCALE_UE8M0) {
      y_s_quant = static_cast<uint8_t>(static_cast<int>(log2f(y_s)) + 127);
    } else {
      y_s_quant = y_s;
    }

    if (lane_id == 0) {
      *scale_output = y_s_quant;
    }

    for (int32_t i = lane_id; i < num_vec_elems; i += kThreadsPerGroup) {
      vec_t input_vec;
      vec_t weight_vec;
      input_vec.cast_load(group_input + i * vec_size);
      weight_vec.cast_load(weight + g * group_size + i * vec_size);
#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float x = static_cast<float>(input_vec[j]);
        if constexpr (HAS_RESIDUAL) {
          x += static_cast<float>(
              residual[static_cast<int64_t>(row) * residual_stride + g * group_size + i * vec_size + j]);
        }
        const float w = static_cast<float>(weight_vec[j]);
        float y;
        if constexpr (GEMMA_STYLE) {
          y = x * inv_rms * (1.f + w);
        } else {
          y = x * inv_rms * w;
        }
        const float q_val = fminf(fmaxf(y / y_s, min_8bit), max_8bit);
        group_output[i * vec_size + j] = DST_DTYPE(q_val);
      }
    }
  }
}

// Column-major scale layout is required for UE8M0 DeepGEMM scales.
template <bool HAS_RESIDUAL, bool GEMMA_STYLE, bool SCALE_UE8M0>
static void launch_rmsnorm_per_block_fp8_quant(
    torch::Tensor input,
    const c10::optional<torch::Tensor>& residual,
    torch::Tensor weight,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit) {
  const int batch_size = input.size(0);
  const int hidden_dim = input.size(1);
  const int num_groups_per_row = hidden_dim / group_size;
  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_stride = output_s.stride(1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(batch_size);
  dim3 block(kBlockThreads);

  const void* residual_ptr = nullptr;
  int residual_stride = 0;
  if (HAS_RESIDUAL) {
    residual_ptr = residual.value().data_ptr();
    residual_stride = residual.value().stride(0);
  }

#define LAUNCH(SRC_DTYPE, DST_DTYPE)                                                             \
  do {                                                                                           \
    if constexpr (SCALE_UE8M0) {                                                                 \
      TORCH_CHECK(is_column_major, "UE8M0 scales must be column-major");                         \
      rmsnorm_per_block_fp8_quant_kernel<SRC_DTYPE, DST_DTYPE, HAS_RESIDUAL, GEMMA_STYLE, true>  \
          <<<grid, block, 0, stream>>>(                                                          \
              static_cast<const SRC_DTYPE*>(input.data_ptr()),                                   \
              static_cast<const SRC_DTYPE*>(residual_ptr),                                       \
              static_cast<const SRC_DTYPE*>(weight.data_ptr()),                                  \
              output_q.data_ptr(),                                                               \
              static_cast<uint32_t*>(output_s.data_ptr()),                                       \
              input.stride(0),                                                                   \
              residual_stride,                                                                   \
              hidden_dim,                                                                        \
              num_groups_per_row,                                                                \
              group_size,                                                                        \
              scale_stride,                                                                      \
              static_cast<float>(eps),                                                           \
              static_cast<float>(min_8bit),                                                      \
              static_cast<float>(max_8bit));                                                     \
    } else {                                                                                     \
      rmsnorm_per_block_fp8_quant_kernel<SRC_DTYPE, DST_DTYPE, HAS_RESIDUAL, GEMMA_STYLE, false> \
          <<<grid, block, 0, stream>>>(                                                          \
              static_cast<const SRC_DTYPE*>(input.data_ptr()),                                   \
              static_cast<const SRC_DTYPE*>(residual_ptr),                                       \
              static_cast<const SRC_DTYPE*>(weight.data_ptr()),                                  \
              output_q.data_ptr(),                                                               \
              static_cast<float*>(output_s.data_ptr()),                                          \
              input.stride(0),                                                                   \
              residual_stride,                                                                   \
              hidden_dim,                                                                        \
              num_groups_per_row,                                                                \
              group_size,                                                                        \
              scale_stride,                                                                      \
              static_cast<float>(eps),                                                           \
              static_cast<float>(min_8bit),                                                      \
              static_cast<float>(max_8bit));                                                     \
    }                                                                                            \
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (output_q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      LAUNCH(scalar_t, __nv_fp8_e4m3);
      return true;
    }
    return false;
  });

#undef LAUNCH
}

static void rmsnorm_per_block_fp8_quant_dispatch(
    torch::Tensor output_q,
    torch::Tensor output_s,
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    int64_t group_size,
    bool scale_ue8m0,
    bool gemma_style,
    const c10::optional<torch::Tensor>& residual) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  CHECK_EQ(input.size(1), weight.size(0));
  CHECK_EQ(input.size(1) % group_size, 0);
  CHECK_EQ(output_q.sizes(), input.sizes());
  CHECK_EQ(output_q.scalar_type(), at::ScalarType::Float8_e4m3fn);

  if (residual.has_value()) {
    CHECK_INPUT(residual.value());
    CHECK_DIM(2, residual.value());
    CHECK_EQ(residual.value().sizes(), input.sizes());
    if (scale_ue8m0) {
      if (gemma_style) {
        launch_rmsnorm_per_block_fp8_quant<true, true, true>(
            input, residual, weight, output_q, output_s, group_size, eps, -448.0, 448.0);
      } else {
        launch_rmsnorm_per_block_fp8_quant<true, false, true>(
            input, residual, weight, output_q, output_s, group_size, eps, -448.0, 448.0);
      }
    } else {
      if (gemma_style) {
        launch_rmsnorm_per_block_fp8_quant<true, true, false>(
            input, residual, weight, output_q, output_s, group_size, eps, -448.0, 448.0);
      } else {
        launch_rmsnorm_per_block_fp8_quant<true, false, false>(
            input, residual, weight, output_q, output_s, group_size, eps, -448.0, 448.0);
      }
    }
  } else {
    if (scale_ue8m0) {
      if (gemma_style) {
        launch_rmsnorm_per_block_fp8_quant<false, true, true>(
            input, residual, weight, output_q, output_s, group_size, eps, -448.0, 448.0);
      } else {
        launch_rmsnorm_per_block_fp8_quant<false, false, true>(
            input, residual, weight, output_q, output_s, group_size, eps, -448.0, 448.0);
      }
    } else {
      if (gemma_style) {
        launch_rmsnorm_per_block_fp8_quant<false, true, false>(
            input, residual, weight, output_q, output_s, group_size, eps, -448.0, 448.0);
      } else {
        launch_rmsnorm_per_block_fp8_quant<false, false, false>(
            input, residual, weight, output_q, output_s, group_size, eps, -448.0, 448.0);
      }
    }
  }
}

void rmsnorm_per_block_fp8_quant(
    torch::Tensor output_q,
    torch::Tensor output_s,
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    int64_t group_size,
    bool scale_ue8m0) {
  rmsnorm_per_block_fp8_quant_dispatch(
      output_q, output_s, input, weight, eps, group_size, scale_ue8m0, false, c10::nullopt);
}

void fused_add_rmsnorm_per_block_fp8_quant(
    torch::Tensor output_q,
    torch::Tensor output_s,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps,
    int64_t group_size,
    bool scale_ue8m0) {
  rmsnorm_per_block_fp8_quant_dispatch(
      output_q, output_s, input, weight, eps, group_size, scale_ue8m0, false, residual);
}

void gemma_rmsnorm_per_block_fp8_quant(
    torch::Tensor output_q,
    torch::Tensor output_s,
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    int64_t group_size,
    bool scale_ue8m0) {
  rmsnorm_per_block_fp8_quant_dispatch(
      output_q, output_s, input, weight, eps, group_size, scale_ue8m0, true, c10::nullopt);
}

void gemma_fused_add_rmsnorm_per_block_fp8_quant(
    torch::Tensor output_q,
    torch::Tensor output_s,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps,
    int64_t group_size,
    bool scale_ue8m0) {
  rmsnorm_per_block_fp8_quant_dispatch(output_q, output_s, input, weight, eps, group_size, scale_ue8m0, true, residual);
}
