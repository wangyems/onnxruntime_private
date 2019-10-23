/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

// The code below is mostly copied from Pytorch PersistentSoftmax.cuh

#include "core/providers/cuda/cu_inc/common.cuh"
#include "softmax_impl.h"

#include <assert.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <limits>
#include <stdint.h>

namespace onnxruntime {
namespace cuda {

constexpr int CUDA_WARP_SIZE = 32;

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
  ReduceOp<acc_t> r;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
      sum[i] = r(sum[i], b);
    }
  }
}

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. element_count can be any integer value <= 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "CUDA_WARP_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architecures, but there is no guarantee this will not change for future arch.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < CUDA_WARP_SIZE) ? next_power_of_two : CUDA_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the batch
  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
  // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
  // the nested loops.
  // This should have no impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = src[i * element_count + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      if (is_log_softmax) {
        sum[i] += std::exp((float)(elements[i][it] - max_value[i]));
      } else {
        elements[i][it] = std::exp((float)(elements[i][it] - max_value[i]));
        sum[i] += elements[i][it];
      }
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
    //if (is_log_softmax) sum[i] = max_value[i] + std::log(sum[i]);
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        if (is_log_softmax) {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] - sum[i];
        } else {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        }
      } else {
        break;
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_backward(output_t* gradInput, const input_t* grad, const input_t* output, int batch_size, int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_backward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < CUDA_WARP_SIZE) ? next_power_of_two : CUDA_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the batch
  int local_idx = threadIdx.x % WARP_SIZE;

  // the first element to process by the current thread
  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
  // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
  // the nested loops.
  // This should have no impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        grad_reg[i][it] = grad[i * element_count + it * WARP_SIZE];
        output_reg[i][it] = output[i * element_count + it * WARP_SIZE];
      } else {
        grad_reg[i][it] = acc_t(0);
        output_reg[i][it] = acc_t(0);
      }
    }
  }

  acc_t sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        //if (is_log_softmax) {
        //  gradInput[i * element_count + it * WARP_SIZE] = (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
        //} else {
          gradInput[i * element_count + it * WARP_SIZE] = (grad_reg[i][it] - output_reg[i][it] * sum[i]);
        //}
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_forward(output_t* dst, const input_t* src, int softmax_elements, int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < CUDA_WARP_SIZE) ? next_power_of_two : CUDA_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        softmax_warp_forward<input_t, output_t, acc_t, 0, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 1:  // 2
        softmax_warp_forward<input_t, output_t, acc_t, 1, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 2:  // 4
        softmax_warp_forward<input_t, output_t, acc_t, 2, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 3:  // 8
        softmax_warp_forward<input_t, output_t, acc_t, 3, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 4:  // 16
        softmax_warp_forward<input_t, output_t, acc_t, 4, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 5:  // 32
        softmax_warp_forward<input_t, output_t, acc_t, 5, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 6:  // 64
        softmax_warp_forward<input_t, output_t, acc_t, 6, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 7:  // 128
        softmax_warp_forward<input_t, output_t, acc_t, 7, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 8:  // 256
        softmax_warp_forward<input_t, output_t, acc_t, 8, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 9:  // 512
        softmax_warp_forward<input_t, output_t, acc_t, 9, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 10:  // 1024
        softmax_warp_forward<input_t, output_t, acc_t, 10, is_log_softmax>
            <<<blocks, threads, 0>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      default:
        break;
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_backward(output_t* grad_input, const input_t* grad, const input_t* output, int softmax_elements, int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
    int warp_size = (next_power_of_two < CUDA_WARP_SIZE) ? next_power_of_two : CUDA_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        softmax_warp_backward<input_t, output_t, acc_t, 0, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 1:  // 2
        softmax_warp_backward<input_t, output_t, acc_t, 1, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 2:  // 4
        softmax_warp_backward<input_t, output_t, acc_t, 2, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 3:  // 8
        softmax_warp_backward<input_t, output_t, acc_t, 3, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 4:  // 16
        softmax_warp_backward<input_t, output_t, acc_t, 4, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 5:  // 32
        softmax_warp_backward<input_t, output_t, acc_t, 5, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 6:  // 64
        softmax_warp_backward<input_t, output_t, acc_t, 6, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 7:  // 128
        softmax_warp_backward<input_t, output_t, acc_t, 7, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 8:  // 256
        softmax_warp_backward<input_t, output_t, acc_t, 8, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 9:  // 512
        softmax_warp_backward<input_t, output_t, acc_t, 9, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 10:  // 1024
        softmax_warp_backward<input_t, output_t, acc_t, 10, is_log_softmax>
            <<<blocks, threads, 0>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      default:
        break;
    }
  }
}

#define SPECIALIZED_SOFTMAX_IMPL(input_t, output_t, acc_t) \
template void dispatch_softmax_forward<input_t, output_t, acc_t, false>(output_t * dst, const input_t* src, int softmax_elements, int softmax_elements_stride, int batch_count);
    
SPECIALIZED_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_SOFTMAX_IMPL(double, double, double)

#define SPECIALIZED_SOFTMAX_GRAD_IMPL(input_t, output_t, acc_t) \
template void dispatch_softmax_backward<input_t, output_t, acc_t, false>(input_t * grad_input, const output_t* grad, const output_t* output, int softmax_elements, int softmax_elements_stride, int batch_count);

SPECIALIZED_SOFTMAX_GRAD_IMPL(float, float, float)
SPECIALIZED_SOFTMAX_GRAD_IMPL(half, half, float)
SPECIALIZED_SOFTMAX_GRAD_IMPL(double, double, double)

}
}