// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/nn/dropout_impl.h"
#include <curand_kernel.h>
#include <algorithm>

#include "thrust/device_ptr.h"
#include "thrust/fill.h"

namespace onnxruntime {
namespace cuda {

constexpr int UNROLL = 4;

template <typename T>
__global__ void DropoutKernel(
    const int64_t N,
    const float ratio,
    const std::pair<uint64_t, uint64_t> seeds,
    const T* X_data,
    T* Y_data,
    bool* mask_data) {
  const float p = 1.0f - ratio;
  const T scale = T(1.0f / p);

  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG step_size = gridDim.x * blockDim.x * UNROLL;
  CUDA_LONG rounded_size = ((N - 1) / step_size + 1) * step_size;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  // We ensure every thread generates the same number of random numbers (by rounding
  // up the size) and at the same timestep (by syncing threads).
  // From CUDA curand documentation:
  //   The Philox_4x32_10 algorithm is closely tied to the thread and block count.
  //   Each thread computes 4 random numbers in the same time thus the most efficient
  //   use of Philox_4x32_10 is to generate a multiple of 4 times number of threads.
  for (CUDA_LONG id = idx; id < rounded_size; id += step_size) {
    float4 rand = curand_uniform4(&state);

    for (CUDA_LONG i = 0; i < UNROLL; i++) {
      CUDA_LONG li = id + gridDim.x * blockDim.x * i;
      if (li < N) {
        mask_data[li] = (&rand.x)[i] < p;
        Y_data[li] = X_data[li] * T(mask_data[li]) * scale;
      }
    }

    __syncthreads();
  }
}

template <typename T>
void DropoutKernelImpl(
    const cudaDeviceProp& prop,
    const int64_t N,
    const float ratio,
    DropoutGenerator& generator,
    const T* X_data,
    T* Y_data,
    bool* mask_data) {
  if (ratio == 0.0f) {
    if (Y_data != X_data) {
      CUDA_CALL_THROW(cudaMemcpyAsync(Y_data, X_data, N * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    thrust::fill_n(thrust::device_pointer_cast(mask_data), N, true);
  } else {
    const int block_size = 256;
    const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
    const int grid_size = std::min(prop.multiProcessorCount * blocks_per_sm, static_cast<int>(CeilDiv(N, block_size)));

    // Compute the number of random numbers generated by each thread, and increment philox generator offset by that amount.
    const uint64_t counter_offset = static_cast<uint64_t>(((N - 1) / (block_size * grid_size * UNROLL) + 1) * UNROLL);
    auto seeds = generator.GetPhiloxSeeds(counter_offset);

    DropoutKernel<T><<<grid_size, block_size, 0>>>(N, ratio, seeds, X_data, Y_data, mask_data);
  }
}

#define SPECIALIZED_DROPOUT_IMPL(T) \
  template void DropoutKernelImpl(  \
      const cudaDeviceProp& prop,   \
      const int64_t N,              \
      const float ratio,            \
      DropoutGenerator& generator,  \
      const T* X_data,              \
      T* Y_data,                    \
      bool* mask_data);

SPECIALIZED_DROPOUT_IMPL(float)
SPECIALIZED_DROPOUT_IMPL(double)
SPECIALIZED_DROPOUT_IMPL(half)

template <typename T>
__global__ void DropoutGradientKernel(
    const int64_t N,
    const T* dY_data,
    const bool* mask_data,
    const T scale,
    T* dX_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  dX_data[id] = dY_data[id] * T(mask_data[id]) * scale;
}

template <typename T>
void DropoutGradientKernelImpl(
    const int64_t N,
    const T* dY_data,
    const bool* mask_data,
    const float ratio,
    T* dX_data) {
  if (ratio == 0.0f) {
    if (dY_data != dX_data) {
      CUDA_CALL_THROW(cudaMemcpyAsync(dX_data, dY_data, N * sizeof(T), cudaMemcpyDeviceToDevice));
    }
  } else {
    const float scale = 1.f / (1.f - ratio);
    const int blocksPerGrid = (N + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    DropoutGradientKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(N, dY_data, mask_data, T(scale), dX_data);
  }
}

#define SPECIALIZED_DROPOUT_GRAD_IMPL(T)   \
  template void DropoutGradientKernelImpl( \
      const int64_t N,                     \
      const T* dY_data,                    \
      const bool* mask_data,               \
      const float scale,                   \
      T* dX_data);

SPECIALIZED_DROPOUT_GRAD_IMPL(float)
SPECIALIZED_DROPOUT_GRAD_IMPL(double)
SPECIALIZED_DROPOUT_GRAD_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
