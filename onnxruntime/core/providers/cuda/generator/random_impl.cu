// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/generator/random_impl.h"

#include <curand_kernel.h>
#include <algorithm>
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

constexpr int UNROLL = 4;

struct DistFunc_RandomNormal {
  __device__ __inline__ float4 operator()(curandStatePhilox4_32_10_t* state) const { return curand_normal4(state); }
};

struct DistFunc_RandomUniform {
  __device__ __inline__ float4 operator()(curandStatePhilox4_32_10_t* state) const { return curand_uniform4(state); }
};

struct TransformFunc_RandomNormal {
  __device__ __inline__ float operator()(const float value, const float scale, const float mean) const {
    return value * scale + mean;
  }
};

struct TransformFunc_RandomUniform {
  __device__ __inline__ float operator()(const float value, const float range, const float from) const {
    // reverse the bounds of curand4 from (0, 1] to [0, 1).
    // ref: https://github.com/pytorch/pytorch/blob/e795315c638228d4170f3797356c09a70b2ed4cd/aten/src/ATen/native/cuda/DistributionTemplates.h#L464
    float reverse_bound_value = value == 1.0f ? 0.0f : value;
    return reverse_bound_value * range + from;
  }
};

template <typename T, typename DistFuncT, typename TransformFuncT>
__global__ void RandomKernel(const int64_t N, const std::pair<uint64_t, uint64_t> seeds, const DistFuncT& dist_func,
                             const TransformFuncT& transform_func, const float alpha, const float beta, T* Y_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG step_size = gridDim.x * blockDim.x * UNROLL;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);
  float4 rand;

  // We ensure every thread generates the same number of random numbers (by rounding
  // up the size) and at the same timestep (by syncing threads).
  // From CUDA curand documentation:
  //   The Philox_4x32_10 algorithm is closely tied to the thread and block count.
  //   Each thread computes 4 random numbers in the same time thus the most efficient
  //   use of Philox_4x32_10 is to generate a multiple of 4 times number of threads.
  for (CUDA_LONG id = idx * UNROLL; id < N; id += step_size) {
    rand = dist_func(&state);

// actual computation
#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      CUDA_LONG li = id + i;
      if (li < N) {
        Y_data[li] = static_cast<T>(transform_func((&rand.x)[i], alpha, beta));
      }
    }

    __syncthreads();
  }
}

template <typename T, typename DistFuncT, typename TransformFuncT>
__global__ void RandomVectorizedKernel(const int64_t N, const std::pair<uint64_t, uint64_t> seeds,
                                       const DistFuncT& dist_func, const TransformFuncT& transform_func,
                                       const float alpha, const float beta, T* Y_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG step_size = gridDim.x * blockDim.x * UNROLL;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);
  float4 rand;

  // Using vectorized data load/store approach when N % 4 == 0 since this is typical case for input shape size.
  using LoadT = aligned_vector<T, UNROLL>;
  for (CUDA_LONG id = idx * UNROLL; id < N; id += step_size) {
    rand = dist_func(&state);
    T r[UNROLL];

// actual computation
#pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      r[ii] = static_cast<T>(transform_func((&rand.x)[ii], alpha, beta));
    }

    // Vectorized writes for Y_data
    *(reinterpret_cast<LoadT*>(&Y_data[id])) = *reinterpret_cast<LoadT*>(&r[0]);

    __syncthreads();
  }
}

template <typename T, typename DistFuncT, typename TransformFuncT>
void RandomKernelImpl(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const DistFuncT& dist_func,
                      const TransformFuncT& transform_func, float alpha, float beta, PhiloxGenerator& generator,
                      T* Y_data) {
  const int block_size = 256;
  const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
  const int grid_size =
      std::min(prop.multiProcessorCount * blocks_per_sm, static_cast<int>(CeilDiv(N, block_size * UNROLL)));

  // Compute the number of random numbers generated by each thread, and increment philox generator offset by that
  // amount.
  const uint64_t counter_offset = static_cast<uint64_t>(((N - 1) / (block_size * grid_size * UNROLL) + 1) * UNROLL);
  auto seeds = generator.NextPhiloxSeeds(counter_offset);

  if (N % UNROLL != 0) {
    RandomKernel<T><<<grid_size, block_size, 0, stream>>>(N, seeds, dist_func, transform_func, alpha, beta, Y_data);
  } else {
    RandomVectorizedKernel<T>
        <<<grid_size, block_size, 0, stream>>>(N, seeds, dist_func, transform_func, alpha, beta, Y_data);
  }
}

#define RANDOM_KERNEL_IMPL(name)                                                                                  \
  template <typename T>                                                                                           \
  void name##KernelImpl(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const float alpha,      \
                        const float beta, PhiloxGenerator& generator, T* Y_data) {                                \
    RandomKernelImpl(prop, stream, N, DistFunc_##name(), TransformFunc_##name(), alpha, beta, generator, Y_data); \
  }

RANDOM_KERNEL_IMPL(RandomNormal)
RANDOM_KERNEL_IMPL(RandomUniform)

#define SPECIALIZED_RANDOM_KERNEL(name, T)                                                                            \
  template void name##KernelImpl(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const float alpha, \
                                 const float beta, PhiloxGenerator& generator, T* Y_data);

#define SPECIALIZED_RANDOM_KERNELS(T)        \
  SPECIALIZED_RANDOM_KERNEL(RandomNormal, T) \
  SPECIALIZED_RANDOM_KERNEL(RandomUniform, T)

SPECIALIZED_RANDOM_KERNELS(float)
SPECIALIZED_RANDOM_KERNELS(double)
SPECIALIZED_RANDOM_KERNELS(half)

}  // namespace cuda
}  // namespace onnxruntime
