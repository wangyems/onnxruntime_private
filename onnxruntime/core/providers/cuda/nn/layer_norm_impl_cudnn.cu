// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "instance_norm_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _LayerNormLinearKernel(
  const int64_t N,
  const int64_t M,
  const T* X,
  const T* scale,
  const T* bias,
  T* Y) {
  int64_t count = N*M;
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, count, NumElementsPerThread);
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < count) {
      const int j = id % M;
      Y[id] = scale[j] * Y[id] + bias[j];
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void LayerNormLinearKernel(
    const int64_t N,
    const int64_t M,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y) {
  int blocksPerGrid = static_cast<int>(CeilDiv(N*M, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  _LayerNormLinearKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(N, M, X, scale, bias, Y);
}

#define LAYERNORM_LINEAR_IMPL(T)         \
  template void LayerNormLinearKernel(  \
    const int64_t N,              \
    const int64_t M,              \
    const T* X,                   \
    const T* scale,               \
    const T* bias,                \
    T* Y);

LAYERNORM_LINEAR_IMPL(float)
LAYERNORM_LINEAR_IMPL(double)
LAYERNORM_LINEAR_IMPL(half)

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _LayerNormGradInternalKernel(
    const int64_t N,
    const int64_t M,
    const T* Y_grad,
    const T* X_data,
    const T* X_mean,
    const T* X_inv_std_var,
    const T* scale,
    T* A,
    T* B,
    T* C) {
  int64_t count = N*M;
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, count, NumElementsPerThread);
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < count) {
      const int i = id / M;
      const int j = id % M;
      T val = (X_data[id] - X_mean[i]) * X_inv_std_var[i];
      A[id] = Y_grad[id] * val;
      B[id] = Y_grad[id] * scale[j] * X_inv_std_var[i];
      C[id] = B[id] * val;
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void LayerNormGradInternalKernel(
    const int64_t N,
    const int64_t M,
    const T* Y_grad,
    const T* X_data,
    const T* X_mean,
    const T* X_inv_std_var,
    const T* scale,
    T* A,
    T* B,
    T* C) {
  int blocksPerGrid = static_cast<int>(CeilDiv(N*M, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  _LayerNormGradInternalKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(N, M, Y_grad, X_data, X_mean, X_inv_std_var, scale, A, B, C);
}

#define LAYERNORM_GRADInternal_IMPL(T)        \
  template void LayerNormGradInternalKernel(  \
      const int64_t N,                        \
      const int64_t M,                        \
      const T* Y_grad,                        \
      const T* X_data,                        \
      const T* X_mean,                        \
      const T* X_inv_std_var,                 \
      const T* scale,                         \
      T* A,                                   \
      T* B,                                   \
      T* C);

LAYERNORM_GRADInternal_IMPL(float)
LAYERNORM_GRADInternal_IMPL(double)
LAYERNORM_GRADInternal_IMPL(half)

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _LayerNormGradXKernel(
                const int64_t N,
                const int64_t M,
                const T* X_data,
                const T* X_mean,
                const T* B,
                const T* mean_B,
                const T* mean_C,
                const T* X_inv_std_var,
                T* X_grad) {
  int64_t count = N*M;
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, count, NumElementsPerThread);
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < count) {
      const int i = id / M;
      X_grad[id] = B[id] - mean_B[i] - (X_data[id] - X_mean[i]) * X_inv_std_var[i] * mean_C[i];
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void LayerNormGradXKernel(
    const int64_t N,
    const int64_t M,
    const T* X_data,
    const T* X_mean,
    const T* B,
    const T* mean_B,
    const T* mean_C,
    const T* X_inv_std_var,
    T* X_grad) {
  int blocksPerGrid = static_cast<int>(CeilDiv(N*M, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  _LayerNormGradXKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(N, M, X_data, X_mean, B, mean_B, mean_C, X_inv_std_var, X_grad);
}

#define LAYERNORM_GRADX_IMPL(T)         \
  template void LayerNormGradXKernel(   \
      const int64_t N,                  \
      const int64_t M,                  \
      const T* X_data,                  \
      const T* X_mean,                  \
      const T* B,                       \
      const T* mean_B,                  \
      const T* mean_C,                  \
      const T* X_inv_std_var,           \
      T* X_grad);

LAYERNORM_GRADX_IMPL(float)
LAYERNORM_GRADX_IMPL(double)
LAYERNORM_GRADX_IMPL(half)
}  // namespace cuda
}  // namespace onnxruntime
