// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "binary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace cuda {

// for now this operator classes are no different than a funciton.
// Eventually once multiple binary gradient ops are needed, we will pass
// its instance from API instead of direct function call.
template <class T>
struct OP_A_DivGrad {
  __device__ __inline__ T operator()(T dy, T b) const {
    return dy / b;
  }
};
template <class T>
struct OP_B_DivGrad {
  __device__ __inline__ T operator()(T dy, T a, T b) const {
    return -dy * a / (b * b);
  }
};

template <typename T, bool a_is_scalar, bool b_is_scalar, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradSimple(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    T* output_da_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG a_index = (a_is_scalar ? 0 : id);
      CUDA_LONG b_index = (b_is_scalar ? 0 : id);
      output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
      output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, bool a_is_scalar, bool b_is_scalar, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradSimple_A(
    const T* b_data,
    const T* dy_data,
    T* output_da_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG b_index = (b_is_scalar ? 0 : id);
      output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, bool a_is_scalar, bool b_is_scalar, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradSimple_B(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG a_index = (a_is_scalar ? 0 : id);
      CUDA_LONG b_index = (b_is_scalar ? 0 : id);
      output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradRhsPerChannelBatch1(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    T* output_da_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG a_index = id;
      CUDA_LONG b_index = fdm_H.div(id);
      output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
      output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradRhsPerChannelBatch1_A(
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    T* output_da_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG b_index = fdm_H.div(id);
      output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradRhsPerChannelBatch1_B(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG a_index = id;
      CUDA_LONG b_index = fdm_H.div(id);
      output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradRhsPerChannelBatchN(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    const fast_divmod fdm_C,
    T* output_da_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG a_index = id;
      CUDA_LONG b_index = fdm_H.div(id);
      int q, r;
      fdm_C.divmod(b_index, q, r);
      b_index = r;
      output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
      output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradRhsPerChannelBatchN_A(
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    const fast_divmod fdm_C,
    T* output_da_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG b_index = fdm_H.div(id);
      int q, r;
      fdm_C.divmod(b_index, q, r);
      b_index = r;
      output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGradRhsPerChannelBatchN_B(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    const fast_divmod fdm_C,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG a_index = id;
      CUDA_LONG b_index = fdm_H.div(id);
      int q, r;
      fdm_C.divmod(b_index, q, r);
      b_index = r;
      output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, bool a_need_compute, bool b_need_compute, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGrad(
    size_t output_rank,
    const int64_t* a_padded_strides,
    const T* a_data,
    const int64_t* b_padded_strides,
    const T* b_data,
    const T* dy_data,
    const fast_divmod* fdm_output_strides,
    T* output_da_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG a_index = (a_need_compute ? 0 : id);
      CUDA_LONG b_index = (b_need_compute ? 0 : id);
      CUDA_LONG offset = id;
      for (int dim = 0; dim < output_rank; dim++) {
        int q, r;
        fdm_output_strides[dim].divmod(offset, q, r);
        // compute index increase based on stride and broadcast
        // note that stride[i-1] == stride[i] means dim[i] is 1 (broadcasting)
        if (a_need_compute) {
          if (a_padded_strides[dim] != a_padded_strides[dim + 1])
            a_index += static_cast<int>(a_padded_strides[dim + 1]) * q;
        }

        if (b_need_compute) {
          if (b_padded_strides[dim] != b_padded_strides[dim + 1])
            b_index += static_cast<int>(b_padded_strides[dim + 1]) * q;
        }
        offset = r;
      }
      output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
      output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, bool b_need_compute, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGrad_A(
    size_t output_rank,
    const int64_t* b_padded_strides,
    const T* b_data,
    const T* dy_data,
    const fast_divmod* fdm_output_strides,
    T* output_da_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG b_index = (b_need_compute ? 0 : id);
      CUDA_LONG offset = id;
      for (int dim = 0; dim < output_rank; dim++) {
        int q, r;
        fdm_output_strides[dim].divmod(offset, q, r);
        // compute index increase based on stride and broadcast
        // note that stride[i-1] == stride[i] means dim[i] is 1 (broadcasting)
        if (b_need_compute) {
          if (b_padded_strides[dim] != b_padded_strides[dim + 1])
            b_index += static_cast<int>(b_padded_strides[dim + 1]) * q;
        }
        offset = r;
      }
      output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, bool a_need_compute, bool b_need_compute, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _DivGrad_B(
    size_t output_rank,
    const int64_t* a_padded_strides,
    const T* a_data,
    const int64_t* b_padded_strides,
    const T* b_data,
    const T* dy_data,
    const fast_divmod* fdm_output_strides,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG a_index = (a_need_compute ? 0 : id);
      CUDA_LONG b_index = (b_need_compute ? 0 : id);
      CUDA_LONG offset = id;
      for (int dim = 0; dim < output_rank; dim++) {
        int q, r;
        fdm_output_strides[dim].divmod(offset, q, r);
        // compute index increase based on stride and broadcast
        // note that stride[i-1] == stride[i] means dim[i] is 1 (broadcasting)
        if (a_need_compute) {
          if (a_padded_strides[dim] != a_padded_strides[dim + 1])
            a_index += static_cast<int>(a_padded_strides[dim + 1]) * q;
        }

        if (b_need_compute) {
          if (b_padded_strides[dim] != b_padded_strides[dim + 1])
            b_index += static_cast<int>(b_padded_strides[dim + 1]) * q;
        }
        offset = r;
      }
      output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void ImplDivGradSimple(
    SimpleBroadcast simpleBroadcast,
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    T* da_output_data,
    T* db_output_data) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  switch (simpleBroadcast) {
    case SimpleBroadcast::NoBroadcast:
      // a, b and dy has the same shape: a_is_scalar = false, b_is_scalar = false
      if (da_output_data && db_output_data)
        _DivGradSimple<T, false, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            a_data,
            b_data,
            dy_data,
            da_output_data,
            db_output_data,
            N);
      else if (da_output_data)
        _DivGradSimple_A<T, false, false, GridDim::maxThreadsPerBlock,GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            b_data,
            dy_data,
            da_output_data,
            N);
      else
        _DivGradSimple_B<T, false, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            a_data,
            b_data,
            dy_data,
            db_output_data,
            N);
      return;
    case SimpleBroadcast::LeftScalar:
      // a is a scalar, b and dy has the same shape
      if (da_output_data && db_output_data)
        _DivGradSimple<T, true, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            a_data,
            b_data,
            dy_data,
            da_output_data,
            db_output_data,
            N);
      else if (da_output_data)
        _DivGradSimple_A<T, true, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            b_data,
            dy_data,
            da_output_data,
            N);
      else
        _DivGradSimple_B<T, true, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            a_data,
            b_data,
            dy_data,
            db_output_data,
            N);
      return;
    case SimpleBroadcast::RightScalar:
      // b is a scalar, a and dy has the same shape
      if (da_output_data && db_output_data)
        _DivGradSimple<T, false, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            a_data,
            b_data,
            dy_data,
            da_output_data,
            db_output_data,
            N);
      else if (da_output_data)
        _DivGradSimple_A<T, false, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            b_data,
            dy_data,
            da_output_data,
            N);
      else
        _DivGradSimple_B<T, false, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
          <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            a_data,
            b_data,
            dy_data,
            db_output_data,
            N);
      return;
    default:
      assert(false);
  }
}

template <typename T>
void ImplDivGradRhsPerChannelBatch1(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const fast_divmod& fdm_H,
    T* da_output_data,
    T* db_output_data) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (da_output_data && db_output_data)
    _DivGradRhsPerChannelBatch1<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        a_data,
        b_data,
        dy_data,
        fdm_H,
        da_output_data,
        db_output_data,
        N);
  else if (da_output_data)
    _DivGradRhsPerChannelBatch1_A<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        b_data,
        dy_data,
        fdm_H,
        da_output_data,
        N);
  else
    _DivGradRhsPerChannelBatch1_B<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        a_data,
        b_data,
        dy_data,
        fdm_H,
        db_output_data,
        N);
}

template <typename T>
void ImplDivGradRhsPerChannelBatchN(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const fast_divmod& fdm_H,
    const fast_divmod& fdm_C,
    T* da_output_data,
    T* db_output_data) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  if (da_output_data && db_output_data)
    _DivGradRhsPerChannelBatchN<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        a_data,
        b_data,
        dy_data,
        fdm_H,
        fdm_C,
        da_output_data,
        db_output_data,
        N);
  else if (da_output_data)
    _DivGradRhsPerChannelBatchN_A<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        b_data,
        dy_data,
        fdm_H,
        fdm_C,
        da_output_data,
        N);
  else
    _DivGradRhsPerChannelBatchN_B<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        a_data,
        b_data,
        dy_data,
        fdm_H,
        fdm_C,
        db_output_data,
        N);
}

template <typename T>
void ImplDivGrad(
    size_t output_rank,
    const int64_t* a_padded_strides,
    const T* a_data,
    const int64_t* b_padded_strides,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const fast_divmod* fdm_output_strides,
    T* da_output_data,
    T* db_output_data) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (a_padded_strides && b_padded_strides) {
    if (da_output_data && db_output_data)
      _DivGrad<T, true, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          a_padded_strides,
          a_data,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          da_output_data,
          db_output_data,
          N);
    else if (da_output_data)
      _DivGrad_A<T, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          da_output_data,
          N);
    else
      _DivGrad_B<T, true, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          a_padded_strides,
          a_data,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          db_output_data,
          N);
  } else if (a_padded_strides) {
    if (da_output_data && db_output_data)
      _DivGrad<T, true, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          a_padded_strides,
          a_data,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          da_output_data,
          db_output_data,
          N);
    else if (da_output_data)
      _DivGrad_A<T, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          da_output_data,
          N);
    else
      _DivGrad_B<T, true, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          a_padded_strides,
          a_data,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          db_output_data,
          N);
  } else {
    if (da_output_data && db_output_data)
      _DivGrad<T, false, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          a_padded_strides,
          a_data,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          da_output_data,
          db_output_data,
          N);
    else if (da_output_data)
      _DivGrad_A<T, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          da_output_data,
          N);
    else
      _DivGrad_B<T, false, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank,
          a_padded_strides,
          a_data,
          b_padded_strides,
          b_data,
          dy_data,
          fdm_output_strides,
          db_output_data,
          N);
  }
}  // namespace cuda

#define SPECIALIZED_DIV_GRAD_IMPL(T)               \
  template void ImplDivGrad<T>(                    \
      size_t output_rank,                          \
      const int64_t* a_padded_strides,             \
      const T* a_data,                             \
      const int64_t* b_padded_strides,             \
      const T* b_data,                             \
      const T* dy_data,                            \
      size_t count,                                \
      const fast_divmod* fdm_output_strides,       \
      T* da_output_data,                           \
      T* db_output_data);                          \
  template void ImplDivGradRhsPerChannelBatch1<T>( \
      const T* a_data,                             \
      const T* b_data,                             \
      const T* dy_data,                            \
      size_t count,                                \
      const fast_divmod& fdm_H,                    \
      T* da_output_data,                           \
      T* db_output_data);                          \
  template void ImplDivGradRhsPerChannelBatchN<T>( \
      const T* a_data,                             \
      const T* b_data,                             \
      const T* dy_data,                            \
      size_t count,                                \
      const fast_divmod& fdm_H,                    \
      const fast_divmod& fdm_C,                    \
      T* da_output_data,                           \
      T* db_output_data);                          \
  template void ImplDivGradSimple<T>(              \
      SimpleBroadcast simpleBroadcast,             \
      const T* a_data,                             \
      const T* b_data,                             \
      const T* dy_data,                            \
      size_t count,                                \
      T* da_output_data,                           \
      T* db_output_data);

SPECIALIZED_DIV_GRAD_IMPL(half)
SPECIALIZED_DIV_GRAD_IMPL(float)
SPECIALIZED_DIV_GRAD_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
