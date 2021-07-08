//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/copy.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

std::vector<int64_t> StridesForTensor(const Tensor& dst) {
  std::vector<int64_t> strides(dst.Shape().NumDimensions());
  for (std::size_t i = 0; i < dst.Shape().NumDimensions(); i++) {
    strides[i] = 1;
    for (std::size_t j = i + 1; j < dst.Shape().NumDimensions(); j++) {
      strides[i] *= dst.Shape()[j];
    }
  }
  return strides;
}

Status DispatchStridedCopy(concurrency::ThreadPool* thread_pool,
                           Tensor& dst,
                           std::ptrdiff_t dst_offset,
                           const std::vector<int64_t> dst_strides,
                           const TensorShape& copy_shape,
                           const Tensor& src,
                           const std::vector<int64_t> src_strides) {
  ORT_RETURN_IF_NOT(dst.DataType() == src.DataType(), "src and dst types must match");
  ORT_RETURN_IF_NOT(dst_strides.size() == src_strides.size() && src_strides.size() == copy_shape.NumDimensions(), "src and dst must have same shape");

  // Manual dispatching: DispatchOnTensorType doesn't work here because we need to pass the type to the MutableData call
#define CALL_FOR_TYPE(T)                                                                                               \
  StridedCopy<T>(thread_pool, dst.MutableData<T>() + dst_offset, dst_strides, copy_shape, src.Data<T>(), src_strides); \
  break

  auto tensor_type = dst.DataType()->AsPrimitiveDataType()->GetDataType();
  switch (tensor_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      CALL_FOR_TYPE(float);
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      CALL_FOR_TYPE(bool);
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      CALL_FOR_TYPE(double);
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      CALL_FOR_TYPE(std::string);
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      CALL_FOR_TYPE(int8_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      CALL_FOR_TYPE(uint8_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      CALL_FOR_TYPE(int16_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      CALL_FOR_TYPE(uint16_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      CALL_FOR_TYPE(int32_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      CALL_FOR_TYPE(uint32_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      CALL_FOR_TYPE(int64_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      CALL_FOR_TYPE(uint64_t);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      CALL_FOR_TYPE(MLFloat16);
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      CALL_FOR_TYPE(BFloat16);
    default:
      ORT_ENFORCE(false, "Unknown tensor type of ", tensor_type);
  }
  return Status::OK();
}

template <typename T>
void StridedCopy(concurrency::ThreadPool* thread_pool,
                 T* dst,
                 const std::vector<int64_t>& dst_strides,
                 const TensorShape& copy_shape,
                 const T* src,
                 const std::vector<int64_t>& src_strides) {
  const auto* src_raw = reinterpret_cast<const uint8_t*>(src);
  auto* dst_raw = reinterpret_cast<uint8_t*>(dst);
  const size_t dims = copy_shape.NumDimensions();
  // We will iterate over the output dimensions
  int64_t num_iterations = 1;
  for (size_t dim = 0; dim < dims; dim++) {
    num_iterations *= copy_shape[dim];
  }

  if (num_iterations == 1) {
    // scalar edge case
    dst[0] = src[0];
    return;
  }

  // TODOs for when we have strided tensors:
  // - Reorder dimensions so that we iterate along the smallest strides first
  // - remove single size dimensions
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, num_iterations,
      {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), 1.0F},
      [copy_shape, dst_strides, dst, dst_raw, src, src_raw, src_strides, dims](std::ptrdiff_t first, std::ptrdiff_t last) {
        // Compute the initial n-dimensional index and addresses
        auto last_dim_size = copy_shape[dims - 1];
        auto last_dst_stride = dst_strides[dims - 1];
        auto last_src_stride = src_strides[dims - 1];
        bool is_string = std::is_same<T, std::string>::value;
        std::vector<int64_t> current_nd_idx(dims);
        {
          int64_t current_index = first;
          for (size_t dim = dims; dim > 0; dim--) {
            auto shape_val = copy_shape[dim - 1];
            // Iterate from dims to 1 so we don't roll over to positive on the bounds check
            current_nd_idx[dim - 1] = current_index % shape_val;
            current_index /= shape_val;
          }
        }

        for (std::ptrdiff_t outer_i = first; outer_i < last;) {
          // Compute the src and dst addresses
          std::ptrdiff_t dst_idx = 0;
          std::ptrdiff_t src_idx = 0;
          for (size_t dim = 0; dim < dims; dim++) {
            dst_idx += current_nd_idx[dim] * dst_strides[dim];
            src_idx += current_nd_idx[dim] * src_strides[dim];
          }

          std::ptrdiff_t inner_end = std::min(last, outer_i + last_dim_size - current_nd_idx[dims - 1]);
          auto iter_size = inner_end - outer_i;
          if (!is_string && last_dst_stride == 1 && last_src_stride == 1) {
            memcpy(dst_raw + dst_idx * sizeof(T), src_raw + src_idx * sizeof(T), iter_size * sizeof(T));
          } else {
            for (std::ptrdiff_t i = outer_i; i < inner_end; i++) {
              dst[dst_idx] = src[src_idx];
              dst_idx += last_dst_stride;
              src_idx += last_src_stride;
            }
          }
          current_nd_idx[dims - 1] += iter_size;

          outer_i = inner_end;

          // update the current_nd_idx if needed
          size_t dim = dims - 1;
          while (dim > 0 && current_nd_idx[dim] >= copy_shape[dim]) {
            current_nd_idx[dim] = 0;
            dim--;
            current_nd_idx[dim]++;
          }
        }
      });
}
}  // namespace onnxruntime
