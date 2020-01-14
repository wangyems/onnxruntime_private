// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/collective/nccl_kernels.h"

namespace onnxruntime {
namespace cuda {

NcclAllReduce::NcclAllReduce(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclAllReduce::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm();

  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    auto onnx_type = input_tensor->DataType();
    const void* input_data = input_tensor->DataRaw();
    size_t input_count = input_tensor->Shape().Size();

    Tensor* output_tensor = context->Output(i, input_tensor->Shape());
    void* output_data = output_tensor->MutableDataRaw();

    ncclDataType_t dtype = GetNcclDataType(onnx_type);
    NCCL_RETURN_IF_ERROR(ncclAllReduce(input_data, output_data, input_count, dtype, ncclSum, comm, stream));
  }

  return Status::OK();
}

NcclAllGather::NcclAllGather(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclAllGather::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm();
  const int rank = nccl_->Rank();
  const int size = nccl_->Size();

  ORT_ENFORCE(context->InputCount() > 0);
  auto onnx_type = context->Input<Tensor>(0)->DataType();
  const size_t element_size = onnx_type->Size();
  ncclDataType_t dtype = GetNcclDataType(onnx_type);

  // Count total number of elements to AllGather.
  int64_t total_count = 0;
  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    total_count += input_tensor->Shape().Size();
  }

  // AllGather requires every rank to receive the same amount of data, and significantly
  // slows down significantly if the data is not aligned.  Nvidia recommends 32-byte alignment,
  // so pad to multiple of 32 and world size.
  // Note: the alignment here needs to be kept in-sync with the alignment in optimizer_graph_builder.cc
  const int64_t alignment = size * 32;
  const int64_t padded_count = total_count + alignment - (total_count % alignment);
  const int64_t padded_size = padded_count * element_size;
  auto fusion_buffer = GetScratchBuffer<void>(padded_size);
  void* fusion_data = fusion_buffer.get();

  // Calculate the range of inputs this rank will send.
  ORT_ENFORCE(padded_count % size == 0);
  const int64_t rank_count = padded_count / size;
  const int64_t rank_bytes = rank_count * element_size;
  const int64_t rank_start = rank * rank_bytes;
  const int64_t rank_end = rank_start + rank_bytes;

  // Copy this rank's inputs to fusion buffer.
  int64_t offset = 0;
  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    const int64_t tensor_bytes = input_tensor->SizeInBytes();

    // Only copy inputs this rank needs to send.
    if (rank_start <= offset && offset < rank_end) {
      ORT_ENFORCE(offset + tensor_bytes <= rank_end, "A single rank must be responsible for the entire tensor.");
      void* fusion_data_at_offset = (int8_t*)fusion_data + offset;
      const void* input_data = input_tensor->DataRaw();
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(fusion_data_at_offset, input_data, tensor_bytes, cudaMemcpyDeviceToDevice));
    }

    offset += tensor_bytes;
  }

  // AllGather.
  const void* fusion_data_rank_offset = (const int8_t*)fusion_data + rank_start;
  NCCL_RETURN_IF_ERROR(ncclAllGather(fusion_data_rank_offset, fusion_data, rank_count, dtype, comm, stream));

  // Copy AllGather results to outputs.
  offset = 0;
  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    const TensorShape& input_shape = input_tensor->Shape();
    const int64_t tensor_bytes = input_tensor->SizeInBytes();
    Tensor* output_tensor = context->Output(i, input_shape);

    // TODO: temporary hack until View is improved (it doesn't work with Alias)
    output_tensor->SetByteOffset(input_tensor->ByteOffset());

    // Only copy outputs that came from other ranks.
    if (offset < rank_start || offset >= rank_end) {
      void* output_data = output_tensor->MutableDataRaw();
      const void* fusion_data_at_offset = (const int8_t*)fusion_data + offset;
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, fusion_data_at_offset, tensor_bytes, cudaMemcpyDeviceToDevice));
    } else {
      const void* input_data = input_tensor->DataRaw();
      void* output_data = output_tensor->MutableDataRaw();
      if (input_data != output_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, tensor_bytes, cudaMemcpyDeviceToDevice));
      }
    }

    offset += tensor_bytes;
  }

  return Status::OK();
}

NcclReduceScatter::NcclReduceScatter(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclReduceScatter::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm();
  const int rank = nccl_->Rank();
  const int size = nccl_->Size();

  ORT_ENFORCE(context->InputCount() > 0);
  auto onnx_type = context->Input<Tensor>(0)->DataType();
  const size_t element_size = onnx_type->Size();
  ncclDataType_t dtype = GetNcclDataType(onnx_type);

  // Count total number of elements to ReduceScatter.
  int64_t total_count = 0;
  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    total_count += input_tensor->Shape().Size();
  }

  // ReduceScatter requires every rank to receive the same amount of data, and significantly
  // slows down significantly if the data is not aligned.  Nvidia recommends 32-byte alignment,
  // so pad to multiple of 32 and world size.
  // Note: the alignment here needs to be kept in-sync with the alignment in optimizer_graph_builder.cc
  const int64_t alignment = size * 32;
  const int64_t padded_count = total_count + alignment - (total_count % alignment);
  const int64_t padded_size = padded_count * element_size;
  auto fusion_buffer = GetScratchBuffer<void>(padded_size);
  void* fusion_data = fusion_buffer.get();

  // Calculate the range of outputs this rank will receive.
  ORT_ENFORCE(padded_count % size == 0);
  const int64_t rank_count = padded_count / size;
  const int64_t rank_bytes = rank_count * element_size;
  const int64_t rank_start = rank * rank_bytes;
  const int64_t rank_end = rank_start + rank_bytes;

  // Copy inputs to fusion buffer.
  int64_t offset = 0;
  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    const int64_t tensor_bytes = input_tensor->SizeInBytes();

    void* fusion_data_at_offset = (int8_t*)fusion_data + offset;
    const void* input_data = input_tensor->DataRaw();
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(fusion_data_at_offset, input_data, tensor_bytes, cudaMemcpyDeviceToDevice));

    offset += tensor_bytes;
  }

  // ReduceScatter.
  void* fusion_data_rank_offset = (int8_t*)fusion_data + rank_start;
  NCCL_RETURN_IF_ERROR(ncclReduceScatter(fusion_data, fusion_data_rank_offset, rank_count, dtype, ncclSum, comm, stream));

  // Copy this rank's ReduceScatter results to outputs.
  offset = 0;
  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    const TensorShape& input_shape = input_tensor->Shape();
    const int64_t tensor_bytes = input_tensor->SizeInBytes();
    Tensor* output_tensor = context->Output(i, input_shape);

    // TODO: temporary hack until View is improved (it doesn't work with Alias)
    output_tensor->SetByteOffset(input_tensor->ByteOffset());

    // Only copy outputs this rank should receive.
    if (rank_start <= offset && offset < rank_end) {
      ORT_ENFORCE(offset + tensor_bytes <= rank_end, "A single rank must be responsible for the entire tensor.");
      void* output_data = output_tensor->MutableDataRaw();
      const void* fusion_data_at_offset = (const int8_t*)fusion_data + offset;
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, fusion_data_at_offset, tensor_bytes, cudaMemcpyDeviceToDevice));
    } else {
      const void* input_data = input_tensor->DataRaw();
      void* output_data = output_tensor->MutableDataRaw();
      if (input_data != output_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, tensor_bytes, cudaMemcpyDeviceToDevice));
      }
    }

    offset += tensor_bytes;
  }

  return Status::OK();
}

static std::vector<std::pair<int, int>> AliasRange(int start, int end) {
  std::vector<std::pair<int, int>> aliases;
  for (int i = start; i < end; i++) {
    aliases.push_back(std::pair<int, int>(i, i));
  }
  return aliases;
}

ONNX_OPERATOR_KERNEL_EX(
    NcclAllReduce,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(AliasRange(0, 1024))
        .TypeConstraint("T", DataTypeImpl::AllFloatingPointTensorTypes()),
    NcclAllReduce);

ONNX_OPERATOR_KERNEL_EX(
    NcclAllGather,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(AliasRange(0, 1024))
        .TypeConstraint("T", DataTypeImpl::AllFloatingPointTensorTypes()),
    NcclAllGather);

ONNX_OPERATOR_KERNEL_EX(
    NcclReduceScatter,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(AliasRange(0, 1024))
        .TypeConstraint("T", DataTypeImpl::AllFloatingPointTensorTypes()),
    NcclReduceScatter);

}  // namespace cuda
}  // namespace onnxruntime
