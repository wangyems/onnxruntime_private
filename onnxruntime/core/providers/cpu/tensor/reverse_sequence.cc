// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/reverse_sequence.h"

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "gsl/gsl"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "core/framework/data_types_internal.h"
#include "core/framework/utils.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ReverseSequence, Input, 0,
    ORT_OP_KERNEL_TYPE_CTRL_ALL_TENSOR_DATA_TYPES);
}

using ReverseSequenceDataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ReverseSequence, Input, 0);
using EnabledReverseSequenceDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ReverseSequence, Input, 0);

ONNX_OPERATOR_KERNEL_EX(ReverseSequence,
                        kOnnxDomain,
                        10,
                        kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T",
                                            BuildKernelDefConstraintsFromTypeList<ReverseSequenceDataTypes>(),
                                            BuildKernelDefConstraintsFromTypeList<EnabledReverseSequenceDataTypes>()),
                        ReverseSequenceOp);

namespace {
template <typename T>
struct ReverseSequenceImplDispatchTarget {
  void operator()(const Tensor& X, Tensor& Y, gsl::span<const int64_t> sequence_lengths,
                  int64_t max_seq_len, int64_t batch_size, int64_t input_size, bool time_major) const;
};
}  // namespace

Status ReverseSequenceOp::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& dims = X.Shape();

  const auto batch_size = time_major_ ? dims[1] : dims[0];
  const auto max_seq_len = time_major_ ? dims[0] : dims[1];
  const auto input_size = dims.SizeFromDimension(2);

  const auto& seq_lengths = *context->Input<Tensor>(1);
  const auto& seq_len_shape = seq_lengths.Shape();

  if (seq_len_shape.NumDimensions() != 1 || seq_len_shape[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_lens shape must be {batch_size}. Got:",
                           seq_len_shape, ". batch_size=", batch_size);
  }

  auto& Y = *context->Output(0, dims);

  utils::MLTypeCallDispatcherFromTypeList<EnabledReverseSequenceDataTypes> dispatcher{X.GetElementType()};
  dispatcher.Invoke<ReverseSequenceImplDispatchTarget>(
      X, Y, seq_lengths.DataAsSpan<int64_t>(), max_seq_len, batch_size, input_size, time_major_);

  return Status::OK();
}

namespace {
int64_t TimeMajorInputOffset(const int64_t max_seq_len,
                             const int64_t batch_size,
                             const int64_t input_size,
                             const int64_t batch_num,
                             const int64_t seq_num) {
  ORT_UNUSED_PARAMETER(max_seq_len);
  return seq_num * batch_size * input_size + batch_num * input_size;
}

int64_t BatchMajorInputOffset(const int64_t max_seq_len,
                              const int64_t batch_size,
                              const int64_t input_size,
                              const int64_t batch_num,
                              const int64_t seq_num) {
  ORT_UNUSED_PARAMETER(batch_size);
  return batch_num * max_seq_len * input_size + seq_num * input_size;
}

int64_t TimeMajorOutputOffset(const int64_t max_seq_len,
                              const int64_t batch_size,
                              const int64_t input_size,
                              const int64_t batch_num,
                              const int64_t seq_num,
                              const int64_t seq_len) {
  ORT_UNUSED_PARAMETER(max_seq_len);
  return (seq_len - seq_num - 1) * batch_size * input_size + batch_num * input_size;
}

int64_t BatchMajorOutputOffset(const int64_t max_seq_len,
                               const int64_t batch_size,
                               const int64_t input_size,
                               const int64_t batch_num,
                               const int64_t seq_num,
                               const int64_t seq_len) {
  ORT_UNUSED_PARAMETER(batch_size);
  return batch_num * max_seq_len * input_size + (seq_len - seq_num - 1) * input_size;
}

template <typename T>
void ReverseSequenceImplDispatchTarget<T>::operator()(
    const Tensor& X,
    Tensor& Y,
    gsl::span<const int64_t> sequence_lengths,
    const int64_t max_seq_len,
    const int64_t batch_size,
    const int64_t input_size,
    bool time_major) const {
  gsl::span<const T> inputs = X.DataAsSpan<T>();
  gsl::span<T> inputs_reverse = Y.MutableDataAsSpan<T>();

  auto input_offset = time_major ? TimeMajorInputOffset : BatchMajorInputOffset;

  auto reversed_output_offset = time_major ? TimeMajorOutputOffset : BatchMajorOutputOffset;

  for (int i = 0; i < batch_size; i++) {
    int64_t seq_len = sequence_lengths[i];

    if (seq_len == 0)
      continue;

    for (int64_t j = 0; j < seq_len; j++) {
      gsl::span<const T> src = inputs.subspan(input_offset(max_seq_len, batch_size, input_size, i, j), input_size);
      gsl::span<T> dest = inputs_reverse.subspan(
          reversed_output_offset(max_seq_len, batch_size, input_size, i, j, seq_len), input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }

    for (int64_t j = seq_len; j < max_seq_len; j++) {
      const auto offset = input_offset(max_seq_len, batch_size, input_size, i, j);
      gsl::span<const T> src = inputs.subspan(offset, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(offset, input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }
  }
}
}  // namespace

}  // namespace onnxruntime
