// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scatter
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

class Scatter final : public OpKernel {
 public:
  Scatter(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");
  }
  ~Scatter() = default;
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

ONNX_CPU_OPERATOR_KERNEL(
    Scatter,
    9,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Scatter);

template <class Tin>
Status CopyScatterData(const Tensor* data_input, const Tensor* indices_input, const Tensor* updates_input,
                       const int64_t axis, Tensor* data_output) {
  const TensorShape& input_data_shape = data_input->Shape();
  const Tin* indices_data = indices_input->template Data<Tin>();
  const auto num_indices = indices_input->Shape().Size();
  for (int64_t i = 0; i < num_indices; ++i) {
    Tin idx = indices_data[i];
    if (idx < 0 || idx >= input_data_shape[axis]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices element out of data bounds, idx=", idx,
                             " data_dim=", input_data_shape[axis]);
    }
  }

  const auto input_elements = input_data_shape.Size();
  const auto element_bytes = data_input->DataType()->Size();
  const auto total_input_bytes = data_input->Size();

  const uint8_t* src_base = reinterpret_cast<const uint8_t*>(data_input->DataRaw());
  uint8_t* dst_base = reinterpret_cast<uint8_t*>(data_output->MutableDataRaw());
  const bool is_string_type = data_input->DataType() == DataTypeImpl::GetType<std::string>();

  // We allow runtime to re-use input for output. If input/output Tensor* are the same
  // we do not copy
  if (src_base != dst_base) {
    if (is_string_type) {
      const std::string* str_begin = data_input->template Data<std::string>();
      const std::string* str_end = str_begin + input_elements;
      std::string* dst = data_output->template MutableData<std::string>();
      std::copy(str_begin, str_end, dst);
    } else {
      memcpy(dst_base, src_base, total_input_bytes);
    }
  }

  // Now poke updates

  const auto& upd_shape = updates_input->Shape();
  const auto num_dims = input_data_shape.NumDimensions();
  assert(num_dims > 0);

  // Allocate and zero out counts. We are expecting that input and
  // update ranks are the same so the vectors of equal length
  // This is validated
  std::vector<int64_t> dim_counters(num_dims);

  // Create weights for each input dimension so we can map
  // update dimensions to input/output dimensions
  std::vector<int64_t> out_weights(num_dims);

  out_weights.back() = 1;
  if (out_weights.size() > 1) {
    for (int64_t i = int64_t(num_dims - 2); i >= 0; --i) {
      out_weights[i] = input_data_shape[i] * out_weights[i + 1];
    }
  }

  const uint8_t* update_data = reinterpret_cast<const uint8_t*>(updates_input->DataRaw());
  for (int64_t index = 0; index < num_indices;) {
    const Tin axis_idx = indices_data[index];

    // Compute the offset
    size_t dst_offset = 0;
    for (size_t i = 0; i < num_dims; ++i) {
      if (i == size_t(axis)) {
        dst_offset += axis_idx * out_weights[i];
      } else {
        dst_offset += dim_counters[i] * out_weights[i];
      }
    }

    const size_t dst_offset_bytes = dst_offset * element_bytes;
    assert(dst_offset_bytes < total_input_bytes);
    if (is_string_type) {
      reinterpret_cast<std::string*>(dst_base)[dst_offset] =
          reinterpret_cast<const std::string*>(update_data)[index];
    } else {
      // Copy the element
      // TODO: Replace memcpy
      auto src_offset_bytes = index * element_bytes;
      memcpy(dst_base + dst_offset_bytes, update_data + src_offset_bytes, element_bytes);
    }

    if (++index == num_indices) {
      break;
    }
    // Increment counters
    for (int64_t i = int64_t(num_dims - 1); i >= 0; --i) {
      auto v = ++dim_counters[i];
      assert(v <= upd_shape[i]);
      if (v < upd_shape[i]) {
        // No carry, done
        break;
      }
      // No carry for the most significant dim
      assert(i > 0);
      dim_counters[i] = 0;
    }
  }
  return Status::OK();
}

Status Scatter::Compute(OpKernelContext* context) const {
  const auto* data_input = context->Input<Tensor>(0);
  const auto& input_data_shape = data_input->Shape();
  const auto axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  const auto* indices_input = context->Input<Tensor>(1);
  const auto* updates_input = context->Input<Tensor>(2);

  if (data_input->DataType() != updates_input->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data type is different from updates type");
  }

  auto& indices_dims = indices_input->Shape().GetDims();
  auto& updates_dims = updates_input->Shape().GetDims();
  if (indices_dims.size() != updates_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Indices and updates must have the same rank");
  }

  for (size_t i = 0; i < indices_dims.size(); ++i) {
    if (indices_dims[i] != updates_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices vs updates dimensions differs at position=", i,
                             " ", indices_dims[i], " vs ", updates_dims[i]);
    }
  }

  // According to the spec the rank of ind/upd shall be the same as input(data)
  // and we also want to make sure that the dimensions of the of the ind/upd do not
  // exceed that of the input
  auto& input_dims = input_data_shape.GetDims();
  if (input_dims.size() != indices_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices/updates must have the same rank as input ind rank=",
                           indices_dims.size(), " vs input rank=", input_dims.size());
  }

  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i] < indices_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices dim=", indices_dims[i], " at pos=", i,
                             " is greater than input dim=", input_dims[i]);
    }
  }

  auto* data_output = context->Output(0, input_data_shape);

  MLDataType Tind_type = indices_input->DataType();
  if (Tind_type == DataTypeImpl::GetType<int32_t>()) {
    return CopyScatterData<int32_t>(data_input, indices_input, updates_input, axis, data_output);
  } else if (Tind_type == DataTypeImpl::GetType<int64_t>()) {
    return CopyScatterData<int64_t>(data_input, indices_input, updates_input, axis, data_output);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expecting indices to be either int32_t or int64_t");
}

}  // namespace onnxruntime
