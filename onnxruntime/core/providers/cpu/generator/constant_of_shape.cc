// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/generator/constant_of_shape.h"
#include "gsl/span"

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    ConstantOfShape,
    9,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<MLFloat16>(),
                                  DataTypeImpl::GetTensorType<float>(),
                                  DataTypeImpl::GetTensorType<double>(),
                                  DataTypeImpl::GetTensorType<int8_t>(),
                                  DataTypeImpl::GetTensorType<int16_t>(),
                                  DataTypeImpl::GetTensorType<int32_t>(),
                                  DataTypeImpl::GetTensorType<int64_t>(),
                                  DataTypeImpl::GetTensorType<uint8_t>(),
                                  DataTypeImpl::GetTensorType<uint16_t>(),
                                  DataTypeImpl::GetTensorType<uint32_t>(),
                                  DataTypeImpl::GetTensorType<uint64_t>(),
                                  DataTypeImpl::GetTensorType<bool>()}),
    ConstantOfShape);

#define FETCH_VALUE_DATA(field, c_type)                                                                   \
  {                                                                                                       \
    c_type t;                                                                                             \
    auto unpack_status = UnpackTensor(t_proto, raw_data, raw_data_len, &t, 1);                            \
    ORT_ENFORCE(unpack_status.IsOK(), "Value attribute unpacking failed:", unpack_status.ErrorMessage()); \
    field = t;                                                                                            \
  }

void onnxruntime::ConstantOfShapeBase::SetValue(const ONNX_NAMESPACE::TensorProto& t_proto) {
  using namespace utils;
  ORT_ENFORCE(t_proto.has_data_type());
  ORT_ENFORCE(TensorProto::DataType_IsValid(t_proto.data_type()));
  tensor_type_ = static_cast<TensorProto_DataType>(t_proto.data_type());
  const void* const raw_data = t_proto.has_raw_data() ? t_proto.raw_data().data() : nullptr;
  const size_t raw_data_len = t_proto.has_raw_data() ? t_proto.raw_data().size() : 0;
  switch (tensor_type_) {
    case TensorProto::BOOL:
      FETCH_VALUE_DATA(value_.ui64_, bool);
      break;
    case TensorProto::FLOAT:
      FETCH_VALUE_DATA(value_.fl_, float);
      break;
    case TensorProto::FLOAT16:
      FETCH_VALUE_DATA(value_.fl16_, MLFloat16);
      break;
    case TensorProto::DOUBLE:
      FETCH_VALUE_DATA(value_.dbl_, double);
      break;
    case TensorProto::INT8:
      FETCH_VALUE_DATA(value_.i64_, int8_t);
      break;
    case TensorProto::INT16:
      FETCH_VALUE_DATA(value_.i64_, int16_t);
      break;
    case TensorProto::INT32:
      FETCH_VALUE_DATA(value_.i64_, int32_t);
      break;
    case TensorProto::INT64:
      FETCH_VALUE_DATA(value_.i64_, int64_t);
      break;
    case TensorProto::UINT8:
      FETCH_VALUE_DATA(value_.ui64_, uint8_t);
      break;
    case TensorProto::UINT16:
      FETCH_VALUE_DATA(value_.ui64_, uint16_t);
      break;
    case TensorProto::UINT32:
      FETCH_VALUE_DATA(value_.ui64_, uint32_t);
      break;
    case TensorProto::UINT64:
      FETCH_VALUE_DATA(value_.ui64_, uint64_t);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype: ", tensor_type_);
      break;
  }
}

#undef FETCH_VALUE_DATA

template <class T>
inline T onnxruntime::ConstantOfShapeBase::AttrValue::GetFromSigned() const {
  return static_cast<T>(i64_);
}

template <class T>
inline T onnxruntime::ConstantOfShapeBase::AttrValue::GetFromUnsigned() const {
  return static_cast<T>(ui64_);
}

template <class T>
inline void FilloutOutput(T value, Tensor* output_tensor) {
  auto out = gsl::make_span(output_tensor->template MutableData<T>(), output_tensor->Shape().Size());
  std::fill(out.begin(), out.end(), value);
}

ConstantOfShapeBase::ConstantOfShapeBase(const OpKernelInfo& info){
  TensorProto t_proto;
  if (info.GetAttr<TensorProto>("value", &t_proto).IsOK()) {
    ORT_ENFORCE(t_proto.dims_size() == 1, "Must have a single dimension");
    ORT_ENFORCE(t_proto.dims()[0] == 1, "Must have a single dimension of 1");
    SetValue(t_proto);
  } else {
    tensor_type_ = TensorProto::FLOAT;
    value_.fl_ = 0.f;
  }
}

Status ConstantOfShapeBase::PrepareCompute(OpKernelContext* ctx, Tensor** output_tensor) const {
  const auto shape_tensor = ctx->Input<Tensor>(0);
  const auto& input_shape = shape_tensor->Shape();

  // If empty the output is a scalar with empty shape
  // TensorShape::Size() will still return 1 and we will output
  // one value
  std::vector<int64_t> output_dims;
  ORT_ENFORCE(input_shape.NumDimensions() > 0, "Must have a valid input shape.");

  const auto span = gsl::make_span(shape_tensor->Data<int64_t>(), input_shape.Size());
  output_dims.insert(output_dims.end(), span.cbegin(), span.cend());

  TensorShape output_shape(output_dims);
  (*output_tensor) = ctx->Output(0, output_shape);

  return Status::OK();
}

void onnxruntime::ConstantOfShape::DispatchTypeAndFillOutput(Tensor* output_tensor) const {
  auto tensor_type = GetTensorType();
  switch (tensor_type) {
    case TensorProto::BOOL:
      FilloutOutput(GetAttrValue().GetFromUnsigned<bool>(), output_tensor);
      break;
    case TensorProto::FLOAT:
      FilloutOutput(GetAttrValue().GetFloat(), output_tensor);
      break;
    case TensorProto::FLOAT16:
      FilloutOutput(GetAttrValue().GetFloat16(), output_tensor);
      break;
    case TensorProto::DOUBLE:
      FilloutOutput(GetAttrValue().GetDouble(), output_tensor);
      break;
    case TensorProto::INT8:
      FilloutOutput(GetAttrValue().GetFromSigned<int8_t>(), output_tensor);
      break;
    case TensorProto::INT16:
      FilloutOutput(GetAttrValue().GetFromSigned<int16_t>(), output_tensor);
      break;
    case TensorProto::INT32:
      FilloutOutput(GetAttrValue().GetFromSigned<int32_t>(), output_tensor);
      break;
    case TensorProto::INT64:
      FilloutOutput(GetAttrValue().GetFromSigned<int64_t>(), output_tensor);
      break;
    case TensorProto::UINT8:
      FilloutOutput(GetAttrValue().GetFromUnsigned<uint8_t>(), output_tensor);
      break;
    case TensorProto::UINT16:
      FilloutOutput(GetAttrValue().GetFromUnsigned<uint16_t>(), output_tensor);
      break;
    case TensorProto::UINT32:
      FilloutOutput(GetAttrValue().GetFromUnsigned<uint32_t>(), output_tensor);
      break;
    case TensorProto::UINT64:
      FilloutOutput(GetAttrValue().GetFromUnsigned<uint64_t>(), output_tensor);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype: ", GetTensorType());
      break;
  }
}

Status ConstantOfShape::Compute(OpKernelContext* ctx) const {

  Tensor* output_tensor = nullptr;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor));

  DispatchTypeAndFillOutput(output_tensor);
  return Status::OK();
}
}  // namespace onnxruntime
