// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_context_internal.h"
#include "nchwc_ops.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

#define ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kMSNchwcDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    ReorderInput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderInput<float>);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    ReorderOutput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderOutput<float>);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    Conv,
    1,
    float,
    KernelDefBuilder()
        .MayInplace(3, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcConv<float>);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    MaxPool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcMaxPool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    GlobalMaxPool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcMaxPool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    AveragePool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcAveragePool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    GlobalAveragePool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcAveragePool);

template <typename T>
Status ReorderInput<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);
  Tensor* Y = context->Output(0, X_shape);
  MlasReorderInput(X_shape.GetDims().data(), X->template Data<T>(), Y->template MutableData<T>());
  return Status::OK();
}

template <typename T>
Status ReorderOutput<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  std::vector<int64_t> Y_shape(X_shape.GetDims());
  ORT_ENFORCE(channels_ <= Y_shape[1]);
  Y_shape[1] = channels_;
  Tensor* Y = context->Output(0, Y_shape);
  MlasReorderOutput(Y_shape.data(), X->template Data<T>(), Y->template MutableData<T>());
  return Status::OK();
}

template <typename T>
Status NchwcConv<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);
  const Tensor* Sum = context->Input<Tensor>(3);

  ORT_RETURN_IF_ERROR(ConvBase::ValidateInputShape(X, W));

  const TensorShape& X_shape = X->Shape();
  const TensorShape& W_shape = W->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);

  const size_t nchwc_block_size = MlasNchwcGetBlockSize();
  ORT_ENFORCE((static_cast<size_t>(X_shape[1]) < nchwc_block_size) || ((X_shape[1] % nchwc_block_size) == 0));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(ConvBase::ComputeKernelShape(W_shape, kernel_shape));
  if (kernel_shape.size() != 2) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported convolution size.");
  }

  std::vector<int64_t> pads(ConvBase::pads_);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(ConvBase::dilations_);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(ConvBase::strides_);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {X_shape[0], W_shape[0]});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(ConvBase::InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, Y_dims);
  T* y_data = Y->template MutableData<T>();

  // Check for the optional Conv/Sum fusion.
  if (Sum != nullptr) {
    const auto& sum_shape = Sum->Shape();
    ORT_RETURN_IF_NOT(Y->Shape() == sum_shape, "output and sum shape must match");
    // If the output was not allocated inplace with the sum tensor, then copy here.
    const float* sum_data = Sum->template Data<T>();
    if (y_data != sum_data) {
      memcpy(y_data, sum_data, sum_shape.Size() * sizeof(T));
    }
  }

  MLAS_ACTIVATION Activation;
  if (ConvBase::activation_.empty()) {
    Activation.ActivationKind = MlasIdentityActivation;
  } else if (ConvBase::activation_ == "Relu") {
    Activation.ActivationKind = MlasReluActivation;
  } else if (ConvBase::activation_ == "LeakyRelu") {
    Activation.ActivationKind = MlasLeakyReluActivation;
    Activation.alpha = ConvBase::alpha_;
  } else if (ConvBase::activation_ == "Tanh") {
    Activation.ActivationKind = MlasTanhActivation;
  } else if (ConvBase::activation_ == "Sigmoid") {
    Activation.ActivationKind = MlasLogisticActivation;
  } else {
    ORT_NOT_IMPLEMENTED("Not implemented fused activation: ", ConvBase::activation_);
  }

  MlasNchwcConv(kernel_shape.size(),
                X_shape.GetDims().data(),
                kernel_shape.data(),
                dilations.data(),
                pads.data(),
                strides.data(),
                Y_dims.data(),
                static_cast<size_t>(ConvBase::group_),
                X->template Data<float>(),
                W->template Data<float>(),
                B != nullptr ? B->template Data<float>() : nullptr,
                y_data,
                &Activation,
                Sum == nullptr,
                const_cast<concurrency::ThreadPool*>(static_cast<OpKernelContextInternal*>(context)->GetOperatorThreadPool()));

  return Status::OK();
}

Status NchwcPoolBase::NchwcPool(OpKernelContext* context, MLAS_POOLING_KIND kind) const {
  const Tensor* X = context->Input<Tensor>(0);

  const TensorShape& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);

  if (!global_pooling_) {
    ORT_RETURN_IF_NOT(kernel_shape_.size() == 2, "kernel_shape num_dims is not compatible with X num_dims.");
  }

  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> output_dims = PoolBase::SetOutputSize(X_shape, X_shape[1], &pads, dilations_, ceil_mode_);
  Tensor* Y = context->Output(0, output_dims);

  MlasNchwcPool(kind,
                2,
                X_shape.GetDims().data(),
                global_pooling_ ? nullptr : kernel_shape_.data(),
                global_pooling_ ? nullptr : dilations_.data(),
                global_pooling_ ? nullptr : pads.data(),
                global_pooling_ ? nullptr : strides_.data(),
                output_dims.data(),
                X->template Data<float>(),
                Y->template MutableData<float>(),
                const_cast<concurrency::ThreadPool*>(static_cast<OpKernelContextInternal*>(context)->GetOperatorThreadPool()));

  return Status::OK();
}

Status NchwcMaxPool::Compute(OpKernelContext* context) const {
  return NchwcPoolBase::NchwcPool(context, MlasMaximumPooling);
}

Status NchwcAveragePool::Compute(OpKernelContext* context) const {
  return NchwcPoolBase::NchwcPool(context, count_include_pad_ ? MlasAveragePoolingIncludePad : MlasAveragePoolingExcludePad);
}

}  // namespace contrib
}  // namespace onnxruntime
