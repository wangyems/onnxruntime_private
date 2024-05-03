// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class FusedConv : public onnxruntime::cuda::Conv<T, false> {
 public:
  using Base = onnxruntime::cuda::Conv<T, false>;
  FusedConv(const OpKernelInfo& info) : onnxruntime::cuda::Conv<T, false>(info) {
    std::string activation;
    ORT_THROW_IF_ERROR(info.GetAttr<std::string>("activation", &activation));
    ORT_THROW_IF_ERROR(MapMode(activation));
    Base::is_fused_node_ = true;
    Base::s_.cudnn_fe_act_attr = cudnn_frontend::graph::Pointwise_attributes().set_mode(activation_mode_fe_);
    // fallback in case fusion fails
    CUDNN_CALL_THROW(cudnnCreateActivationDescriptor(&activation_desc_));
    CUDNN_CALL_THROW(cudnnSetActivationDescriptor(
        activation_desc_, activation_mode_, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        std::numeric_limits<double>::max()));
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FusedConv);

  Status ComputeInternal(OpKernelContext* context) const override {
    ORT_RETURN_IF_ERROR(Base::ComputeInternal(context));
    std::lock_guard<OrtMutex> lock(Base::s_.mutex);
    typedef typename onnxruntime::cuda::ToCudaType<T>::MappedType CudaT;
    auto cudnnHandle = this->GetCudnnHandle(context);
    const auto alpha = onnxruntime::cuda::Consts<CudaT>::One;
    const auto beta = onnxruntime::cuda::Consts<CudaT>::Zero;
    if (!Base::s_.act_fused) {
      ORT_RETURN_IF_ERROR(Base::s_.y_tensor.Set(
          Base::s_.y_dims.AsShapeVector(), ::onnxruntime::cuda::CudnnTensor::GetDataType<CudaT>()));
      CUDNN_RETURN_IF_ERROR(cudnnActivationForward(cudnnHandle, activation_desc_, &alpha, Base::s_.y_tensor,
                                                   Base::s_.y_data, &beta, Base::s_.y_tensor, Base::s_.y_data));
    }
    return Status::OK();
  }

 private:
  Status MapMode(const std::string& activaton_mode) {
    if (activaton_mode == "Relu") {
      activation_mode_fe_ = cudnn_frontend::PointwiseMode_t::RELU_FWD;
      activation_mode_ = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU;
    } else if (activaton_mode == "Tanh") {
      activation_mode_fe_ = cudnn_frontend::PointwiseMode_t::TANH_FWD;
      activation_mode_ = cudnnActivationMode_t::CUDNN_ACTIVATION_TANH;
    } else {
      return ORT_MAKE_STATUS(
          StatusCategory::ONNXRUNTIME, StatusCode::INVALID_ARGUMENT,
          "unsupported conv activation mode \"", activaton_mode, "\"");
    }
    return Status::OK();
  }
  cudnn_frontend::PointwiseMode_t activation_mode_fe_;
  cudnnActivationMode_t activation_mode_;
  cudnnActivationDescriptor_t activation_desc_ = nullptr;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv<float>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
