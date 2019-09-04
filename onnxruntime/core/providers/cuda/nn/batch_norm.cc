// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "batch_norm.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace std;
namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                           \
      BatchNormalization,                                            \
      kOnnxDomain,                                                   \
      7, 8,                                                          \
      T,                                                             \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .TypeConstraint("X", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("scale", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("mean", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("var", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      BatchNormalization,                                            \
      kOnnxDomain,                                                   \
      9,                                                             \
      T,                                                             \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .TypeConstraint("X", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("scale", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("mean", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("var", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);

template <typename T>
Status BatchNorm<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* B = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* var = p_op_kernel_context->Input<Tensor>(4);

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var));

  const TensorShape& x_shape = X->Shape();
  const TensorShape& channel_shape = mean->Shape();

  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  Tensor* running_mean = p_op_kernel_context->Output(1, channel_shape);
  Tensor* running_var = p_op_kernel_context->Output(2, channel_shape);
  Tensor* saved_mean = p_op_kernel_context->Output(3, channel_shape);
  Tensor* saved_var = p_op_kernel_context->Output(4, channel_shape);

  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->template Data<T>());
  auto b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
  auto mean_data = reinterpret_cast<const CudaT*>(mean->template Data<T>());
  auto var_data = reinterpret_cast<const CudaT*>(var->template Data<T>());

  auto y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  CudnnTensor data_desc;
  vector<int64_t> new_dims;
  BatchNormHelper::NormalizeDims(x_shape, new_dims);
  ORT_RETURN_IF_ERROR(data_desc.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));

  // For half data type, the alpha, beta, scale, B, mean, var need to be float type
  if (X->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    CudnnTensor scale_desc;
    ORT_RETURN_IF_ERROR(scale_desc.Set(new_dims, CudnnTensor::GetDataType<float>()));
    CudnnTensor bn_tensor_desc;
    ORT_RETURN_IF_ERROR(bn_tensor_desc.Set(data_desc, cudnn_batch_norm_mode_));

    // Convert the scale, B, mean, var to float
    const int64_t C = x_shape.GetDims()[1];
    auto f_scale = GetScratchBuffer<float>(C);
    auto f_B = GetScratchBuffer<float>(C);
    auto f_mean = GetScratchBuffer<float>(C);
    auto f_var = GetScratchBuffer<float>(C);
    Impl_Cast<CudaT, float>(scale_data, f_scale.get(), C);
    Impl_Cast<CudaT, float>(b_data, f_B.get(), C);
    Impl_Cast<CudaT, float>(mean_data, f_mean.get(), C);
    Impl_Cast<CudaT, float>(var_data, f_var.get(), C);

    CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardInference(
        CudnnHandle(),
        cudnn_batch_norm_mode_,
        &alpha,
        &beta,
        data_desc,
        x_data,
        data_desc,
        y_data,
        bn_tensor_desc,
        f_scale.get(),
        f_B.get(),
        f_mean.get(),
        f_var.get(),
        epsilon_));

    return Status::OK();
  }

  CudnnTensor bn_tensor_desc;
  ORT_RETURN_IF_ERROR(bn_tensor_desc.Set(data_desc, cudnn_batch_norm_mode_));

  // in BatchNorm Forward Training mode if all 5 outputs present
  if (running_mean && running_var && saved_mean && saved_var) {
    auto running_mean_data = reinterpret_cast<CudaT*>(running_mean->template MutableData<T>());
    auto running_var_data = reinterpret_cast<CudaT*>(running_var->template MutableData<T>());
    auto saved_mean_data = reinterpret_cast<CudaT*>(saved_mean->template MutableData<T>());
    auto saved_inv_var_data = reinterpret_cast<CudaT*>(saved_var->template MutableData<T>());

    CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardTraining(
        CudnnHandle(),
        cudnn_batch_norm_mode_,
        &alpha,
        &beta,
        data_desc,
        x_data,
        data_desc,
        y_data,
        bn_tensor_desc,
        scale_data,
        b_data,
        momentum_,
        running_mean_data,
        running_var_data,
        epsilon_,
        saved_mean_data,
        saved_inv_var_data));
    // in BatchNorm Forward Inference mode if only Y output present
  } else {
    CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardInference(
        CudnnHandle(),
        cudnn_batch_norm_mode_,
        &alpha,
        &beta,
        data_desc,
        x_data,
        data_desc,
        y_data,
        bn_tensor_desc,
        scale_data,
        b_data,
        mean_data,
        var_data,
        epsilon_));
  }
  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status BatchNorm<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      BatchNormalizationGrad,                                                   \
      kOnnxDomain,                                                              \
      9,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BatchNormalizationGrad<T>);

template <typename T>
Status BatchNormalizationGrad<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* dY = ctx->Input<Tensor>(0);
  const Tensor* X = ctx->Input<Tensor>(1);
  const Tensor* Scale = ctx->Input<Tensor>(2);
  const Tensor* saved_mean = ctx->Input<Tensor>(3);
  const Tensor* saved_variance = ctx->Input<Tensor>(4);
  const TensorShape input_shape = X->Shape();
  const TensorShape channel_shape = saved_mean->Shape();

  // no B here, but B has same size as Scale, so can validate inputs for gradient with this substitute
  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, Scale, Scale, saved_mean, saved_variance));

  auto dY_data = reinterpret_cast<const CudaT*>(dY->template Data<T>());
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto Scale_data = reinterpret_cast<const CudaT*>(Scale->template Data<T>());
  auto saved_mean_data = reinterpret_cast<const CudaT*>(saved_mean->template Data<T>());
  auto saved_variance_data = reinterpret_cast<const CudaT*>(saved_variance->template Data<T>());

  auto dX_data = reinterpret_cast<CudaT*>(ctx->Output(0, input_shape)->template MutableData<T>());
  auto dScale_data = reinterpret_cast<CudaT*>(ctx->Output(1, channel_shape)->template MutableData<T>());
  auto dBias_data = reinterpret_cast<CudaT*>(ctx->Output(2, channel_shape)->template MutableData<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  CudnnTensor input_tensor, scale_bias_tensor;
  vector<int64_t> new_dims;
  BatchNormHelper::NormalizeDims(input_shape, new_dims);
  ORT_RETURN_IF_ERROR(input_tensor.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(scale_bias_tensor.Set(input_tensor, cudnn_batch_norm_mode_));

  // note this is only valid for cudnnBatchNormalizationForwardTraining, not ForwardInference
  CUDNN_RETURN_IF_ERROR(
      cudnnBatchNormalizationBackward(
          CudnnHandle(),
          cudnn_batch_norm_mode_,
          &alpha,
          &beta,
          &alpha,
          &beta,
          input_tensor,
          X_data,
          input_tensor,
          dY_data,
          input_tensor,
          dX_data,
          scale_bias_tensor,
          Scale_data,
          dScale_data,
          dBias_data,
          epsilon_,
          saved_mean_data,
          saved_variance_data));
  return Status::OK();
}

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status BatchNormalizationGrad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
SPECIALIZED_GRADIENT(double)

}  // namespace cuda
}  // namespace onnxruntime
