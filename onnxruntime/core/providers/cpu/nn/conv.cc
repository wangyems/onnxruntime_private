// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/cpu/nn/conv_impl.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <>
Status Conv<float>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;
  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(ComputeKernelShape(W->Shape(), kernel_shape));

  std::vector<int64_t> pads(pads_);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(dilations_);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(strides_);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const float* Xdata = X->template Data<float>();
  float* Ydata = Y->template MutableData<float>();

  const size_t kernel_rank = kernel_shape.size();

  if (kernel_rank == 2 || kernel_rank == 3) {
    MLAS_ACTIVATION Activation;
    if (activation_.empty()) {
        Activation.ActivationKind = MlasIdentityActivation;
    } else if (activation_ == "Relu") {
        Activation.ActivationKind = MlasReluActivation;
    } else if (activation_ == "LeakyRelu") {
        Activation.ActivationKind = MlasLeakyReluActivation;
        Activation.alpha = alpha_;
    } else if (activation_ == "Tanh") {
        Activation.ActivationKind = MlasTanhActivation;
    } else if (activation_ == "Sigmoid") {
        Activation.ActivationKind = MlasLogisticActivation;
    } else {
      ORT_NOT_IMPLEMENTED("Not implemented fused activation: ", activation_);
    }

    // Get access to the internal threadpool
    // Temporarily derive concurrency parameters without access to session state
    auto ctx_internal = static_cast<OpKernelContextInternal*>(context);
    auto thread_pool = ctx_internal->GetOperatorThreadPool();

    MLAS_CONV_PARAMETERS Parameters;
    size_t WorkingBufferSize;
    MlasConvPrepare(&Parameters,
                    kernel_rank,
                    static_cast<size_t>(N),
                    static_cast<size_t>(group_),
                    static_cast<size_t>(C / group_),
                    input_shape.GetDims().data(),
                    kernel_shape.data(),
                    dilations.data(),
                    pads.data(),
                    strides.data(),
                    output_shape.GetDims().data(),
                    static_cast<size_t>(M / group_),
                    &Activation,
                    &WorkingBufferSize,
                    thread_pool->NumThreads());

    auto working_data = WorkingBufferSize > 0 ? alloc->Alloc(sizeof(float) * WorkingBufferSize) : nullptr;
    BufferUniquePtr working_buffer(working_data, BufferDeleter(alloc));

    MlasConv(&Parameters,
             Xdata,
             W->template Data<float>(),
             B != nullptr ? B->template Data<float>() : nullptr,
             static_cast<float*>(working_buffer.get()),
             Ydata,
             const_cast<concurrency::ThreadPool*>(thread_pool));
  } else {
    const int64_t input_image_size = input_shape.Size();
    const int64_t output_image_size = output_shape.Size();
    const int64_t kernel_size = TensorShape(kernel_shape).Size();
    const int64_t X_offset = C / group_ * input_image_size;
    const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / group_;
    const int64_t W_offset = W->Shape().Size() / group_;
    const int64_t kernel_dim = C / group_ * kernel_size;
    const int64_t col_buffer_size = kernel_dim * output_image_size;

    auto col_data = alloc->Alloc(sizeof(float) * col_buffer_size);
    BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
    float* col_buffer_data = static_cast<float*>(col_buffer.get());

    TensorShape image_shape = X->Shape().Slice(1);
    std::vector<int64_t> col_buffer_shape{kernel_dim};
    col_buffer_shape.insert(col_buffer_shape.end(), output_shape.GetDims().begin(),
                            output_shape.GetDims().end());

    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        math::Im2colNd<float, CPUMathUtil, StorageOrder::NCHW>()(
            Xdata + group_id * X_offset,
            image_shape.GetDims().data(),
            col_buffer_shape.data(),
            C * input_image_size,
            col_buffer_size,
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            static_cast<int>(kernel_shape.size()),
            col_buffer_data,
            &CPUMathUtil::Instance());
        math::Gemm<float, CPUMathUtil>(
            CblasNoTrans,
            CblasNoTrans,
            M / group_,
            output_image_size,
            kernel_dim,
            1,
            W->template Data<float>() + group_id * W_offset,
            col_buffer_data,
            0,
            Ydata + group_id * Y_offset,
            &CPUMathUtil::Instance());
      }

      if (B != nullptr) {
        auto Ymatrix = EigenMatrixMap<float>(Ydata, output_image_size, M);
        auto Bvec = ConstEigenVectorMap<float>(B->template Data<float>(), M);
        Ymatrix.rowwise() += Bvec.transpose();
      }

      FuseActivation(activation_, Ydata, Y_offset * group_, alpha_);

      Xdata += X_offset * group_;
      Ydata += Y_offset * group_;
    }
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    Conv,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);
}  // namespace onnxruntime
