// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear_matmul.h"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    QLinearMatMul,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearMatMul);

Status QLinearMatMul::Compute(OpKernelContext* ctx) const {
  MatMulComputeHelper helper;
  const auto* a = ctx->Input<Tensor>(IN_A);

  const uint8_t* b_start;
  bool b_signed;  // can't modify b_is_signed_, this is a const method
  if (packed_b_) {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_));
    b_start = static_cast<uint8_t *>(packed_b_.get());
    b_signed = b_is_signed_;
  } else {
    const Tensor* b = ctx->Input<Tensor>(IN_B);
    if (b == nullptr) {
      // the framework has checks to ensure this won't happen,
      // just need this to shutup static analysis. 
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
          "Required input B can not be null!");
    }
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
    b_start = static_cast<const uint8_t*>(b->DataRaw());
    b_signed = b->IsDataType<int8_t>();
  }

  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  // validate offsets
  const auto* a_offset = ctx->Input<Tensor>(IN_Azero);
  const auto* b_offset = ctx->Input<Tensor>(IN_Bzero);
  const auto* y_offset = ctx->Input<Tensor>(IN_Yzero);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_offset),
              "QLinearMatmul : weight zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  const auto* a_scale = ctx->Input<Tensor>(IN_Ascale);
  const auto* b_scale = ctx->Input<Tensor>(IN_Bscale);
  const auto* y_scale = ctx->Input<Tensor>(IN_Yscale);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale),
              "QLinearMatmul : weight scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  auto a_scale_data = *(a_scale->template Data<float>());
  auto b_scale_data = *(b_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  const float real_multiplier = (a_scale_data * b_scale_data) / y_scale_data;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
  auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) *
                                       static_cast<size_t>(helper.M()) * static_cast<size_t>(helper.N()));
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());

  MLAS_GEMM_U8X8_PARAMETERS gemm_params;
  gemm_params.M = static_cast<size_t>(helper.M());
  gemm_params.N = static_cast<size_t>(helper.N());
  gemm_params.K = static_cast<size_t>(helper.K());
  gemm_params.lda = gemm_params.K;
  gemm_params.ZeroPointA = *a_offset->template Data<uint8_t>();
  gemm_params.ldb = gemm_params.N;
  gemm_params.ZeroPointB = static_cast<const uint8_t*>(b_offset->DataRaw());
  gemm_params.C = gemm_output;
  gemm_params.ldc = gemm_params.N;
  gemm_params.BIsPacked = bool(packed_b_);
  gemm_params.BIsSigned = b_signed;

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    gemm_params.A = a->template Data<uint8_t>() + helper.LeftOffsets()[i];
    gemm_params.B = b_start + (gemm_params.BIsPacked ? 0UL : helper.RightOffsets()[i]);

    MlasGemm(&gemm_params, ctx->GetOperatorThreadPool());

    MlasRequantizeOutput(gemm_output,
                         y->template MutableData<uint8_t>() + helper.OutputOffsets()[i],
                         nullptr,
                         static_cast<size_t>(helper.M()),
                         static_cast<size_t>(helper.N()),
                         &real_multiplier,
                         false,
                         *y_offset->template Data<uint8_t>());
  }

  return Status::OK();
}

}  // namespace onnxruntime
