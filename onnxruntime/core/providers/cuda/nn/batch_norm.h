// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl_util"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class BatchNorm final : public CudaKernel {
 public:
  BatchNorm(const OpKernelInfo& op_kernel_info)
      : CudaKernel{op_kernel_info},
        cudnn_batch_norm_mode_(CUDNN_BATCHNORM_SPATIAL) {
    float tmp_epsilon;
    ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
    epsilon_ = ClampCudnnBatchNormEpsilon(static_cast<double>(tmp_epsilon));

    // spatial or not
    int64_t tmp_spatial;
    if (op_kernel_info.GetAttr<int64_t>("spatial", &tmp_spatial).IsOK()) {
      spatial_ = tmp_spatial;
    }

    if (spatial_ == 0) {
      cudnn_batch_norm_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;  // TODO add test case for this when implemented in CPU as well.
    }

    float tmp_momentum;
    if (op_kernel_info.GetAttr<float>("momentum", &tmp_momentum).IsOK()) {
      momentum_ = static_cast<double>(tmp_momentum);
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  double epsilon_;
  int64_t spatial_ = 1;  // default as per spec
  cudnnBatchNormMode_t cudnn_batch_norm_mode_;
  double momentum_;
};

template <typename T>

class BatchNormalizationGrad final : public CudaKernel {
 public:
  BatchNormalizationGrad(const OpKernelInfo& info) 
        : CudaKernel{info},
        cudnn_batch_norm_mode_(CUDNN_BATCHNORM_SPATIAL) {
    float tmp_epsilon;
    ORT_ENFORCE(info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
    epsilon_ = ClampCudnnBatchNormEpsilon(static_cast<double>(tmp_epsilon));

    // spatial or not
    int64_t tmp_spatial;
    if (info.GetAttr<int64_t>("spatial", &tmp_spatial).IsOK()) {
      spatial_ = tmp_spatial;
    }

    if (spatial_ == 0) {
      cudnn_batch_norm_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  double epsilon_;
  int64_t spatial_ = 1;
  cudnnBatchNormMode_t cudnn_batch_norm_mode_;
};

}  // namespace cuda
}  // namespace onnxruntime
