// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class Attention final : public CudaKernel, public AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  mutable std::unique_ptr<MHARunner> fused_fp16_runner_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
