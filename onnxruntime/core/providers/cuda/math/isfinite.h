// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename TSrc>
class IsFiniteOp final : public CudaKernel {
 public:
  IsFiniteOp(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename TSrc>
void IsFinite(const TSrc* input, bool* output, size_t N);

}  // namespace cuda
}  // namespace onnxruntime