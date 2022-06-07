// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

class Range : public OpKernel {
 public:
  explicit Range(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
