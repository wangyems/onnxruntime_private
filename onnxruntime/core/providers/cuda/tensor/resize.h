// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/tensor/upsample.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Resize : public Upsample<T> {
 public:
  Resize(OpKernelInfo info) : Upsample(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    return Upsample<T>::ComputeInternal(context);
  }
};

}  // namespace cuda
}  // namespace onnxruntime
