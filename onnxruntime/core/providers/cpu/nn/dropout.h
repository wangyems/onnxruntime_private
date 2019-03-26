// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "core/common/common.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <random>
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class Dropout final : public OpKernel {
 public:
  Dropout(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("ratio", &ratio_).IsOK());
    keep_prob_ = 1.0f - ratio_;

    // TODO: enable following when is_train is present
    /*int64_t is_train = 1;
      ORT_ENFORCE(info.GetAttr("is_train", &is_train).IsOK());
      is_train_ = (is_train == 1);*/
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool is_train_ = false;
  float ratio_;
  float keep_prob_;
};
}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {
class DropoutGrad final : public OpKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("ratio", &ratio_).IsOK());
    keep_prob_ = 1.0f - ratio_;

    // TODO: enable following when is_train is present
    /*int64_t is_train = 1;
        ORT_ENFORCE(info.GetAttr("is_train", &is_train).IsOK());
        is_train_ = (is_train == 1);*/
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool is_train_ = false;
  float ratio_;
  float keep_prob_;
};

}  // namespace contrib
}  // namespace onnxruntime
