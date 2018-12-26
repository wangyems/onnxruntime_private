// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

class SliceBase {
 protected:
  SliceBase(const OpKernelInfo& info) {
    has_axes_ = info.GetAttrs("axes", axes_).IsOK();
    info.GetAttrs("starts", starts_);
    info.GetAttrs("ends", ends_);
  }

  Status PrepareForCompute(const size_t dimension_count, const std::vector<int64_t>& input_dimensions,
                           std::vector<int64_t>& starts, std::vector<int64_t>& output_dims) const;

  mutable bool has_axes_;
  mutable std::vector<int64_t> starts_, ends_, axes_;
};

template <typename T, typename Tind>
struct Slice final : public OpKernel, public SliceBase {
  Slice(const OpKernelInfo& info) : OpKernel(info), SliceBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};  // namespace onnxruntime

}  // namespace onnxruntime
