// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"

namespace onnxruntime::cuda {

onnxruntime::common::Status RegisterCudaNhwcKernels(onnxruntime::KernelRegistry& kernel_registry);

}  // namespace onnxruntime::cuda

#ifndef DISABLE_CONTRIB_OPS
namespace onnxruntime::contrib::cuda {

onnxruntime::common::Status RegisterCudaNhwcContribKernels(onnxruntime::KernelRegistry& kernel_registry);

}  // namespace onnxruntime::contrib::cuda
#endif
