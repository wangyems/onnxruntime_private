// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reshape.h"

namespace onnxruntime {
namespace hip {

ONNX_OPERATOR_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    13,
    kHipExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType<OrtMemTypeCPUInput>(1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    5, 12,
    kHipExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType<OrtMemTypeCPUInput>(1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape_1,
    kOnnxDomain,
    1,
    4,
    kHipExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Reshape_1);

}  // namespace hip
}  // namespace onnxruntime
