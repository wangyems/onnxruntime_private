// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
interface IMLOperatorRegistry;

#include "core/common/status.h"
#include "core/framework/data_transfer.h"
#include "IWinmlExecutionProvider.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlBufferRegion.h"

namespace onnxruntime
{
    class IExecutionProvider;
    class IAllocator;
    class CustomRegistry;
    class InferenceSession;
    class KernelRegistry;
}

namespace Dml
{
    std::unique_ptr<onnxruntime::IExecutionProvider> CreateExecutionProvider(
        IDMLDevice* dmlDevice,
        ID3D12CommandQueue* commandQueue,
        bool enableMetacommands,
        bool enableBfcAllocator);

    void FlushContext(onnxruntime::IExecutionProvider* provider);
    void ReleaseCompletedReferences(onnxruntime::IExecutionProvider* provider);

    onnxruntime::common::Status CopyTensor(
        onnxruntime::IExecutionProvider* provider,
        const onnxruntime::Tensor& src, onnxruntime::Tensor& dst
    );

    void* CreateGPUAllocationFromD3DResource(ID3D12Resource* pResource);
    void FreeGPUAllocation(void* ptr);

    void RegisterDmlOperators(IMLOperatorRegistry* registry);
    void RegisterCpuOperatorsAsDml(onnxruntime::KernelRegistry* registry);

} // namespace Dml
