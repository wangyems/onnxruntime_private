// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlReservedResourceWrapper.h"

namespace Dml
{
    class DmlSubAllocator;

    class AllocationInfo : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown>
    {
    public:
        AllocationInfo(
            DmlSubAllocator* owner,
            size_t id,
            uint64_t pooledResourceId,
            DmlResourceWrapper* resourceWrapper,
            size_t requestedSize)
            : m_owner(owner)
            , m_allocationId(id)
            , m_pooledResourceId(pooledResourceId)
            , m_resourceWrapper(resourceWrapper)
            , m_requestedSize(requestedSize)
        {}

        ~AllocationInfo();

        DmlSubAllocator* GetOwner() const
        {
            return m_owner;
        }

        ID3D12Resource* GetD3D12Resource() const
        {
            return m_resourceWrapper->GetD3D12Resource();
        }

        ComPtr<DmlResourceWrapper> DetachResourceWrapper()
        {
            return std::move(m_resourceWrapper);
        }

        size_t GetRequestedSize() const
        {
            return m_requestedSize;
        }

        size_t GetId() const
        {
            return m_allocationId;
        }

        uint64_t GetPooledResourceId() const
        {
            return m_pooledResourceId;
        }

    private:
        DmlSubAllocator* m_owner;
        size_t m_allocationId; // For debugging purposes
        uint64_t m_pooledResourceId;
        Microsoft::WRL::ComPtr<DmlResourceWrapper> m_resourceWrapper;

        // The size requested during Alloc(), which may be smaller than the physical resource size
        size_t m_requestedSize;
    };
} // namespace Dml
