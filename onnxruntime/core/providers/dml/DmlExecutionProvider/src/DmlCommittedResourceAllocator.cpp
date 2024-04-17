// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlCommittedResourceAllocator.h"
#include "DmlResourceWrapper.h"
#include "DmlCommittedResourceWrapper.h"

namespace Dml
{
    DmlCommittedResourceAllocator::DmlCommittedResourceAllocator(ID3D12Device* device) :
      m_device(device),
      m_resources(std::make_shared<std::vector<ID3D12Pageable*>>())
    { }

    ComPtr<DmlResourceWrapper> DmlCommittedResourceAllocator::Alloc(size_t size)
    {
        ComPtr<ID3D12Resource> resource;
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ORT_THROW_IF_FAILED(m_device->CreateCommittedResource(
            unmove_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)),
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(resource.GetAddressOf())
        ));

        //We keep a list of the pageable resources so we can control their residency
        //Please note that this would keep the resources alive
        ID3D12Pageable* pageable;
        ORT_THROW_IF_FAILED(resource->QueryInterface(&pageable));
        m_resources->push_back(pageable);

        //So we provide a callback for deleting our references when the resource wrapper is destroyed
        //We also need to use a weak reference to our reference list, as the allocator might be destroyed before the resource wrappers do
        ComPtr<DmlResourceWrapper> resourceWrapper;
        wil::MakeOrThrow<DmlCommittedResourceWrapper>(std::move(resource)).As(&resourceWrapper);
        resourceWrapper->AddReleaseCallback(&DmlCommittedResourceAllocator::OnResourceRelease, new std::weak_ptr(m_resources));

        return resourceWrapper;
    }

    DmlCommittedResourceAllocator::~DmlCommittedResourceAllocator()
    {
        for (auto item : *m_resources)
        {
            item->Release();
        }
    }

    void DmlCommittedResourceAllocator::SetResidency(bool value)
    {
        if (m_isResident == value) return;

        if (value)
        {
            ORT_THROW_IF_FAILED(m_device->MakeResident(UINT(m_resources->size()), m_resources->data()));
        }
        else
        {
            ORT_THROW_IF_FAILED(m_device->Evict(UINT(m_resources->size()), m_resources->data()));
        }

        m_isResident = value;
    }

    void DmlCommittedResourceAllocator::OnResourceRelease(void * context, ID3D12Resource * resource)
    {
        //Retrieve the weak reference to the reference list
        //If the allocator is destroyed by this time the `resources` will be nullptr        
        auto resourcesRef = static_cast<std::weak_ptr<std::vector<ID3D12Pageable*>>*>(context);
        auto resources = resourcesRef->lock();
        delete resourcesRef;

        if (!resources) return;

        //We retrieve the pageable of the destroyed resource
        ComPtr<ID3D12Pageable> pageable;
        ORT_THROW_IF_FAILED(resource->QueryInterface(pageable.GetAddressOf()));

        //Then remove it from the list of resources and release it
        for (auto& item : *resources)
        {
            if (item == resource)
            {
                item->Release();

                std::swap(item, resources->back());
                resources->pop_back();
                break;
            }
        }
    }
}
