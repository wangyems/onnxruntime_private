// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/framework/TestAllocatorManager.h"
#include "core/framework/allocatormgr.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_allocator.h"
#endif  //  USE_CUDA

namespace onnxruntime {
namespace test {

static std::string GetAllocatorId(const std::string& name, const int id, const bool isArena) {
  std::ostringstream ss;
  if (isArena)
    ss << "arena_";
  else
    ss << "device_";
  ss << name << "_" << id;
  return ss.str();
}

static Status RegisterAllocator(std::unordered_map<std::string, AllocatorPtr>& map,
                                std::unique_ptr<IDeviceAllocator> allocator, size_t /*memory_limit*/,
                                bool use_arena) {
  auto& info = allocator->Info();
  auto allocator_id = GetAllocatorId(info.name, info.id, use_arena);

  auto status = Status::OK();
  if (map.find(allocator_id) != map.end())
    status = Status(common::ONNXRUNTIME, common::FAIL, "allocator already exists");
  else {
    if (use_arena)
      map[allocator_id] = std::make_shared<DummyArena>(std::move(allocator));
    else
      map[allocator_id] = std::move(allocator);
  }

  return status;
}

AllocatorManager& AllocatorManager::Instance() {
  static AllocatorManager s_instance_;
  return s_instance_;
}

AllocatorManager::AllocatorManager() {
  InitializeAllocators();
}

Status AllocatorManager::InitializeAllocators() {
  auto cpu_alocator = onnxruntime::make_unique<CPUAllocator>();
  ORT_RETURN_IF_ERROR(RegisterAllocator(map_, std::move(cpu_alocator), std::numeric_limits<size_t>::max(), true));
#ifdef USE_CUDA
  auto cuda_alocator = onnxruntime::make_unique<CUDAAllocator>(static_cast<OrtDevice::DeviceId>(0), CUDA);
  ORT_RETURN_IF_ERROR(RegisterAllocator(map_, std::move(cuda_alocator), std::numeric_limits<size_t>::max(), true));

  auto cuda_pinned_alocator = onnxruntime::make_unique<CUDAPinnedAllocator>(static_cast<OrtDevice::DeviceId>(0), CUDA_PINNED);
  ORT_RETURN_IF_ERROR(RegisterAllocator(map_, std::move(cuda_pinned_alocator), std::numeric_limits<size_t>::max(), true));
#endif  // USE_CUDA

  return Status::OK();
}

AllocatorManager::~AllocatorManager() {
}

AllocatorPtr AllocatorManager::GetAllocator(const std::string& name, const int id, bool arena) {
  auto allocator_id = GetAllocatorId(name, id, arena);
  auto entry = map_.find(allocator_id);
  ORT_ENFORCE(entry != map_.end(), "Allocator not found:", allocator_id);
  return entry->second;
}
}  // namespace test
}  // namespace onnxruntime
