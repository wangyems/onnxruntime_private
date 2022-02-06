// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include "core/common/status.h"
#include "core/graph/graph_viewer.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
struct KernelCreateInfo;
class ExecutionProviders;
class IExecutionProvider;
class KernelRegistry;
class OpKernel;
class SessionState;

// Kernel registries' manager.
// There're 2 kinds of kernel registries with priority from high to low as below,
// 1. Custom execution provider type specific kernel registries.
// 2. common execution provider type specific kernel registries.
// The 1st and 2nd ones are shared across sessions.

// This class is not thread safe.
class KernelRegistryManager {
 public:
  KernelRegistryManager() = default;

  // Register kernels from providers and stores them in `stock_provider_registries_`
  Status RegisterKernels(const ExecutionProviders& execution_providers) ORT_MUST_USE_RESULT;

  // Register one kernel registry per-EP that will exist in `special_provider_registries_`
  // in addition to the kernel registry per-EP in `stock_provider_registries_`.
  // The precedence of these registries is such that they are of lower priority than the custom
  // registries but are of higher priority than the one in `stock_provider_registries_`.
  Status RegisterSpecialKernelRegistry(const std::string& type,
                                       std::shared_ptr<KernelRegistry> kernel_registry) ORT_MUST_USE_RESULT;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // The registry passed in this function has highest priority than anything already in this KernelRegistryManager,
  // and anything registered from RegisterKernels
  // For example, if you do:
  // RegisterKernels(providers)
  // RegisterSpecialKernelRegistry(special_kernels)
  // RegisterKernelRegistry(A);
  // RegisterKernelRegistry(B);
  // Then B > A > special_kernels > providers
  void RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry);

  /**
   * Search kernel registry by provider type.
   * @param type provider type string
   * @return It returns all the possible results. The returned value may contain garbage that doesn't belong to
   *         this provider. Caller should do the filtering. The returned value won't have no nullptrs.
   */
  std::vector<const KernelRegistry*> GetKernelRegistriesByProviderType(const std::string& type) const {
    std::vector<const KernelRegistry*> result;

    // First, look in all the custom registries
    for (auto& registry : custom_kernel_registries_) {
      result.push_back(registry.get());
    }

    // Second, look in the "special" EP registry
    auto iter = special_provider_registries_.find(type);
    if (iter != special_provider_registries_.end()) {
      result.push_back(iter->second.get());
    }

    // Third, look in the "stock" EP registry
    iter = stock_provider_registries_.find(type);
    if (iter != stock_provider_registries_.end()) {
      result.push_back(iter->second.get());
    }

    return result;
  }
#endif

#if !defined(ORT_MINIMAL_BUILD)
  // This function assumes the node is already assigned to an execution provider
  // Don't call this function before graph partition is done
  Status SearchKernelRegistry(const onnxruntime::Node& node,
                              /*out*/ const KernelCreateInfo** kernel_create_info) const;

  /**
   * Whether this node can be run on this provider
   */
  static bool HasImplementationOf(const KernelRegistryManager& r, const Node& node, const std::string& provider_type);
#endif

  /**
   * Search the kernel registries given a kernel def hash.
   */
  bool SearchKernelRegistriesByHash(HashValue kernel_def_hash,
                                    const KernelCreateInfo** kernel_create_info) const;

  Status CreateKernel(const onnxruntime::Node& node,
                      const IExecutionProvider& execution_provider,
                      SessionState& session_state,
                      const KernelCreateInfo& kernel_create_info, std::unique_ptr<OpKernel>& out) const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelRegistryManager);

 private:
  // Each kernel registry in this collection only belongs to one specific provider (key is provider type).
  // All the kernel registries in this container are the "stock" EP kernel registries (i.e.) the kernel
  // registries that come along with an EP implementation
  std::unordered_map<std::string, std::shared_ptr<KernelRegistry>> stock_provider_registries_;

  // Each kernel registry in this collection only belongs to one specific provider (key is provider type).
  // All the kernel registries in this container are the "stock" EP kernel registries (i.e.) the kernel
  // registries that come along with an EP implementation
  std::unordered_map<std::string, std::shared_ptr<KernelRegistry>> special_provider_registries_;

  // Each kernel registry may contain kernels from many different providers.
  // in order to search kernels from a specific provider, we have to iterate all its elements
  std::list<std::shared_ptr<KernelRegistry>> custom_kernel_registries_;
};
}  // namespace onnxruntime
