// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/nuphar_provider_factory.h"
#include <atomic>
#include "nuphar_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
//#include "core/codegen/passes/utils/codegen_context.h"  // TODO: remove it

namespace onnxruntime {
struct NupharExecutionProviderFactory : IExecutionProviderFactory {
  NupharExecutionProviderFactory(bool allow_unaligned_buffers, int device_id, const char* settings)
      : device_id_(device_id),
        settings_(settings),
        allow_unaligned_buffers_(allow_unaligned_buffers) {}
  ~NupharExecutionProviderFactory() = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  int device_id_;
  std::string settings_;
  bool allow_unaligned_buffers_;
};

std::unique_ptr<IExecutionProvider> NupharExecutionProviderFactory::CreateProvider() {
  NupharExecutionProviderInfo info(allow_unaligned_buffers_, device_id_, settings_, /*per_node_parallel*/ true);
  return std::make_unique<NupharExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool allow_unaligned_buffers, int device_id, const char* settings) {
  return std::make_shared<onnxruntime::NupharExecutionProviderFactory>(allow_unaligned_buffers, device_id, settings);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nuphar, _In_ OrtSessionOptions* options, int allow_unaligned_buffers, int device_id, _In_ const char* settings) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Nuphar(static_cast<bool>(allow_unaligned_buffers), device_id, settings));
  return nullptr;
}
