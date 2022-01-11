// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <memory>

#include "core/providers/tvm/tvm_provider_factory.h"
#include "core/session/abi_session_options_impl.h"

#include "tvm_execution_provider.h"


namespace onnxruntime {

struct StvmProviderFactory : IExecutionProviderFactory {
  StvmProviderFactory(const TvmExecutionProviderInfo& info) : info_{info} {}
  ~StvmProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<StvmExecutionProvider>(info_);
 }

 private:
    TvmExecutionProviderInfo info_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Stvm(const char* settings) {
    TvmExecutionProviderInfo info = TvmExecutionProviderInfo::FromOptionsString(settings);
    return std::make_shared<StvmProviderFactory>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Stvm(const TvmExecutionProviderInfo& info)
{
    return std::make_shared<StvmProviderFactory>(info);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Stvm,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* settings) {
  onnxruntime::TvmExecutionProviderInfo info = onnxruntime::TvmExecutionProviderInfo::FromOptionsString(settings);
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Stvm(info));
  return nullptr;
}
