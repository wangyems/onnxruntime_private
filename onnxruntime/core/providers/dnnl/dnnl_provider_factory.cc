// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include <atomic>
#include <cassert>
#include "dnnl_execution_provider.h"

using namespace onnxruntime;

extern onnxruntime::ProviderHost* g_host;

namespace onnxruntime {

void SetProviderHost(ProviderHost& host);

struct DnnlProviderFactory : Provider_IExecutionProviderFactory {
  DnnlProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~DnnlProviderFactory() override {}

  std::unique_ptr<Provider_IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<Provider_IExecutionProvider> DnnlProviderFactory::CreateProvider() {
  DNNLExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return onnxruntime::make_unique<DNNLExecutionProvider>(info);
}

struct Dnnl_Provider : Provider {
  std::shared_ptr<Provider_IExecutionProviderFactory> CreateExecutionProviderFactory(int use_arena) override {
#if defined(_WIN32) && !defined(_OPENMP)
    {
      // We crash when unloading DNNL on Windows when OpenMP also unloads (As there are threads
      // still running code inside the openmp runtime DLL if OMP_WAIT_POLICY is set to ACTIVE).
      // To avoid this, we pin the OpenMP DLL so that it unloads as late as possible.
      HMODULE handle{};
#ifdef _DEBUG
      constexpr const char* dll_name = "vcomp140d.dll";
#else
      constexpr const char* dll_name = "vcomp140.dll";
#endif
      ::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, dll_name, &handle);
      assert(handle);  // It should exist
    }
#endif
    return std::make_shared<DnnlProviderFactory>(use_arena != 0);
  }

  void SetProviderHost(ProviderHost& host) {
    onnxruntime::SetProviderHost(host);
  }
} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
