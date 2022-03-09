// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_EXECUTION_PROVIDER_OPTIONS_H
#define TVM_EXECUTION_PROVIDER_OPTIONS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

#include "core/framework/provider_options.h"

#include "tvm_defaults.h"


namespace onnxruntime {

namespace tvm {
namespace cpu_targets {
// TODO(vvchernov): avx and avx512 need more careful differentiation for target
const std::string LLVM_TARGET_AVX = "llvm -mcpu=corei7-avx";
const std::string LLVM_TARGET_AVX2 = "llvm -mcpu=core-avx2";
const std::string LLVM_TARGET_SKYLAKE_AVX512 = "llvm -mcpu=skylake-avx512";
const std::string LLVM_TARGET_AVX512 = "llvm -mcpu=skylake-avx512";
}  // namespace cpu_targets

using TVMTensorShape = std::vector<int64_t>;
using TVMTensorShapes = std::vector<TVMTensorShape>;
using TVMInputShapes = std::unordered_map<std::string, TVMTensorShape>;
using InputsInfoMap = std::map<size_t, TVMTensorShape>;

// Information needed to construct an TVM execution provider.
struct TvmEPOptions {
  std::string executor{tvm::default_executor_type};
  std::string target{tvm::default_target_str};
  std::string target_host{tvm::default_target_str};
  unsigned int opt_level{tvm::default_opt_level};
  bool freeze_weights = true;
  bool to_nhwc = false;
  std::string tuning_file_path{""};
  std::string tuning_type{tvm::default_tuning_type};
  std::string input_names_str{""};
  std::string input_shapes_str{""};
  TVMInputShapes input_shapes{};
  TVMTensorShapes output_shapes{};

  static TvmEPOptions FromOptionsString(const char* options);
  static TvmEPOptions FromProviderOptions(const ProviderOptions& options);
  static std::string whitespace_trimming(const std::string& str);

  void optionsPostprocess();

  bool checkGPUTarget() const;
private:
  void setInputShapes();
  void targetPostprocess();
  void ProcessCPUTarget();
  void ProcessGPUTarget();
  void targetHostPostprocess();
  void optLevelPostprocess();
};

std::ostream& operator<<(std::ostream& out, const TvmEPOptions& options);

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_EXECUTION_PROVIDER_OPTIONS_H
