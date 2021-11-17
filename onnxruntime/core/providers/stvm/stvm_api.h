// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_API_H
#define STVM_API_H

#include "stvm_common.h"

namespace stvm {
    tvm::runtime::Module TVMCompile(const std::string& onnx_txt,
                                    const std::string& target,
                                    const std::string& target_host,
                                    int opt_level,
                                    int opset,
                                    bool freeze_params,
                                    const std::vector<std::vector<int64_t>>& input_shapes,
                                    const std::string& tuning_logfile);
    void TVMSetInputs(tvm::runtime::Module& mod, std::vector<size_t>& inds, std::vector<DLTensor>& inputs);
    void TVMRun(tvm::runtime::Module& mod, std::vector<DLTensor>& outputs, tvm::runtime::TVMRetValue *ret);
}  // namespace stvm

#endif  // STVM_API_H