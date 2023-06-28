// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// #include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {

template <typename T>
void BatchScaleImpl(cudaStream_t stream,
                    int64_t input_element_count,
                    const T* input_data,
                    const std::vector<const T*>& scales,
                    const std::vector<T*>& outputs);

}  // namespace cuda
}  // namespace onnxruntime
