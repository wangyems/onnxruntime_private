// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin>
void GatherImpl(
    const int64_t input_block_size,
    const int64_t indices_max,
    const Tin* indices_data,
    const fast_divmod* output_strides,
    const T* input_data,
    T* output_data,
    const size_t N);

template <typename T, typename Tin>
void GatherGradImpl(
    const int64_t input_block_size,
    const int64_t indices_max,
    const Tin* indices_data,
    const fast_divmod* output_strides,
    const T* grad_data,
    T* output_data,
    const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
