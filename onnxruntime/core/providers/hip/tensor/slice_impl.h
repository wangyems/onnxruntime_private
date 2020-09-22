// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/hip/shared_inc/hip_utils.h"

namespace onnxruntime {
namespace hip {

template <bool is_grad>
Status SliceImplEx(const size_t element_size,
                   const int32_t dimension_count,
                   const TArray<int64_t>& starts,
                   const TArray<int64_t>& steps,
                   const TArray<int64_t>& input_strides,
                   const TArray<fast_divmod>& output_strides,
                   const void* input_data,
                   void* output_data,
                   const size_t N);

Status SliceImpl(const size_t element_size,
                 const int32_t dimension_count,
                 const TArray<int64_t>& starts,
                 const TArray<int64_t>& steps,
                 const TArray<int64_t>& input_strides,
                 const TArray<fast_divmod>& output_strides,
                 const void* input_data,
                 void* output_data,
                 const size_t N);

Status SliceImplGrad(const size_t element_size,
                     const int32_t dimension_count,
                     const TArray<int64_t>& starts,
                     const TArray<int64_t>& steps,
                     const TArray<int64_t>& input_strides,
                     const TArray<fast_divmod>& output_strides,
                     const void* input_data,
                     void* output_data,
                     const size_t N);
}  // namespace hip
}  // namespace onnxruntime
