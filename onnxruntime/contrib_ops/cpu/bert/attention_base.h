// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class AttentionBase {
 protected:
  AttentionBase(const OpKernelInfo& info);

  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const Tensor*& mask_index,  // For dummy mask with shape (1, 1) or (batch_size, 1), it will be updated to nullptr.
                     const Tensor* past) const;

  Tensor* GetPresent(OpKernelContext* context,
                     const Tensor* past,
                     int batch_size,
                     int head_size,
                     int sequence_length,
                     int& past_sequence_length) const;

  int num_heads_;             // number of attention heads
  int head_size_;             // size of each attention head
  bool is_unidirectional_;    // whether every token can only attend to previous tokens.
  bool is_input_dim_swapped_; // whether the input_shape is (S, B, NH) instead of (B, S, NH)
};

}  // namespace contrib
}  // namespace onnxruntime
