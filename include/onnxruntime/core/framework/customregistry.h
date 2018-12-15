// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/graph/schema_registry.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

/**
   Represents a registry that contains both custom kernels and custom schemas.
*/
class CustomRegistry : public KernelRegistry, public onnxruntime::OnnxRuntimeOpSchemaRegistry {
 public:
  CustomRegistry() = default;
  ~CustomRegistry() override = default;

  /**
   * Register a kernel definition together with kernel factory method to this session.
   * If any conflict happened between registered kernel def and built-in kernel def,
   * registered kernel will have higher priority.
   * Call this before invoking Initialize().
   * @return OK if success.
   */
  common::Status RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, const KernelCreateFn& kernel_creator);

  common::Status RegisterCustomKernel(KernelCreateInfo&);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomRegistry);
};

}  // namespace onnxruntime
