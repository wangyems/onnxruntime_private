
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <atomic>
#include "core/framework/onnx_object.h"
#include "core/framework/onnx_object_cxx.h"

/**
 * Configuration information for a single Run.
 */
struct ONNXRuntimeRunOptions : public onnxruntime::ObjectBase<ONNXRuntimeRunOptions> {
  unsigned run_log_verbosity_level = 0;  ///< applies to a particular Run() invocation
  std::string run_tag;                   ///< to identify logs generated by a particular Run() invocation

  /// set to 'true' to terminate any currently executing Run() calls that are using this
  /// ONNXRuntimeRunOptions instance. the individual calls will exit gracefully and return an error status.
  bool terminate = false;
  ONNXRuntimeRunOptions() = default;
  ~ONNXRuntimeRunOptions() = default;

  // disable copy, move and assignment. we don't want accidental copies, to ensure that the instance provided to
  // the Run() call never changes and the terminate mechanism will work.
  ONNXRuntimeRunOptions(const ONNXRuntimeRunOptions&) = delete;
  ONNXRuntimeRunOptions(ONNXRuntimeRunOptions&&) = delete;
  ONNXRuntimeRunOptions& operator=(const ONNXRuntimeRunOptions&) = delete;
  ONNXRuntimeRunOptions& operator=(ONNXRuntimeRunOptions&&) = delete;
};

namespace onnxruntime {
using RunOptions = ONNXRuntimeRunOptions;
}
