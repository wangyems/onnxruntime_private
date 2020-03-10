// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <inference_engine.hpp>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace openvino_ep {

class IBackend{
  public:
  virtual void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) = 0;
};

class BackendFactory {
  public:

  static std::shared_ptr<IBackend>
  MakeBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
              const std::vector<int>& input_indexes,
              const std::unordered_map<std::string, int>& output_names,
              std::string type, InferenceEngine::Precision precision);
};

} // namespace openvino_ep
} // namespace onnxruntime
