// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlReluGrad {
 public:
  enum InputTensors : int {
    IN_dY = 0,
    IN_X = 1
  };

  enum OutputTensors : int {
    OUT_dX = 0
  };

  DnnlReluGrad();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
