// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/training/generic_registry.h"
#include "core/graph/training/graph_augmenter.h"

namespace onnxruntime {
namespace training {
namespace in_graph_optimizer {

struct OptimizerInfo {
  std::string name_;
  std::vector<std::string> params_;
};

class OptimizerBuilder {
 public:
  OptimizerBuilder(const std::string name) : name_(name) {}

  virtual ~OptimizerBuilder() {}

  virtual common::Status Build(const std::vector<std::string>& weights,
                               const std::vector<std::string>& gradients,
                               const OptimizerInfo& opt_info,
                               GraphAugmenter::GraphDefs& graph_defs) const = 0;

  const std::string& Name() const { return name_; }

 private:
  std::string name_;
};

class OptimizerBuilderRegistry : public GenericRegistry<OptimizerBuilder> {
 public:
  // Register optimizer builders.
  void RegisterBuilders();

  static OptimizerBuilderRegistry& GetInstance() {
    static OptimizerBuilderRegistry instance;
    return instance;
  }

 private:
  OptimizerBuilderRegistry() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerBuilderRegistry);
};

}  // namespace in_graph_optimizer
}  // namespace training
}  // namespace onnxruntime
