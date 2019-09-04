// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that insert an addtional output to the matched node.
class InsertMaxPoolOutput : public RewriteRule {
 public:
  InsertMaxPoolOutput() noexcept
      : RewriteRule("InsertMaxPoolOutput") {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const override;
};

// Rewrite rule that adjust Batch Normalization nodes to have 5 outputs for training mode
// instead of 1 for inference mode
class AdjustBatchNormOutputs : public RewriteRule {
 public:
  AdjustBatchNormOutputs() noexcept
      : RewriteRule("AdjustBatchNormOutputs") {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"BatchNormalization"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const override;
};
}  // namespace onnxruntime
