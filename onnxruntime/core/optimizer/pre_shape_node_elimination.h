#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@class PreShapeNodeElimination

Rewrite rule that eliminates some particular nodes if the next node is a Shape node.

It is attempted to be triggered only on nodes with op type "Cast".
*/
class PreShapeNodeElimination : public RewriteRule {
 public:
  PreShapeNodeElimination() noexcept : RewriteRule("PreShapeNodeElimination") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Cast", "Transpose"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
