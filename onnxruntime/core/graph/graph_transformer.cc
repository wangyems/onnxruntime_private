// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_transformer.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

Status RuleBasedGraphTransformer::Register(const std::string& op_type, std::unique_ptr<RewriteRule> rule) {
  if (HasRules(op_type)) {
    op_to_rules_[op_type] = std::vector<std::unique_ptr<RewriteRule>>();
  }

  op_to_rules_[op_type].push_back(std::move(rule));
  return Status::OK();
}

Status TopDownRuleBasedTransformer::Apply(Graph& graph, bool& modified) const {
  ONNXRUNTIME_RETURN_IF_ERROR(graph.Resolve());
  const std::vector<NodeIndex>* order;
  ONNXRUNTIME_RETURN_IF_ERROR(graph.GetNodesInTopologicalOrder(order));
  assert(order);

  GraphEditor graph_editor(graph);

  for (NodeIndex i : *order) {
    auto node = graph.GetNode(i);
    if (!node) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT);
    }

    // Get the rules that should be fired for this node.
    if (!HasRules(node->OpType())) {
      continue;
    }
    const std::vector<std::unique_ptr<RewriteRule>>& rules = GetRewriteRules(node->OpType());
    for (const auto& rule : rules) {
      ONNXRUNTIME_RETURN_IF_ERROR(rule->CheckConditionAndApply(&graph_editor, node, &modified));
    }
  }

  // Resolve the graph at the end of all passes.
  if (modified) {
    ONNXRUNTIME_RETURN_IF_ERROR(graph.Resolve());
  }

  return Status::OK();
}

}  // namespace onnxruntime
