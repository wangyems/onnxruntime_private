// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <gsl/gsl>

#include "api.h"

namespace onnx_layout_transformation {

// Struct containing information for a handler functions. Decreases binary size and allows perm_inv to be precomputed.
struct HandlerArgs {
  int64_t opset;
  api::Graph& graph;
  api::Node& transpose;
  api::Node& node;
  const std::vector<int64_t>& perm;
  const std::vector<int64_t>& perm_inv;
  /// <summary>
  /// Allows handlers to selectively optimize transposes attached to a specific node input
  /// </summary>
  size_t transpose_input_index;
  bool skip_cost_check;
};

typedef bool HandlerFunction(HandlerArgs& args);


/////// <Helper Utils> ///////
/* Small utilities for editing nodes and manipulating axes/permutations */

// Replaces all node inputs referencing old_value with references to new_value. Values must be non-empty strings.
// This is an alternative to using MoveOutput for cases when the values aren't node outputs (if one is an initializer,
// for example).
static void ReplaceValueReferences(std::vector<std::unique_ptr<api::Node>>& nodes,
                                   const std::string_view old_value, const std::string_view new_value) {
  for (std::unique_ptr<api::Node>& node : nodes) {
    const std::vector<std::string_view>& inputs = node->Inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == old_value) {
        node->SetInput(i, new_value);
      }
    }
  }
}

// Create a node with a single attribute of type vector<int64_t>
static std::unique_ptr<api::Node> MakeNode1Attr(api::Graph& graph, const std::string_view op_type,
                                                const std::string_view input, const std::string_view attr_name,
                                                const std::vector<int64_t>& attr_val) {
  std::vector<std::string_view> inputs;
  inputs.push_back(input);
  std::unique_ptr<api::Node> node = graph.AddNode(op_type, inputs);
  node->SetAttributeInts(attr_name, attr_val);
  return node;
}

// Creates a Transpose node. Does not update output ValueInfo.
static std::unique_ptr<api::Node> MakeTranspose(api::Graph& graph, const std::string_view input,
                                                const std::vector<int64_t>& perm) {
  return MakeNode1Attr(graph, "Transpose", input, "perm", perm);
}

// Creates a Squeeze/Unsqueeze node. Does not update output ValueInfo.
static std::unique_ptr<api::Node> MakeSqueezeOrUnsqueeze(int64_t opset, api::Graph& graph, 
                                                         const std::string_view op_type, const std::string_view input,
                                                         const std::vector<int64_t>& axes) {
  if (opset < 13) {
    return MakeNode1Attr(graph, op_type, input, "axes", axes);
  }
  std::vector<int64_t> axes_shape;
  axes_shape.push_back(axes.size());
  const std::string_view axes_initializer = graph.AddInitializerInt64(axes_shape, axes);
  std::vector<std::string_view> inputs;
  inputs.push_back(input);
  inputs.push_back(axes_initializer);
  return graph.AddNode(op_type, inputs);
}

// Returns whether perm is a valid permutation (contains each value from 0 to perm.size() - 1 exactly once)
bool IsValidPerm(const std::vector<int64_t>& perm) {
  size_t rank = perm.size();
  int64_t rank_int = gsl::narrow_cast<int64_t>(rank);
  std::vector<bool> used_dims(rank);
  for (size_t i = 0; i < rank; ++i) {
    int64_t x = perm[i];
    size_t x_size_t = gsl::narrow_cast<size_t>(x);
    if (x < 0 || x >= rank_int || used_dims[x_size_t]) {
      return false;
    }
    used_dims[x_size_t] = true;
  }
  return true;
}

// Computes inverse permutation. Unsafe if perm is not a valid permutation.
static std::vector<int64_t> InvertPerm(const std::vector<int64_t>& perm) {
  size_t rank = perm.size();
  auto perm_inv = std::vector<int64_t>(rank);
  for (size_t i = 0; i < rank; ++i) {
    size_t j = gsl::narrow_cast<size_t>(perm[i]);
    perm_inv[j] = gsl::narrow_cast<int64_t>(i);
  }
  return perm_inv;
}

// Computes composition of perm1 and perm2. Unsafe if perm1 or perm2 are not valid permutations.
static std::vector<int64_t> ComposePerm(const std::vector<int64_t>& perm1, const std::vector<int64_t>& perm2) {
  std::vector<int64_t> perm;
  for (int64_t p : perm2) {
    perm.push_back(perm1[gsl::narrow_cast<size_t>(p)]);
  }
  return perm;
}

// Computes permutation from channel last to channel first ordering of given rank. Nearly all handlers work for any
// permutation, but some are restricted. Also used for layout transformation. Rank must be >= 1.
static std::vector<int64_t> ChannelLastToFirstPerm(size_t rank) {
  std::vector<int64_t> p(rank);
  p[0] = 0;
  p[1] = rank - 1;
  for (size_t i = 2; i < rank; ++i) {
    p[i] = i - 1;
  }
  return p;
}

// Adds 1 dimensions to indices of shape corresponding to axes. Unsafe if axes has negative/duplicated entries.
static std::vector<int64_t> UnsqueezeShape(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes) {
  size_t new_rank = shape.size() + axes.size();
  auto new_shape = std::vector<int64_t>(new_rank);
  for (int64_t a : axes) {
    // Fill unsqueezed axes with 1s
    new_shape[gsl::narrow_cast<size_t>(a)] = 1;
  }
  size_t j = 0;
  for (size_t i = 0; i < new_rank; i++) {
    // Fill remaining axes with existing shape. Skip prefilled 1s.
    if (new_shape[i] != 1) {
      new_shape[i] = shape[j];
      ++j;
    }
  }
  return new_shape;
}

// Computes new perm for unsqueezed version of a tensor. Unsafe if axes/perm is not valid.
// New perm reorders non-1 dimensions in the same way and leaves 1-dims from unsqueeze unchanged.
// Ex:
// perm = [2, 0, 1] means shape [A, B, C] -> [C, A, B]. If axes = [0, 3], map to
// result = [0, 4, 1, 3, 2] means shape [1, A, B, 1, C] -> [1, C, A, 1, B]
static std::vector<int64_t> UnsqueezePerm(const std::vector<int64_t>& axes, const std::vector<int64_t>& perm) {
  std::vector<int64_t> new_perm;
  size_t old_rank = perm.size();
  size_t new_rank = old_rank + axes.size();
  auto axes_bit_map = std::vector<bool>(new_rank);
  for (int64_t a : axes) {
    // TODO: fix negatives before this
    if (a < 0) {
      a += new_rank;
    }
    axes_bit_map[gsl::narrow_cast<size_t>(a)] = true;
  }
  std::vector<int64_t> axes_map;  // maps old axes to new (unsqueezed) axes
  for (size_t i = 0; i < new_rank; ++i) {
    if (!axes_bit_map[i]) {
      axes_map.push_back(i);
    }
  }
  size_t j = 0;
  for (size_t i = 0; i < new_rank; ++i) {
    if (axes_bit_map[i]) {
      // Leave 1s in the same place
      new_perm.push_back(i);
    } else {
      // Take next axis from perm
      size_t perm_axis = gsl::narrow_cast<size_t>(perm[j++]);
      new_perm.push_back(axes_map[perm_axis]);
    }
  }
  return new_perm;
}

/////// </Helper Utils> ///////

/////// <Core Helpers> ///////
/* These helpers hide the most gnarly parts of the transpose optimizer. */

static const std::string_view HelpHandleUnsqueeze(HandlerArgs& args, std::vector<int64_t> axes);

static const std::string_view UnsqueezeValue(int64_t opset, api::Graph& graph, const std::string_view input, const std::vector<int64_t>& axes) {
  std::unique_ptr<api::Tensor> constant = graph.GetConstant(input);
  auto consumers = graph.GetValueConsumers(input);

  if (constant != nullptr && consumers->comprehensive) {
    if (consumers->nodes.size() > 0) {
      auto squeeze_ptr = MakeSqueezeOrUnsqueeze(opset, graph, "Squeeze", input, axes);
      api::Node& squeeze = *squeeze_ptr;
      const std::string_view sq_out = squeeze.Outputs()[0];
      graph.CopyValueInfo(input, sq_out);
      ReplaceValueReferences(consumers->nodes, input, sq_out);
    }
    auto new_shape = UnsqueezeShape(constant->Shape(), axes);
    graph.ReshapeInitializer(input, new_shape);
    return input;
  }
  std::unique_ptr<api::Node> node = graph.GetNodeProducingOutput(input);
  if (node != nullptr && node->IsOp("Squeeze")) {
    const std::vector<std::string_view>& inputs = node->Inputs();
    std::optional<std::vector<int64_t>> squeeze_axes = std::nullopt;
    if (*graph.Opset() < 13) {
      squeeze_axes = node->GetAttributeInts("axes");
    } else if (inputs.size() == 2) {
      std::unique_ptr<api::Tensor> axes_const = graph.GetConstant(inputs[1]);
      if (axes_const != nullptr) {
        squeeze_axes = axes_const->DataInt64();
      }
    }
    if (squeeze_axes != std::nullopt && *squeeze_axes == axes) {
      if (consumers->comprehensive && consumers->nodes.size() == 0) {
        graph.RemoveNode(*node);
        if (*graph.Opset() >= 13 && !graph.HasValueConsumers(inputs[1])) {
          graph.RemoveInitializer(inputs[1]);
        }
      }
      return inputs[0];
    }
  }
  auto squeeze_ptr = MakeSqueezeOrUnsqueeze(opset, graph, "Unsqueeze", input, axes);
  api::Node& squeeze = *squeeze_ptr;
  const std::string_view sq_out = squeeze.Outputs()[0];
  graph.CopyValueInfo(input, sq_out);
  graph.GetValueInfo(sq_out)->UnsqueezeDims(axes);
  if (node != nullptr && node->IsOp("Transpose")) {
    auto perm = node->GetAttributeInts("perm");
    if (perm != std::nullopt) {
      auto perm_inv = InvertPerm(*perm);
      HandlerArgs args{opset, graph, *node, squeeze, *perm, perm_inv, 0, false};
      return HelpHandleUnsqueeze(args, axes);
    }
  }
  return sq_out;
}

static const std::string_view TransposeValue(api::Graph& graph, const std::string_view input, const std::vector<int64_t>& perm, const std::vector<int64_t>& perm_inv) {
  // 3 cases: Transpose, Const, other.

  // TODO: update shape?
  std::unique_ptr<api::Tensor> constant = graph.GetConstant(input);
  auto consumers = graph.GetValueConsumers(input);

  if (constant != nullptr && consumers->comprehensive) {
    if (consumers->nodes.size() > 0) {
      auto transpose_inv_ptr = MakeTranspose(graph, input, perm_inv);
      api::Node& transpose_inv = *transpose_inv_ptr;
      const std::string_view transpose_out = transpose_inv.Outputs()[0];
      graph.CopyValueInfo(input, transpose_out);
      ReplaceValueReferences(consumers->nodes, input, transpose_out);
    }
    graph.TransposeInitializer(input, perm);
    return input;
  }
  std::unique_ptr<api::Node> node = graph.GetNodeProducingOutput(input);
  if (node != nullptr && node->IsOp("Transpose")) {
    std::optional<std::vector<int64_t>> perm2 = node->GetAttributeInts("perm");
    if (perm2 != std::nullopt) {
      if (*perm2 == perm_inv) {
        std::string_view pre_transpose_value = node->Inputs()[0];
        if (consumers->comprehensive && consumers->nodes.size() == 0) {
          graph.RemoveNode(*node);
        }
        return pre_transpose_value;
      }
      const std::vector<int64_t>& perm_combined = ComposePerm(*perm2, perm);
      auto transpose_ptr = MakeTranspose(graph, node->Inputs()[0], perm_combined);
      api::Node& transpose = *transpose_ptr;
      const std::string_view transpose_out = transpose.Outputs()[0];
      graph.CopyValueInfo(input, transpose_out);
      graph.GetValueInfo(transpose_out)->PermuteDims(perm);
      if (consumers->comprehensive && consumers->nodes.size() == 0) {
        graph.RemoveNode(*node);
      }
      return transpose_out;
    }
  }
  for (size_t i = 0; i < consumers->nodes.size(); ++i) {
    api::Node& consumer = *consumers->nodes[i];
    if (consumer.IsOp("Transpose") && consumer.GetAttributeInts("perm") == perm) {
      return consumer.Outputs()[0];
    }
  }
  auto transpose_ptr = MakeTranspose(graph, input, perm);
  api::Node& transpose = *transpose_ptr;
  const std::string_view transpose_out = transpose.Outputs()[0];
  graph.CopyValueInfo(input, transpose_out);
  graph.GetValueInfo(transpose_out)->PermuteDims(perm);
  return transpose_out;
}

static bool NormalizeInputRanks(int64_t opset, api::Graph& graph, api::Node& node, size_t rank, std::vector<size_t>* indices = nullptr) {
  auto inputs = node.Inputs();
  std::unique_ptr<std::vector<size_t>> indices_storage;
  if (indices == nullptr) {
    size_t num_inputs = inputs.size();
    indices_storage = std::make_unique<std::vector<size_t>>(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      (*indices_storage)[i] = i;
    }
    indices = &(*indices_storage);
  }
  for (size_t i : *indices) {
    std::optional<std::vector<int64_t>> shape = graph.GetValueInfo(inputs[i])->Shape();
    if (shape == std::nullopt || shape->size() > rank) {
      return false;
    }
  }
  for (size_t i : *indices) {
    std::optional<std::vector<int64_t>> shape = graph.GetValueInfo(inputs[i])->Shape();
    size_t rank_diff = rank - shape->size();
    if (rank_diff > 0) {
      std::vector<int64_t> axes;
      for (size_t j = 0; j < rank_diff; ++j) {
        axes.push_back(j);
      }
      node.SetInput(i, "");
      const std::string_view unsq_out = UnsqueezeValue(opset, graph, inputs[i], axes);
      node.SetInput(i, unsq_out);
      inputs = node.Inputs();
    }
  }
  return true;
}

static void TransposeInputs(api::Graph& graph, api::Node& node, const std::vector<int64_t>& perm, std::vector<size_t>* indices = nullptr) {
  std::unique_ptr<std::vector<size_t>> indices_storage;
  if (indices == nullptr) {
    size_t num_inputs = node.Inputs().size();
    indices_storage = std::make_unique<std::vector<size_t>>(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      (*indices_storage)[i] = i;
    }
    indices = &(*indices_storage);
  }
  auto perm_inv = InvertPerm(perm);
  for (size_t j : *indices) {
    // X -> Node
    std::string_view inp = node.Inputs()[j];
    node.SetInput(j, "");  // "" -> Node
    const std::string_view trans1_out = TransposeValue(graph, inp, perm, perm_inv);  // X -> T, "" -> Node
    node.SetInput(j, trans1_out);  // X -> T -> Node
  }
  return;
}

inline static void TransposeFirstInput(api::Graph& graph, api::Node& node, const std::vector<int64_t>& perm) {
  std::vector<size_t> indices {0};
  TransposeInputs(graph, node, perm, &indices);
}

static int EstimateValueRank(api::Graph& graph, const std::string_view input) {
  auto value_info = graph.GetValueInfo(input);
  std::optional<std::vector<int64_t>> shape = value_info->Shape();
  if (shape == std::nullopt) {
    return 5;
  }
  int rank = 0;
  for (int64_t d : *shape) {
    if (d != 1) {
      ++rank;
    }
  }
  return rank;
}

static HandlerFunction* GetHandler(api::Node& node, bool allow_extended_ops);

static bool CanLikelyRemoveTranspose(api::Graph& graph, api::Node& transpose) {
  auto consumers = graph.GetValueConsumers(transpose.Outputs()[0]);
  if (!consumers->comprehensive) {
    return false;
  }
  for (auto& node : consumers->nodes) {
    if (GetHandler(*node, true) == nullptr) {
      return false;
    }
  }
  return true;
}

static int EstimateTransposeValueCost(api::Graph& graph, const std::string_view input, const std::vector<int64_t>& perm_inv) {
  std::unique_ptr<api::Tensor> constant = graph.GetConstant(input);
  if (constant != nullptr) {
    return 0;
  }
  std::unique_ptr<api::Node> node = graph.GetNodeProducingOutput(input);
  if (node != nullptr && node->IsOp("Transpose")) {
    std::optional<std::vector<int64_t>> perm2 = node->GetAttributeInts("perm");
    if (perm2 != std::nullopt) {
      if (*perm2 == perm_inv && CanLikelyRemoveTranspose(graph, *node)) {
        return -EstimateValueRank(graph, input);
      } else {
        return 0;
      }
    }
  }
  return EstimateValueRank(graph, input);
}

static int EstimateTransposeInputsCost(api::Graph& graph, api::Node& node, const std::vector<int64_t>& perm_inv, std::vector<size_t>* indices = nullptr) {
  auto inputs = node.Inputs();
  std::unique_ptr<std::vector<size_t>> indices_storage;
  if (indices == nullptr) {
    size_t num_inputs = inputs.size();
    indices_storage = std::make_unique<std::vector<size_t>>(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      (*indices_storage)[i] = i;
    }
    indices = &(*indices_storage);
  }
  int cost = 0;
  for (size_t j : *indices) {
    cost += EstimateTransposeValueCost(graph, inputs[j], perm_inv);
  }
  return cost;
}

static bool IsIdentityPerm(const std::vector<int64_t>& perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != (int64_t)i) {
      return false;
    }
  }
  return true;
}

static const std::string_view TransposeOutput(api::Graph& graph, api::Node& node, size_t i, const std::vector<int64_t>& perm, const std::vector<int64_t>& perm_inv) {
  // Make transpose without input, then add it to avoid cyclic reference.
  auto transpose_ptr = MakeTranspose(graph, "", perm);
  api::Node& transpose = *transpose_ptr;
  graph.MoveOutput(node, i, transpose, 0);
  const std::string_view new_output = node.Outputs()[i];
  transpose.SetInput(0, new_output);
  const std::string_view old_output = transpose.Outputs()[0];
  graph.CopyValueInfo(old_output, new_output);
  graph.GetValueInfo(new_output)->PermuteDims(perm_inv);
  return old_output;
}

static void TransposeOutputs(api::Graph& graph, api::Node& node, const std::vector<int64_t>& perm) {
  if (IsIdentityPerm(perm)) {
    return;
  }
  auto perm_inv = InvertPerm(perm);
  for (size_t j = 0; j < node.Outputs().size(); ++j) {
    TransposeOutput(graph, node, j, perm, perm_inv);
  }
}

static bool HandleSimpleNodeBase(HandlerArgs& args, bool broadcast, std::vector<size_t>* indices = nullptr) {
  // indices must be null if broadcast is true
  size_t rank = args.perm.size();
  if (!args.skip_cost_check && (indices == nullptr || indices->size() > 1) 
      && EstimateTransposeInputsCost(args.graph, args.node, args.perm, indices) >= 0) {
    return false;
  }
  if (broadcast && !NormalizeInputRanks(args.opset, args.graph, args.node, rank, indices)) {
    return false;
  }
  TransposeInputs(args.graph, args.node, args.perm_inv, indices);
  TransposeOutputs(args.graph, args.node, args.perm);
  return true;
}

static bool HandleSimpleNodeBroadcast(HandlerArgs& args) {
  return HandleSimpleNodeBase(args, /*broadcast*/ true);
}

static bool HandleSimpleNode(HandlerArgs& args) {
  return HandleSimpleNodeBase(args, /*broadcast*/ false);
}

static bool HandleSimpleNode1Inp(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  std::vector<size_t> indices {0};
  return HandleSimpleNodeBase(args, /*broadcast*/ false, &indices);
}

static bool HandleSimpleNodeAxis(HandlerArgs& args, bool has_default, int64_t default_axis=0) {
  size_t rank = args.perm.size();
  std::optional<int64_t> axis = args.node.GetAttributeInt("axis");
  if (axis == std::nullopt) {
    if (has_default) {
      axis = default_axis;
    } else {
      return false;
    }
  }
  if (*axis < 0) {
    *axis += rank;
  }
  if (*axis < 0 || (uint64_t)*axis >= args.perm.size()) return false;
  if (!HandleSimpleNodeBase(args, /*broadcast*/ false)) {
    return false;
  }
  args.node.SetAttributeInt("axis", args.perm[(size_t)*axis]);
  return true;
}

static bool HandleSplit(HandlerArgs& args) {
  return HandleSimpleNodeAxis(args, /*has_default*/ true, /*default_axis*/ 0);
}

static bool HandleConcat(HandlerArgs& args) {
  return HandleSimpleNodeAxis(args, /*has_default*/ false);
}

static bool HandleSoftHardMax(HandlerArgs& args) {
  // TODO: add rank to args?
  size_t rank = args.perm.size();
  if (args.opset >= 13) {
    return HandleSimpleNodeAxis(args, /*has_default*/ true, /*default_axis*/ -1);
  }
  int64_t axis = 1;
  std::optional<int64_t> axis_attr = args.node.GetAttributeInt("axis");
  if (axis_attr != std::nullopt) {
    axis = *axis_attr;
  }
  // TODO: consolidate this?
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || (uint64_t)axis >= rank) return false;
  for (size_t i = 0; i < rank; ++i) {
    bool to_lhs = i < (uint64_t)axis;
    bool from_lhs = args.perm[i] < axis;
    if (to_lhs != from_lhs) {
      return false;
    }
  }
  return HandleSimpleNode(args);
}

static bool HandleShape(HandlerArgs& args) {
  TransposeInputs(args.graph, args.node, args.perm_inv);
  size_t rank = args.perm.size();

  std::vector<int64_t> new_perm;
  if (args.opset >= 15) {
    int64_t start = args.node.GetAttributeIntDefault("start", 0);
    int64_t end = args.node.GetAttributeIntDefault("end", (int64_t)rank);
    if (start < 0) {
      start += rank;
    }
    if (end < 0) {
      end += rank;
    }
    size_t start_idx = (size_t)std::clamp(start, (int64_t)0, (int64_t)rank);
    size_t end_idx = (size_t)std::clamp(end, (int64_t)0, (int64_t)rank);
    for (size_t i = start_idx; i < end_idx; ++i) {
      new_perm.push_back(args.perm[i]);
    }
    args.node.ClearAttribute("start");
    args.node.ClearAttribute("end");
  } else {
    new_perm = args.perm;
  }

  std::vector<int64_t> perm_shape {(int64_t)new_perm.size()};
  const std::string_view perm_const = args.graph.AddInitializerInt64(perm_shape, new_perm);

  std::vector<std::string_view> gather_inputs;
  gather_inputs.push_back("");
  gather_inputs.push_back(perm_const);
  auto gather_ptr = args.graph.AddNode("Gather", gather_inputs);
  api::Node& gather = *gather_ptr;
  gather.SetAttributeInt("axis", 0);
  args.graph.MoveOutput(args.node, 0, gather, 0);
  const std::string_view new_output = args.node.Outputs()[0];
  gather.SetInput(0, new_output);
  args.graph.CopyValueInfo(gather.Outputs()[0], new_output);
  if (new_perm.size() != rank) {
    auto info = args.graph.GetValueInfo(new_output);
    std::vector<int64_t> new_shape {(int64_t)rank};
    info->SetShape(&new_shape);
  }
  return true;
}

static std::vector<int64_t> PermutePads(const std::vector<int64_t>& pads, const std::vector<int64_t>& perm) {
  size_t rank = perm.size();
  std::vector<int64_t> new_pads;
  for (int64_t i : perm) {
    new_pads.push_back(pads[(size_t)i]);
  }
  for (int64_t i : perm) {
    new_pads.push_back(pads[(size_t)i + rank]);
  }
  return new_pads;
}

static bool HandlePad(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  size_t rank = args.perm.size();
  int64_t opset = args.opset;

  if (opset < 11) {
    std::optional<std::vector<int64_t>> pads = args.node.GetAttributeInts("pads");
    if (pads == std::nullopt) {
      return false;
    }
    std::vector<int64_t> new_pads = PermutePads(*pads, args.perm_inv);
    args.node.SetAttributeInts("pads", new_pads);
  }

  TransposeFirstInput(args.graph, args.node, args.perm_inv);
  TransposeOutputs(args.graph, args.node, args.perm);

  if (opset < 11) {
    return true;
  }

  std::string_view pads_input = args.node.Inputs()[1];
  std::vector<int64_t> pads_shape { (int64_t)rank * 2 };
  std::shared_ptr<api::Tensor> pads_const = args.graph.GetConstant(pads_input);
  if (pads_const != nullptr) {
    auto pads = pads_const->DataInt64();
    std::vector<int64_t> new_pads = PermutePads(pads, args.perm_inv);
    std::string_view new_pads_const = args.graph.AddInitializerInt64(pads_shape, new_pads);
    args.node.SetInput(1, new_pads_const);
    if (!args.graph.HasValueConsumers(pads_input)) {
      args.graph.RemoveInitializer(pads_input);
    }
    return true;
  }

  std::vector<int64_t> pads_perm = args.perm_inv;
  for (int64_t p : args.perm_inv) {
    pads_perm.push_back(p + rank);
  }
  std::string_view pads_perm_const = args.graph.AddInitializerInt64(pads_shape, pads_perm);

  std::vector<std::string_view> gather_inputs { pads_input, pads_perm_const };
  auto gather_ptr = args.graph.AddNode("Gather", gather_inputs);
  api::Node& gather = *gather_ptr;
  std::string_view gather_output = gather.Outputs()[0];
  args.graph.CopyValueInfo(pads_input, gather_output);
  gather.SetAttributeInt("axis", 0);
  args.node.SetInput(1, gather_output);

  return true;
}

static std::vector<int64_t> SqueezePerm(const std::vector<int64_t>& axes, const std::vector<int64_t>& perm) {
  std::vector<int64_t> axes_map;
  size_t j = 0;
  for (size_t i = 0; i < perm.size(); ++i) {
    bool removed = false;
    for (int64_t a : axes) {
      if (i == (size_t)a) {
        removed = true;
      }
    }
    if (removed) {
      axes_map.push_back(-1);
    } else {
      axes_map.push_back((int64_t)j++);
    }
  }
  std::vector<int64_t> new_perm;
  for (int64_t p : perm) {
    if (axes_map[(size_t)p] != -1) {
      new_perm.push_back(axes_map[(size_t)p]);
    }
  }
  return new_perm;
}

static std::vector<int64_t> PermuteAxes(const std::vector<int64_t>& axes, const std::vector<int64_t>& perm) {
  // axes may be negative
  // TODO: clarify perm vs perm_inv
  size_t rank = perm.size();
  // For sorting
  auto new_axes_bit_map = std::vector<bool>(perm.size());
  for (int64_t a : axes) {
    if (a < 0) {
      a += (int64_t)rank;
    }
    new_axes_bit_map[(size_t)perm[(size_t)a]] = true;
  }
  std::vector<int64_t> new_axes;
  for (size_t a = 0; a < rank; a++) {
    if (new_axes_bit_map[a]) {
      new_axes.push_back((int64_t)a);
    }
  }
  return new_axes;
}

static bool HandleReduceOp(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  // TODO: compress this impl

  int64_t keepdims = args.node.GetAttributeIntDefault("keepdims", 1);

  std::optional<std::vector<int64_t>> axes = args.node.GetAttributeInts("axes");
  // TODO: (compress impl) empty axes
  if (axes == std::nullopt) {
    if (keepdims != 0) {
      TransposeFirstInput(args.graph, args.node, args.perm_inv);
      TransposeOutputs(args.graph, args.node, args.perm);
    } else {
      TransposeFirstInput(args.graph, args.node, args.perm_inv);
    }
    return true;
  }
  std::vector<int64_t> new_axes = PermuteAxes(*axes, args.perm);
  args.node.SetAttributeInts("axes", new_axes);

  if (keepdims != 0) {
    TransposeFirstInput(args.graph, args.node, args.perm_inv);
    TransposeOutputs(args.graph, args.node, args.perm);
    return true;
  }
  else {
    TransposeFirstInput(args.graph, args.node, args.perm_inv);
    std::vector<int64_t> new_perm = SqueezePerm(new_axes, args.perm);
    TransposeOutputs(args.graph, args.node, new_perm);
    return true;
  }
}

static bool HandleReduceSum(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  // TODO: compress this impl

  if (args.opset < 13) {
    return HandleReduceOp(args);
  }

  int64_t keepdims = args.node.GetAttributeIntDefault("keepdims", 1);

  const std::vector<std::string_view>& inputs = args.node.Inputs();
  std::unique_ptr<api::Tensor> axes_const = nullptr;
  bool empty_axes = false;
  if (inputs.size() < 2 || inputs[1] == "") {
    empty_axes = true;
  } else {
    axes_const = args.graph.GetConstant(inputs[1]);
    if (axes_const != nullptr && axes_const->DataInt64().size() == 0) {
      empty_axes = true;
    }
  }
  if (empty_axes) {
    int64_t noop_with_empty_axes = args.node.GetAttributeIntDefault("noop_with_empty_axes", 0);
    if (noop_with_empty_axes != 0 || keepdims != 0) {
      TransposeFirstInput(args.graph, args.node, args.perm_inv);
      TransposeOutputs(args.graph, args.node, args.perm);
    } else {
      TransposeFirstInput(args.graph, args.node, args.perm_inv);
    }
    return true;
  }
  if (axes_const == nullptr) {
    // TODO: technically we can handle this with Gather if keepdims is true
    return false;
  }

  auto axes = axes_const->DataInt64();
  std::vector<int64_t> new_axes = PermuteAxes(axes, args.perm);
  std::vector<int64_t> axes_shape { (int64_t)new_axes.size() };
  std::string_view new_axes_const = args.graph.AddInitializerInt64(axes_shape, new_axes);
  std::string_view axes_inp = inputs[1];
  args.node.SetInput(1, new_axes_const);
  if (!args.graph.HasValueConsumers(axes_inp)) {
    args.graph.RemoveInitializer(axes_inp);
  }

  if (keepdims != 0) {
    TransposeFirstInput(args.graph, args.node, args.perm_inv);
    TransposeOutputs(args.graph, args.node, args.perm);
    return true;
  }
  else {
    TransposeFirstInput(args.graph, args.node, args.perm_inv);
    std::vector<int64_t> new_perm = SqueezePerm(new_axes, args.perm);
    TransposeOutputs(args.graph, args.node, new_perm);
    return true;
  }

  return true;
}

static bool HandleSqueeze(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  const std::vector<size_t> indices { 0 };
  std::vector<int64_t> new_axes;
  if (args.opset < 13) {
    std::optional<std::vector<int64_t>> axes = args.node.GetAttributeInts("axes");
    // TODO: (compress impl) empty axes
    if (axes == std::nullopt) {
      return false;
    }
    new_axes = PermuteAxes(*axes, args.perm);
    args.node.SetAttributeInts("axes", new_axes);
  } else {
    const std::vector<std::string_view>& inputs = args.node.Inputs();
    if (inputs.size() < 2) {
      return false;
    }
    std::string_view axes_inp = inputs[1];
    if (axes_inp == "") {
      return false;
    }
    std::unique_ptr<api::Tensor> axes_const = args.graph.GetConstant(axes_inp);
    if (axes_const == nullptr) {
      return false;
    }
    auto axes = axes_const->DataInt64();
    new_axes = PermuteAxes(axes, args.perm);
    std::vector<int64_t> axes_shape { (int64_t)new_axes.size() };
    std::string_view new_axes_const = args.graph.AddInitializerInt64(axes_shape, new_axes);
    args.node.SetInput(1, new_axes_const);
    if (!args.graph.HasValueConsumers(axes_inp)) {
      args.graph.RemoveInitializer(axes_inp);
    }
  }
  TransposeFirstInput(args.graph, args.node, args.perm_inv);
  std::vector<int64_t> new_perm = SqueezePerm(new_axes, args.perm);
  TransposeOutputs(args.graph, args.node, new_perm);
  return true;
}


static const std::string_view HelpHandleUnsqueeze(HandlerArgs& args, std::vector<int64_t> axes) {
  TransposeFirstInput(args.graph, args.node, args.perm_inv);
  std::vector<int64_t> new_perm = UnsqueezePerm(axes, args.perm);
  return TransposeOutput(args.graph, args.node, 0, new_perm, InvertPerm(new_perm));
}

static bool HandleUnsqueeze(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  std::vector<int64_t> axes;
  if (args.opset < 13) {
    std::optional<std::vector<int64_t>> axes_attr = args.node.GetAttributeInts("axes");
    // TODO: (compress impl) empty axes
    if (axes_attr == std::nullopt) {
      return false;
    }
    axes = *axes_attr;
  } else {
    const std::vector<std::string_view>& inputs = args.node.Inputs();
    std::unique_ptr<api::Tensor> axes_const = args.graph.GetConstant(inputs[1]);
    if (axes_const == nullptr) {
      return false;
    }
    axes = axes_const->DataInt64();
  }
  HelpHandleUnsqueeze(args, axes);
  return true;
}


static bool HandleQuantizeDequantizeLinear(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  size_t rank = args.perm.size();

  if (args.opset >= 13) {
    auto inputs = args.node.Inputs();
    bool all_scalars = true;
    for (size_t i = 1; i < 3; ++i) {
      std::optional<std::vector<int64_t>> inp_shape = args.graph.GetValueInfo(inputs[i])->Shape();
      if (inp_shape == std::nullopt || inp_shape->size() > 0) {
        all_scalars = false;
      }
    }
    if (!all_scalars) {
      int64_t axis = args.node.GetAttributeIntDefault("axis", 1);
      if (axis < 0) {
        axis += rank;
      }
      if (axis < 0 || (size_t)axis >= args.perm.size()) {
        return false;
      }
      args.node.SetAttributeInt("axis", args.perm[(size_t)axis]);
    }
  }

  TransposeFirstInput(args.graph, args.node, args.perm_inv);
  TransposeOutputs(args.graph, args.node, args.perm);

  return true;
}

static bool HandleArgMinMax(HandlerArgs& args) {
  size_t rank = args.perm.size();

  int64_t keepdims = args.node.GetAttributeIntDefault("keepdims", 1);
  int64_t axis = args.node.GetAttributeIntDefault("axis", 0);
  if (axis < 0) {
    axis += (int64_t)rank;
  }
  int64_t new_axis = args.perm[(size_t)axis];
  std::vector<int64_t> new_axes {new_axis};
  args.node.SetAttributeInt("axis", new_axis);

  TransposeInputs(args.graph, args.node, args.perm_inv);
  if (keepdims != 0) {
    TransposeOutputs(args.graph, args.node, args.perm);
  } else {
    TransposeOutputs(args.graph, args.node, SqueezePerm(new_axes, args.perm));
  }
  return true;
}

static bool HandleSlice(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  size_t rank = args.perm.size();

  if (args.opset < 10) {
    std::optional<std::vector<int64_t>> axes = args.node.GetAttributeInts("axes");
    if (axes == std::nullopt) {
      std::optional<std::vector<int64_t>> starts = args.node.GetAttributeInts("starts");
      if (starts == std::nullopt) {
        // Invalid model. TODO: raise exception
        return false;
      }
      size_t num_starts = starts->size();
      axes = std::vector<int64_t>();
      axes->reserve(num_starts);
      for (size_t i = 0; i < num_starts; ++i) {
        axes->push_back(i);
      }
    }
    std::vector<int64_t> new_axes;
    for (int64_t a : *axes) {
       // TODO: consolidate this
      if (a < 0) {
        a += rank;
      }
      if (a < 0 || (size_t)a >= rank) {
        return false;
      }
      new_axes.push_back(args.perm[(size_t)a]);
    }
    args.node.SetAttributeInts("axes", new_axes);
    TransposeFirstInput(args.graph, args.node, args.perm_inv);
    TransposeOutputs(args.graph, args.node, args.perm);
    return true;
  }

  std::vector<std::string_view> inputs = args.node.Inputs();
  if (inputs.size() < 3) {
    return false;
  }
  std::vector<int64_t> new_axes;
  if (inputs.size() < 4 || inputs[3] == "") {
    const std::optional<std::vector<int64_t>> starts_shape = args.graph.GetValueInfo(inputs[1])->Shape();
    if (starts_shape == std::nullopt || starts_shape->size() != 1 || (*starts_shape)[0] < 0) {
      return false;
    }
    size_t ndims = (size_t)(*starts_shape)[0];
    for (size_t i = 0; i < ndims; ++i) {
      new_axes.push_back(args.perm[i]);
    }
    std::vector<int64_t> axes_shape { (int64_t)new_axes.size() };
    std::string_view new_axes_const = args.graph.AddInitializerInt64(axes_shape, new_axes);
    if (inputs.size() == 3) {
      args.node.AddInput(new_axes_const);
    } else {
      args.node.SetInput(3, new_axes_const);
    }
  } else {
    std::string_view axes_inp = inputs[3];
    std::unique_ptr<api::Tensor> axes_const = args.graph.GetConstant(axes_inp);
    if (axes_const == nullptr) {
      return false;
    }
    auto axes = axes_const->DataInt64();
    for (int64_t a : axes) {
      // TODO: consolidate this
      if (a < 0) {
        a += rank;
      }
      if (a < 0 || (size_t)a >= rank) {
        return false;
      }
      new_axes.push_back(args.perm[(size_t)a]);
    }
    std::vector<int64_t> axes_shape { (int64_t)new_axes.size() };
    std::string_view new_axes_const = args.graph.AddInitializerInt64(axes_shape, new_axes);
    args.node.SetInput(3, new_axes_const);
    if (!args.graph.HasValueConsumers(axes_inp)) {
      args.graph.RemoveInitializer(axes_inp);
    }
  }
  TransposeFirstInput(args.graph, args.node, args.perm_inv);
  TransposeOutputs(args.graph, args.node, args.perm);
  return true;
}

static bool HandleTile(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  size_t rank = args.perm.size();
  std::vector<int64_t> perm_shape {(int64_t)rank};

  std::string_view repeats_inp = args.node.Inputs()[1];
  std::unique_ptr<api::Tensor> repeats_const = args.graph.GetConstant(repeats_inp);
  if (repeats_const != nullptr) {
    const std::vector<int64_t>& repeats = repeats_const->DataInt64();
    std::vector<int64_t> new_repeats;
    for (int64_t p : args.perm_inv) {
      new_repeats.push_back(repeats[(size_t)p]);
    }
    std::string_view new_repeats_const = args.graph.AddInitializerInt64(perm_shape, new_repeats);
    args.node.SetInput(1, new_repeats_const);
    if (!args.graph.HasValueConsumers(repeats_inp)) {
      args.graph.RemoveInitializer(repeats_inp);
    }
  } else {
    std::string_view perm_inv_const = args.graph.AddInitializerInt64(perm_shape, args.perm_inv);
    std::vector<std::string_view> gather_inputs {repeats_inp, perm_inv_const};
    auto gather_node_ptr = args.graph.AddNode("Gather", gather_inputs);
    api::Node& gather_node = *gather_node_ptr;
    std::string_view gather_output = gather_node.Outputs()[0];
    args.graph.CopyValueInfo(repeats_inp, gather_output);
    args.node.SetInput(1, gather_output);
  }
  TransposeFirstInput(args.graph, args.node, args.perm_inv);
  TransposeOutputs(args.graph, args.node, args.perm);
  return true;
}

static bool HandleTranspose(HandlerArgs& args) {
  // Two cases: 1 perm match, 2 they don't

  // TODO: assert perm is valid
  std::optional<std::vector<int64_t>> node_perm = args.node.GetAttributeInts("perm");
  if (node_perm == std::nullopt) {
    return false;
  }

  const std::string_view transpose_input = args.transpose.Inputs()[0];
  const std::string_view node_output = args.node.Outputs()[0];
  if (args.perm_inv == *node_perm) {
    auto consumers = args.graph.GetValueConsumers(args.node.Outputs()[0]);
    if (consumers->comprehensive) {
      ReplaceValueReferences(consumers->nodes, node_output, transpose_input);
    }
    else {
      auto transpose_inp_consumers = args.graph.GetValueConsumers(transpose_input);
      std::unique_ptr<api::Node> transpose_inp_node = args.graph.GetNodeProducingOutput(transpose_input);
      if (transpose_inp_node != nullptr && transpose_inp_consumers->comprehensive) {
        args.node.SetInput(0, "");
        ReplaceValueReferences(transpose_inp_consumers->nodes, transpose_input, node_output);
        const std::vector<std::string_view>& transpose_inp_outputs = transpose_inp_node->Outputs();
        size_t i;
        for (i = 0; i < transpose_inp_outputs.size(); ++i) {
          if (transpose_inp_outputs[i] == transpose_input) break;
        }
        args.graph.MoveOutput(args.node, 0, *transpose_inp_node, i);
      } else {
        std::vector<std::string_view> single_empty_input {""};
        auto identity_ptr = args.graph.AddNode("Identity", single_empty_input);
        api::Node& identity = *identity_ptr;
        args.graph.MoveOutput(args.node, 0, identity, 0);
        identity.SetInput(0, transpose_input);
      }
    }
    args.graph.RemoveNode(args.node);
  } else {
    std::vector<int64_t> new_perm = ComposePerm(args.perm, *node_perm);
    args.node.SetAttributeInts("perm", new_perm);
    args.node.SetInput(0, transpose_input);
  }
  if (!args.graph.HasValueConsumers(args.transpose.Outputs()[0])) {
    args.graph.RemoveNode(args.transpose);
  }
  return true;
}

static bool HandleQLinearConcat(HandlerArgs& args) {
  size_t rank = args.perm.size();

  std::vector<size_t> indices;
  size_t num_inputs = args.node.Inputs().size();
  for (size_t i = 2; i < num_inputs; i += 3) {
    indices.push_back(i);
  }
  if (EstimateTransposeInputsCost(args.graph, args.node, args.perm, &indices) >= 0) {
    return false;
  }

  std::optional<int64_t> axis = args.node.GetAttributeInt("axis");
  if (axis == std::nullopt) {
    return false;
  }
  if (*axis < 0) {
    *axis += rank;
  }
  if (*axis < 0 || (size_t)*axis >= rank) {
    return false;
  }
  args.node.SetAttributeInt("axis", args.perm[(size_t)*axis]);
  TransposeInputs(args.graph, args.node, args.perm_inv, &indices);
  TransposeOutputs(args.graph, args.node, args.perm);
  return true;
}

static bool HandleQLinearBinaryOp(HandlerArgs& args) {
  std::vector<size_t> indices { 0, 3 };
  size_t rank = args.perm.size();
  if (EstimateTransposeInputsCost(args.graph, args.node, args.perm, &indices) >= 0) {
    return false;
  }
  if (!NormalizeInputRanks(args.opset, args.graph, args.node, rank, &indices)) {
    return false;
  }
  TransposeInputs(args.graph, args.node, args.perm_inv, &indices);
  TransposeOutputs(args.graph, args.node, args.perm);
  return true;
}

static bool HandleQLinearPoolOp(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  int64_t channels_last = args.node.GetAttributeIntDefault("channels_last", 1);
  size_t rank = args.perm.size();
  if (rank < 2) return false;
  // Channels last to first perm
  auto p = ChannelLastToFirstPerm(rank);
  if ((!channels_last && args.perm == p) || (channels_last && args.perm_inv == p)) {
    args.node.SetAttributeInt("channels_last", 1 - channels_last);
    TransposeFirstInput(args.graph, args.node, args.perm_inv);
    TransposeOutputs(args.graph, args.node, args.perm);
    return true;
  }
  return false;
}

static bool HandleMaxPool(HandlerArgs& args) {
  auto outputs = args.node.Outputs();
  if (outputs.size() == 2 && outputs[1] != "") {
    return false;
  }
  size_t rank = args.perm.size();
  if (args.perm != ChannelLastToFirstPerm(rank)) {
    return false;
  }
  auto inputs = args.node.Inputs();
  std::shared_ptr<api::Node> new_node = args.graph.AddNode("NhwcMaxPool", inputs, /*num_outputs*/ 1, "com.microsoft");
  new_node->CopyAttributes(args.node);
  new_node->ClearAttribute("storage_order");
  args.graph.MoveOutput(args.node, 0, *new_node, 0);
  args.graph.RemoveNode(args.node);
  TransposeFirstInput(args.graph, *new_node, args.perm_inv);
  TransposeOutputs(args.graph, *new_node, args.perm);
  return true;
}

static const std::unordered_map<std::string_view, HandlerFunction*> handler_map {

  {"Cast", &HandleSimpleNode}, {"Exp", &HandleSimpleNode}, {"Identity", &HandleSimpleNode},
  {"LeakyRelu", &HandleSimpleNode}, {"Log", &HandleSimpleNode}, {"Reciprocal", &HandleSimpleNode},
  {"Relu", &HandleSimpleNode}, {"Sigmoid", &HandleSimpleNode}, {"Sqrt", &HandleSimpleNode},
  {"Tanh", &HandleSimpleNode}, {"Abs", &HandleSimpleNode}, {"Ceil", &HandleSimpleNode}, {"Floor", &HandleSimpleNode},
  {"Erf", &HandleSimpleNode}, {"HardSigmoid", &HandleSimpleNode}, {"Round", &HandleSimpleNode},
  {"IsInf", &HandleSimpleNode}, {"IsNaN", &HandleSimpleNode}, {"Neg", &HandleSimpleNode}, {"Not", &HandleSimpleNode},
  {"Selu", &HandleSimpleNode}, {"Shrink", &HandleSimpleNode}, {"Sign", &HandleSimpleNode},
  {"Softplus", &HandleSimpleNode}, {"Softsign", &HandleSimpleNode}, {"ThresholdedRelu", &HandleSimpleNode},
  {"Celu", &HandleSimpleNode}, {"HardSwish", &HandleSimpleNode},

  {"Sin", &HandleSimpleNode}, {"Cos", &HandleSimpleNode}, {"Tan", &HandleSimpleNode},
  {"Sinh", &HandleSimpleNode}, {"Cosh", &HandleSimpleNode}, {"Tanh", &HandleSimpleNode},
  {"Asin", &HandleSimpleNode}, {"Acos", &HandleSimpleNode}, {"Atan", &HandleSimpleNode},
  {"Asinh", &HandleSimpleNode}, {"Acosh", &HandleSimpleNode}, {"Atanh", &HandleSimpleNode},

  {"Add", &HandleSimpleNodeBroadcast}, {"Max", &HandleSimpleNodeBroadcast}, {"Min", &HandleSimpleNodeBroadcast},
  {"Mul", &HandleSimpleNodeBroadcast}, {"Sub", &HandleSimpleNodeBroadcast}, {"Div", &HandleSimpleNodeBroadcast},
  {"And", &HandleSimpleNodeBroadcast}, {"Or", &HandleSimpleNodeBroadcast}, {"Xor", &HandleSimpleNodeBroadcast},
  {"Mod", &HandleSimpleNodeBroadcast}, {"PRelu", &HandleSimpleNodeBroadcast}, {"BitShift", &HandleSimpleNodeBroadcast},
  {"Equal", &HandleSimpleNodeBroadcast}, {"Greater", &HandleSimpleNodeBroadcast}, {"Less", &HandleSimpleNodeBroadcast},
  {"GreaterOrEqual", &HandleSimpleNodeBroadcast}, {"LessOrEqual", &HandleSimpleNodeBroadcast},
  {"Mean", &HandleSimpleNodeBroadcast}, {"Sum", &HandleSimpleNodeBroadcast},  {"Pow", &HandleSimpleNodeBroadcast},
  {"Where", &HandleSimpleNodeBroadcast},

  {"Clip", &HandleSimpleNode1Inp}, {"CastLike", &HandleSimpleNode1Inp},

  {"Transpose", &HandleTranspose},
  {"Concat", &HandleConcat},
  {"Split", &HandleSplit},
  {"Shape", &HandleShape},
  {"Pad", &HandlePad},
  {"ReduceSum", &HandleReduceSum},

  {"ReduceLogSum", &HandleReduceOp}, {"ReduceLogSumExp", &HandleReduceOp}, {"ReduceMax", &HandleReduceOp},
  {"ReduceMean", &HandleReduceOp}, {"ReduceMin", &HandleReduceOp}, {"ReduceProd", &HandleReduceOp},
  {"ReduceSumSquare", &HandleReduceOp}, {"ReduceL1", &HandleReduceOp}, {"ReduceL2", &HandleReduceOp},

  {"ArgMin", &HandleArgMinMax}, {"ArgMax", &HandleArgMinMax},

  {"Squeeze", &HandleSqueeze},
  {"Unsqueeze", &HandleUnsqueeze},
  {"Slice", &HandleSlice},
  {"Tile", &HandleTile},

  {"Softmax", &HandleSoftHardMax}, {"Hardmax", &HandleSoftHardMax}, {"LogSoftmax", &HandleSoftHardMax},

  {"QuantizeLinear", &HandleQuantizeDequantizeLinear}, {"DequantizeLinear", &HandleQuantizeDequantizeLinear},
};

static const std::unordered_map<std::string_view, HandlerFunction*> extended_handler_map {
  {"com.microsoft.QLinearReduceMean", &HandleReduceOp},
  {"com.microsoft.QLinearSigmoid", &HandleSimpleNode1Inp},
  {"com.microsoft.QLinearLeakyRelu", &HandleSimpleNode1Inp},
  {"com.microsoft.QLinearConcat", &HandleQLinearConcat},
  {"com.microsoft.QLinearAdd", &HandleQLinearBinaryOp},
  {"com.microsoft.QLinearMul", &HandleQLinearBinaryOp},
  {"com.microsoft.QLinearAveragePool", &HandleQLinearPoolOp},
  {"com.microsoft.QLinearGlobalAveragePool", &HandleQLinearPoolOp},
  {"MaxPool", &HandleMaxPool},
};

static HandlerFunction* GetHandler(api::Node& node, bool allow_extended_ops) {
  std::string key;
  auto domain = node.Domain();
  auto op_type = node.OpType();
  if (domain == "") {
    key = std::string(op_type);
  } else if (domain == "com.microsoft") {
    key = "com.microsoft." + std::string(op_type);
  } else {
    return nullptr;
  }

  auto match = handler_map.find(key);
  if (match != handler_map.end()) {
    HandlerFunction* fn = match->second;
    return fn;
  } else if (allow_extended_ops) {
    match = extended_handler_map.find(key);
    if (match != extended_handler_map.end()) {
      HandlerFunction* fn = match->second;
      return fn;
    }
  }
  return nullptr;
}

bool ProcessTranspose(HandlerArgs& args, bool allow_extended_ops) {
  if (!CanLikelyRemoveTranspose(args.graph, args.transpose) && !args.node.IsOp("Transpose")) {
    return false;
  }
  HandlerFunction* fn = GetHandler(args.node, allow_extended_ops);
  if (fn == nullptr) {
    return false;
  }
  return fn(args);
}

bool Optimize(api::Graph& graph, bool allow_extended_ops) {
  auto opset = graph.Opset();
  if (opset == std::nullopt || *opset > kMaxSupportedOpset || *opset < kMinSupportedOpset) {
    return false;
  }
  if (allow_extended_ops) {
    auto ms_opset = graph.Opset("com.microsoft");
    if (ms_opset == std::nullopt || *ms_opset != 1) {
      allow_extended_ops = false;
    }
  }
  const std::vector<std::unique_ptr<api::Node>> nodes = graph.Nodes();
  bool changed = false;
  for (size_t i = 0; i < nodes.size(); ++i) {
    api::Node& node = *nodes[i];
    const std::vector<std::string_view> &inputs = node.Inputs();
    for (size_t j = 0; j < inputs.size(); ++j) {
      const std::string_view inp = inputs[j];
      if (inp == "") {
        continue;
      }
      std::unique_ptr<api::Node> transpose = graph.GetNodeProducingOutput(inp);
      if (transpose != nullptr && transpose->IsOp("Transpose")) {
        std::optional<std::vector<int64_t>> perm = transpose->GetAttributeInts("perm");
        if (perm != std::nullopt) {
          std::vector<int64_t> perm_inv = InvertPerm(*perm);
          HandlerArgs ctx = { *opset, graph, *transpose, node, *perm, perm_inv, j, false };
          if (ProcessTranspose(ctx, allow_extended_ops)) {
            changed = true;
            break;
          }
        }
      }
    }
  }
  return changed;
}

static bool ChangeLayout(api::Graph& graph, std::unordered_map<std::string_view, LayoutHandler*>& layout_handler_map, bool last_to_first, bool allow_extended_ops) {
  const std::vector<std::unique_ptr<api::Node>> nodes = graph.Nodes();
  bool changed = false;
  for (size_t i = 0; i < nodes.size(); ++i) {
    api::Node* node = &(*nodes[i]);
    auto match = layout_handler_map.find(node->OpType());
    if (match != layout_handler_map.end()) {
      std::unique_ptr<api::Node> new_node;
      LayoutHandler* handler = match->second;
      LayoutHandlerResult result = handler(graph, *node);
      if (!result.should_transpose) {
        continue;
      }
      size_t rank = result.rank;
      if (result.new_op_type != std::nullopt || result.new_domain != std::nullopt) {
        std::string_view new_op_type;
        if (result.new_op_type != std::nullopt) {
          new_op_type = *result.new_op_type;
        } else {
          new_op_type = node->OpType();
        }
        std::string_view new_domain;
        if (result.new_domain != std::nullopt) {
          new_domain = *result.new_domain;
        } else {
          new_domain = node->Domain();
        }
        auto inputs = node->Inputs();
        auto outputs = node->Outputs();
        new_node = graph.AddNode(new_op_type, inputs, outputs.size(), new_domain);
        for (size_t j = 0; j < outputs.size(); ++j) {
          if (outputs[j] != "") {
            graph.MoveOutput(*node, j, *new_node, j);
          }
        }
        new_node->CopyAttributes(*node);
        graph.RemoveNode(*node);
        node = &(*new_node);
      }
      auto perm = ChannelLastToFirstPerm(rank);
      auto perm_inv = InvertPerm(perm);
      if (last_to_first) {
        std::swap(perm, perm_inv);
      }
      TransposeFirstInput(graph, *node, perm_inv);
      TransposeOutputs(graph, *node, perm);
      changed = true;
    }
  }
  if (changed) {
    Optimize(graph, allow_extended_ops);
  }
  return changed;
}

bool ChannelLastToChannelFirst(api::Graph& graph, std::unordered_map<std::string_view, LayoutHandler*>& layout_handler_map, bool allow_extended_ops) {
  return ChangeLayout(graph, layout_handler_map, /*last_to_first*/ true, allow_extended_ops);
}

bool ChannelFirstToChannelLast(api::Graph& graph, std::unordered_map<std::string_view, LayoutHandler*>& layout_handler_map, bool allow_extended_ops) {
  return ChangeLayout(graph, layout_handler_map, /*last_to_first*/ false, allow_extended_ops);
}

}  // namespace onnx_layout_transformation
