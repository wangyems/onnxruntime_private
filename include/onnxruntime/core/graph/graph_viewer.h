// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "core/framework/session_options.h"
#include <unordered_set>

namespace onnxruntime {
class Function;
struct IndexedSubGraph;
}  // namespace onnxruntime

namespace onnxruntime {

// use value-based compare to make sure transformer output order is consistent
struct NodeCompare {
  bool operator()(const Node* n1, const Node* n2) const;
};

/**
@class GraphViewer
Class that provides a read-only view of the Graph.
@remarks If the underlying Graph is changed, GetNodesInTopologicalOrder and GetRootNodes may become invalid.
*/
class GraphViewer {
 public:
  /**
  Construct a GraphViewer from the provided Graph instance.
  */
  explicit GraphViewer(const Graph& graph);

  /**
  Construct a GraphViewer from the provided Graph instance, filtering to the nodes specified in the IndexedSubGraph
  */
  explicit GraphViewer(const Graph& graph, const IndexedSubGraph& filter_info);

  /** Gets the Graph name. */
  const std::string& Name() const noexcept;

  /** Gets the Graph description. */
  const std::string& Description() const noexcept;

  /**
  Gets a tensor created from an initializer.
  @param tensor_name The tensor name
  @param[out] value Sets the pointer to the TensorProto if found, or nullptr if not.
  @returns True if found. False if not.
  */
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const;

  /** Returns true if an initializer value can be overridden by a graph input with the same name. */
  bool CanOverrideInitializer() const noexcept;

  /**
  Gets the Graph inputs, excluding initializers.
  @returns Collection of NodeArg pointers for the graph inputs, excluding inputs that have matching initializers.
  @remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
           Inputs are for filter_info_ if set.
  */
  const std::vector<const NodeArg*>& GetInputs() const noexcept;

  /**
  Gets the Graph inputs, including any initializers.
  @returns Collection of NodeArg pointers for all the graph inputs.
  @remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
           Inputs are for filter_info_ if set.
  */
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept;

  /**
  Gets the Graph outputs.
  @returns Collection of NodeArg pointers for all the graph outputs.
  @remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
           Outputs are for filter_info_ if set.
  */
  const std::vector<const NodeArg*>& GetOutputs() const noexcept;

  /** Gets all ValueInfo NodeArg instances in the Graph.
  @remarks NOT filtered using filter_info_.
  */
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept;

  /**
  Gets the Node instance at the specified index.
  @param node_index Index to retrieve Node from.
  @remarks May return nullptr if index no longer points to a valid node due to the node being freed, or if
           node is excluded by filter_info_.
  */
  const Node* GetNode(NodeIndex node_index) const;

  /**  Gets an iterator over all the valid Nodes in the Graph.
  @remarks Nodes are filtered using filter_info_ if set.
  */
  const ConstGraphNodes& Nodes() const noexcept;

  /** Gets the number of valid nodes in the Graph. 
  @remarks Returns the number of nodes in filter_info_ if set.
  */
  int NumberOfNodes() const noexcept;

  /** Gets the maximum NodeIndex value used by Nodes in the Graph. */
  int MaxNodeIndex() const noexcept;

  /** Gets the NodeIndex values for the Graph nodes, sorted into topological order.  
  @remarks Filtered using filter_info_ if set.
  */
  const std::vector<NodeIndex>& GetNodesInTopologicalOrder(ExecutionOrder order = ExecutionOrder::DEFAULT) const;

  /**
  Gets the NodeIndex values for the root nodes in the Graph.
  The root nodes are the topmost nodes in the Graph that receive inputs from the Graph inputs
  and no other nodes in the Graph.
  @remarks Not supported if filter_info_ is set.
  */
  const std::vector<NodeIndex>& GetRootNodes() const;

  /** Gets all tensors created from initializers. */
  const InitializedTensorSet& GetAllInitializedTensors() const noexcept;

  /**
  Gets the NodeArg instance for the given name.
  @returns A NodeArg if found, a nullptr if not.
  */
  const NodeArg* GetNodeArg(const std::string& name) const;

  /** Gets the map of operator domains to their opset versions. */
  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept {
    return graph_->DomainToVersionMap();
  }

  /** Checks if this is a Subgraph */
  bool IsSubgraph() const;

  /** Get the internal graph*/
  const Graph& GetGraph() const { return *graph_; }

  /**
  returns true if 'name' is an initializer, and is constant and cannot be overridden at runtime.
  @param check_outer_scope If true and the 'graph_' is a subgraph, check parent graph/s for 'name' 
                           if the name is not found in 'graph_'.
  */
  bool IsConstantInitializer(const std::string& name, bool check_outer_scope) const;

  /** Get the Node containing this Graph if IsSubgraph is true. Returns nullptr otherwise. */
  const Node* ParentNode() const noexcept { return graph_->ParentNode(); }

#if !defined(ORT_MINIMAL_BUILD)
  /** Get the consumer nodes of a node arg */
  std::vector<const Node*> GetConsumerNodes(const std::string& node_arg_name) const {
    return graph_->GetConsumerNodes(node_arg_name);
  }

  /** Get the producer node of a node arg */
  const Node* GetProducerNode(const std::string& node_arg_name) const {
    return graph_->GetProducerNode(node_arg_name);
  }
#endif

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphViewer);
  GraphViewer(const Graph& graph, const IndexedSubGraph* filter_info);

  const Graph* graph_;
  ConstGraphNodes graph_nodes_;

  // The NodeIndex values of the graph nodes sorted in topological order.
  std::vector<NodeIndex> nodes_in_topological_order_;

#if !defined(ORT_MINIMAL_BUILD)
  // The NodeIndex values of the graph nodes sorted in topological order with priority.
  std::vector<NodeIndex> nodes_in_topological_order_with_priority_;
#endif

  // Graph root nodes.
  std::vector<NodeIndex> root_nodes_;

  // if we're limiting the view to an IndexedSubGraph we need to create a few pieces of infrastructure that would
  // usually come from the full graph
  const IndexedSubGraph* filter_info_{nullptr};
  std::unordered_set<NodeIndex> filtered_node_indices_;
  std::vector<const NodeArg*> filtered_node_inputs_;
  std::vector<const NodeArg*> filtered_node_inputs_including_initializers_;
  std::vector<const NodeArg*> filtered_node_outputs_;
};
}  // namespace onnxruntime
