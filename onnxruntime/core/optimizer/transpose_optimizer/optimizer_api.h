// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <unordered_set>

namespace onnx_layout_transformation {
namespace api {

/* This file defines the API for the transpose optimizer and layout transformation tool. The API consists of a set of
 * abstract classes and methods for graph manipulation that must be implemented to utilize the optimizer tool. The
 * tool attempts to make no assumptions about how ONNX models are represented, other than named values (node outputs,
 * initializers, etc.) should have names representable as string_view objects.
 *
 * Abstract classes like api::GraphRef and api::NodeRef should be thought of as interfaces for manipulating a
 * graph/node, not the graph/node itself. This allows the implementer to use their existing model representation and
 * create interface instances on the fly as they are requested by the optimizer.
 *
 * Since abstract class instances are created on the fly at the optimizer's request (when finding a node with a
 * certain output, for example), they are returned from the implementer as unique_ptr types. Consequently, the
 * implementer does not need to worry about memory management for these classes and no cache of created classes needs
 * to be maintained. However they should be small, ideally containing only pointers to the concrete objects that they
 * manipulate.
 *
 * All editing methods are guaranteed to maintain graph integrity (acyclic, valid input/output names), but validity
 * as an ONNX model may be temporarily violated (ops may have fewer inputs than allowed, incorrect datatypes, etc).
 * Node/output names are generated by the implementer. To avoid having an output generated by multiple nodes during
 * editing, output nodes cannot be directly manipulated by the API. The only method that changes a node's outputs is
 * the MoveOutput method, which transfers an output and all its consumers from one node to another.
 *
 * Some methods for querying value producers/consumers can only be implemented efficiently through the use of indexes,
 * which may be queried regularly between edits and must be carefully maintained.
 */

/// <summary>
/// Enum of DataTypes using standard ONNX values. Casting to/from int32_t is encouraged.
/// </summary>
enum class DataType : int32_t {
  UNDEFINED = 0,
  FLOAT = 1,   // float
  UINT8 = 2,   // uint8_t
  INT8 = 3,    // int8_t
  UINT16 = 4,  // uint16_t
  INT16 = 5,   // int16_t
  INT32 = 6,   // int32_t
  INT64 = 7,   // int64_t
  STRING = 8,  // string
  BOOL = 9,    // bool
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
};

/// <summary>
/// An interface for a constant tensor value used by initializers
/// </summary>
class TensorRef {
 public:
  /// <returns>The shape of the tensor. Values are nonnegative.</returns>
  virtual std::vector<int64_t> Shape() const = 0;

  virtual size_t NumElements() const = 0;

  /// <returns>The dtype of the tensor.</returns>
  virtual DataType DType() const = 0;

  /// <summary>
  /// Retrieves copy of raw data bytes from the tensor. Used for reading initializers specifying axes/pads/scales.
  /// </summary>
  /// <returns>Flattened tensor data in bytes</returns>
  virtual std::vector<uint8_t> Data() const = 0;

  virtual ~TensorRef(){};
};

/// <summary>
/// Interface for accessing/manipulating type/shape information about a value in a graph. The value is either from a
/// graph input, graph initializer, or node output. Must be able to provide up-to-date information on the value of
/// that name unless that value is removed from the graph (in which case behavior is undefined).
/// </summary>
class ValueInfoRef {
 public:
  /// <returns>The name of the value in the graph</returns>
  virtual std::string_view Name() const = 0;

  /// <returns>
  /// The inferred/declared tensor shape of the value. nullopt if rank is unknown, otherwise a vector with entries
  /// representing the dimensions of the value. Use -1 for unknown dimensions.
  /// </returns>
  virtual std::optional<std::vector<int64_t>> Shape() const = 0;

  /// <returns>The inferred/declared dtype of the value. UNDEFINED (0) if dtype is unknown.</returns>
  virtual DataType DType() const = 0;

  /// <summary>
  /// Set the inferred tensor shape. Only used for values that are node outputs. Graph inputs will not change shape
  /// and initializers are expected to update corresponding ValueInfo shapes when the initializer tensor is modified.
  /// </summary>
  /// <param name="shape">nullptr to set an unknown shape. Else, a vector of dim values, with -1 for unknown.</param>
  virtual void SetShape(const std::vector<int64_t>* shape) = 0;

  /// <summary>
  /// Reorders the dimensions of the inferred tensor shape. Only used for values that are node outputs. Has no effect
  /// if rank is unknown. Behavior is undefined if rank does not match number of dimensions in perm. Preferred to
  /// SetShape since it can maintain symbolic shape information.
  /// </summary>
  /// <param name="perm">Permutation for dimensions. An ordering of the values 0 ... rank - 1.</param>
  virtual void PermuteDims(const std::vector<int64_t>& perm) = 0;

  /// <summary>
  /// Inserts constant dimensions of value 1 at the specified axes of the inferred tensor shape. Only used for values
  /// that are node outputs. Has no effect if rank is unknown. Behavior is undefined if axes are negative or exceed
  /// rank + axes.size() - 1. Preferred to SetShape since it can maintain symbolic shape information.
  /// </summary>
  /// <param name="axes">Indices of dimensions to add. Indices are relative to final shape.</param>
  virtual void UnsqueezeDims(const std::vector<int64_t>& axes) = 0;

  virtual ~ValueInfoRef(){};
};

/// <summary>
/// Interface for accessing/manipulating a node in a graph. Information should remain up-to-date even if node is
/// modified, unless it is removed from the graph. Behavior is undefined for methods called on removed nodes.
/// </summary>
class NodeRef {
 public:
  /// <returns>Op computed by the node</returns>
  virtual std::string_view OpType() const = 0;

  /// <returns>Domain containing the op. Empty string if node has no domain set.</returns>
  virtual std::string_view Domain() const = 0;

  /// <returns>Names of input values. Empty string may be included for optional inputs.</returns>
  virtual std::vector<std::string_view> Inputs() const = 0;

  /// <returns>Names of output values. Empty string may be included for optional outputs.</returns>
  virtual std::vector<std::string_view> Outputs() const = 0;

  /// <param name="name">Name of the attribute to return</param>
  /// <returns>
  /// The attribute value, or nullopt if the attribute is not present on the node, or is not of type int.
  /// </returns>
  virtual std::optional<int64_t> GetAttributeInt(std::string_view name) const = 0;

  /// <param name="name">Name of the attribute to return</param>
  /// <returns>
  /// The attribute value, or nullopt if the attribute is not present on the node, or is not of type int[].
  /// </returns>
  virtual std::optional<std::vector<int64_t>> GetAttributeInts(std::string_view name) const = 0;

  /// <summary>
  /// Sets an int attribute with name and value. Overwrites existing value if present.
  /// </summary>
  /// <param name="name">Name of the attribute to set</param>
  /// <param name="value">New value of attribute</param>
  virtual void SetAttributeInt(std::string_view name, int64_t value) = 0;

  /// <summary>
  /// Sets an int[] attribute with name and value. Overwrites existing value if present.
  /// </summary>
  /// <param name="name">Name of the attribute to set</param>
  /// <param name="value">New value of attribute</param>
  virtual void SetAttributeInts(std::string_view name, const std::vector<int64_t>& value) = 0;

  /// <summary>
  /// Copies all attributes from a node to this node
  /// </summary>
  /// <param name="node">Node to copy attributes from</param>
  virtual void CopyAttributes(const NodeRef& node) = 0;

  /// <summary>
  /// Removes attribute with name if present
  /// </summary>
  /// <param name="name">Name of the attribute to clear</param>
  virtual void ClearAttribute(std::string_view name) = 0;

  /// <summary>
  /// Sets the ith input of this node. Supports optional inputs. Expands size if i is out of bounds, padding with ""
  /// as needed.
  /// </summary>
  /// <param name="i">Index of the input to update.</param>
  /// <param name="name">Name of value to use as input or "" for missing optional values.</param>
  virtual void SetInput(size_t i, std::string_view name) = 0;

  /// <summary>
  /// Convenience method. Returns whether node is of the specified op type and domain
  /// </summary>
  /// <param name="op_type">Op type</param>
  /// <param name="domain">Domain. Empty string and "onnx.ai" are treated as equal.</param>
  /// <returns></returns>
  virtual bool IsOp(std::string_view op_type, std::string_view domain = "") const {
    if (OpType() != op_type) {
      return false;
    }
    std::string_view node_domain = Domain();
    return node_domain == domain ||
           ((domain == "" || domain == "ai.onnx") && (node_domain == "" || node_domain == "ai.onnx"));
  }

  /// <summary>
  /// Convenience method. Returns value of int attribute with name, or given default if unset.
  /// </summary>
  /// <param name="name">Attribute name</param>
  /// <param name="default_value">Default value</param>
  /// <returns>Attribute value or default value</returns>
  virtual int64_t GetAttributeIntDefault(std::string_view name, int64_t default_value) const {
    return GetAttributeInt(name).value_or(default_value);
  }

  /// <summary>
  /// Returns the Execution Provider assigned to this node. Any empty string means this node is
  /// not assigned to any EP.
  /// </summary>
  /// <returns>EP type or empty string</returns>
  virtual const std::string& GetExecutionProviderType() const = 0;

  /// <summary>
  /// Returns the schema since version for the op_type of this node. Value of -1 means it is not set.
  /// </summary>
  /// <returns>since version or default value -1</returns>
  virtual int SinceVersion() const = 0;

  virtual ~NodeRef(){};
};

/// <summary>
/// Information regarding the consumers of a value.
/// </summary>
struct ValueConsumers {
  /// <summary>
  /// List of nodes in the current graph containing value as an input
  /// </summary>
  std::vector<std::unique_ptr<NodeRef>> nodes;

  /// <summary>
  /// True if all consumers of the value are present in the nodes list. False if the value is used as a graph output
  /// or within subgraphs.
  /// </summary>
  bool comprehensive = true;
};

/// <summary>
/// Interface for accessing/manipulating a graph or subregion of a graph in a model. Additionally contains methods
/// for querying some model-level information (like model opsets).
///
/// No ability to access subgraphs is provided, but values that are not used exclusively in this graph can be
/// indicated as such by setting `comprehensive` to `false` on `GetValueConsumers` queries.
///
/// In most use cases, the interface will allow for access to the entire graph, but to restrict access to a portion,
/// implementers should refrain from returning references to nodes outside of the subregion. Filtering the outputs of
/// the `Nodes`, `GetValueConsumers`, and `GetNodeProducingOutput` methods is sufficient (and `comprehensive`
/// should be set to `false` as needed).
///
/// Access to parent graphs should be restricted, except GetConstant which may return initializers from parent graphs.
/// </summary>
class GraphRef {
 public:
  /// <param name="domain">Domain name to find in model opset_import</param>
  /// <returns>Opset of domain declared in model, or nullopt if domain is not present</returns>
  virtual std::optional<int64_t> Opset(std::string_view domain) const = 0;

  /// <returns>Topologically-sorted list of nodes in the graph</returns>
  virtual std::vector<std::unique_ptr<NodeRef>> Nodes() const = 0;

  /// <summary>
  /// Checks whether the value name refers to a constant initializer and if so, returns a Tensor corresponding to it.
  /// Constants from parent graphs may be included.
  /// </summary>
  /// <param name="name">Value name. Must be nonempty.</param>
  /// <returns>Tensor corresponding to the constant initializer or nullptr</returns>
  virtual std::unique_ptr<TensorRef> GetConstant(std::string_view name) const = 0;

  /// <summary>
  /// Checks whether the value name refers to a constant initializer in the current graph and if so, returns a Tensor
  /// corresponding to it. The constant must be mutable (able to be edited by Transpose/ReshapeInitializer).
  /// </summary>
  /// <param name="name">Value name. Must be nonempty.</param>
  /// <returns>Tensor corresponding to the mutable constant initializer from this graph, or nullptr</returns>
  virtual std::unique_ptr<TensorRef> GetLocalConstant(std::string_view name) const = 0;

  /// <summary>
  /// Returns a ValueInfo instance for querying info about the value with the given name. Behavior is undefined if
  /// the name does not refer to a value in the graph.
  /// <param name="name">Value name. Must be nonempty.</param>
  /// <returns>A ValueInfo instance corresponding to the value with the given name</returns>
  virtual std::unique_ptr<ValueInfoRef> GetValueInfo(std::string_view name) const = 0;

  /// <summary>
  /// Returns a ValueConsumers object characterizing the current consumers of the value with the specified name. nodes
  /// contains nodes consuming the value as an input. comprehensive is true if the nodes in the result are the only
  /// references to the value within the model (it isn't referenced in subgraphs or graph outputs).
  /// </summary>
  /// <param name="name">The name of the value. Must be nonempty.</param>
  /// <returns>ValueConsumers corresponding to usage of specified value within the model</returns>
  virtual std::unique_ptr<ValueConsumers> GetValueConsumers(std::string_view name) const = 0;

  /// <summary>
  /// Determines if the specified value is a node output and if so returns that node.
  /// </summary>
  /// <param name="name">The name of the value. Must be nonempty.</param>
  /// <returns>Node producing the value or nullptr (or nullptr if value is not a node output)</returns>
  virtual std::unique_ptr<NodeRef> GetNodeProducingOutput(std::string_view name) const = 0;

  /// <summary>
  /// Transposes an initializer "in place". Existing ValueInfo for the initializer must subsequently return the
  /// updated shape. Behavior is undefined if name does not correspond to an initializer in this graph, or if rank of
  /// initializer does not match length of perm.
  /// </summary>
  /// <param name="name">The name of the initializer</param>
  /// <param name="perm">Permutation for transpose. An ordering of the values 0 ... rank - 1.</param>
  virtual void TransposeInitializer(std::string_view name, const std::vector<int64_t>& perm) = 0;

  // Like TransposeInitializer. Product of dims will always match number of elements. Should be fast since
  // data buffer is unchanged.

  /// <summary>
  /// Reshapes an initializer "in place". Existing ValueInfo for the initializer must subsequently return the
  /// updated shape. Behavior is undefined if name does not correspond to an initializer in this graph, or if number
  /// of elements does not match requested shape.
  /// </summary>
  /// <param name="name">The name of the initializer</param>
  /// <param name="shape">New shape. Dimensions are nonnegative.</param>
  virtual void ReshapeInitializer(std::string_view name, const std::vector<int64_t>& shape) = 0;

  /// <summary>
  /// Creates a new node in the graph with the specified op type. Node name and output names are automatically
  /// generated. Outputs of created node have unspecified shapes/dtypes. They will be populated afterwards using
  /// CopyValueInfo.
  /// </summary>
  /// <param name="op_type">The new node's op type</param>
  /// <param name="inputs">Inputs for the node. "" for missing optional inputs.</param>
  /// <param name="num_outputs">
  /// Number of outputs for the node. Names automatically generated. Optional outputs not supported.
  /// </param>
  /// <param name="domain">The new node's domain. Empty string signifies default onnx domain.</param>
  /// <returns>The new node</returns>
  virtual std::unique_ptr<NodeRef> AddNode(std::string_view op_type, const std::vector<std::string_view>& inputs,
                                           size_t num_outputs, std::string_view domain = /*kOnnxDomain*/ "") = 0;

  /// <summary>
  /// Creates a copy of the provided node in the graph with the specified op type and domain.
  /// </summary>
  /// <param name="op_type">The new node's op type</param>
  /// <param name="domain">The new node's domain. Empty string signifies default onnx domain.</param>
  /// <returns>The new node</returns>
  virtual std::unique_ptr<NodeRef> CopyNode(const api::NodeRef& source_node, std::string_view op_type, std::string_view domain = "") = 0;

  /// <summary>
  /// Deletes a node from the graph. Behavior is undefined if node has any consumers.
  /// </summary>
  /// <param name="node">Node to remove</param>
  virtual void RemoveNode(NodeRef& node) = 0;

  /// <summary>
  /// Removes an initializer. Behavior is undefined if initializer has any consumers, or if name does not refer to an
  /// initializer.
  /// </summary>
  /// <param name="name">Name of initializer to remove</param>
  virtual void RemoveInitializer(std::string_view name) = 0;

  /// <summary>
  /// Creates an initializer with the specified dtype, shape, and data. Returns the name.
  /// </summary>
  /// <param name="dtype">DataType for new initializer.</param>
  /// <param name="shape">Dimensions for new initializer. Entries are Nonnegative.</param>
  /// <param name="data">
  /// Raw bytes for new initializer. Length matches product of dimensions and size of the dtype
  /// </param>
  /// <returns>Generated name for the initializer</returns>
  virtual std::string_view AddInitializer(DataType dtype, const std::vector<int64_t>& shape,
                                          const std::vector<uint8_t>& data) = 0;

  /// <summary>
  /// "Moves" an output from one node to another, (effectively transferring the output name, shape, type,
  /// and all consumers, even those in subgraphs). Creates a new output for the source node where the moved output.
  /// was taken from (with unspecified shape/dtype). The destination node's output is guaranteed to have no consumers
  /// before the call and can be deleted once replaced.
  ///
  /// For example, to remove two canceling transposes we could create an Identity node and use MoveOutput to move
  /// the output from the last Transpose to the identity. All former consumers of the transpose op would then
  /// consume the identity op.
  ///
  /// The replacement output is useful when the old op will still need to be used. For example, when pushing
  /// a transpose through a Relu op, we create a new Transpose and move the output from the Relu to it. Then assign
  /// the input of the Transpose to be the newly-generated output from the Relu.
  /// </summary>
  /// <param name="src_node">Node to move the output from. Will be given a new replacement output.</param>
  /// <param name="src_idx">Index of the output to move and then generate a replacement for.</param>
  /// <param name="dst_node">Node to mode the output to.</param>
  /// <param name="dst_idx">Index of the output to replace and delete. Has no consumers.</param>
  virtual void MoveOutput(NodeRef& src_node, size_t src_idx, NodeRef& dst_node, size_t dst_idx) = 0;

  /// <summary>
  /// Copies shape and dtype value info from one output to another, potentially including data that cannot be encoded
  /// in the ValueInfo class (like symbolic shape information). If already set, the destination dtype should be equal
  /// to the source dtype.
  /// </summary>
  /// <param name="src_name"></param>
  /// <param name="dst_name"></param>
  virtual void CopyValueInfo(std::string_view src_name, std::string_view dst_name) = 0;

  /// <summary>
  /// Returns whether there are any consumers of the value with the given name. Override default implementation to
  /// avoid call to GetValueConsumers.
  /// </summary>
  /// <param name="name">The name of the value. Must be nonempty.</param>
  /// <returns>true if the value is not currently referenced anywhere in the model</returns>
  virtual bool HasValueConsumers(std::string_view name) const {
    auto consumers = GetValueConsumers(name);
    bool unused = consumers->comprehensive && consumers->nodes.size() == 0;
    return !unused;
  }

  virtual ~GraphRef(){};
};

}  // namespace api

constexpr int64_t kMinSupportedOpset = 7;
constexpr int64_t kMaxSupportedOpset = 16;

enum class OptimizerMode {
  OPTIMIZE_TRANSPOSE,        // simple transpose optimization
  OPTIMIZE_LAYOUT_TRANSFORM  // transpose optimization post layout transformation
};

/// <summary>
/// Gets a list of layout sensitive ops defined by ONNX standard.
/// </summary>
/// <returns>const reference to an unordered set of op_types which are layout sensitive</returns>
const std::unordered_set<std::string_view>& GetLayoutSensitiveOps();

struct OptimizeResult {
  std::optional<std::string> error_msg;  // set if there was an error
  bool graph_modified{false};
};

/// <summary>
/// Performs transpose optimization on a graph. Returns true if the graph was modified.
///
/// Models outside the supported opset range will be returned unchanged.
///
/// Optimization generally consists of swapping Transpose ops with following ops until a matching Transpose op is
/// encountered. Transpose ops with inverse permutations are canceled. Uses heuristics to attempt to minimize the
/// total cost of Transpose ops and only push Transposes when doing so has some benefit.
/// </summary>
/// <param name="graph">The graph to optimize (or a portion of a graph, see api::GraphRef docs)</param>
/// <param name="allow_extended_ops">Whether com.microsoft ops can be used for optimization</param>
/// <param name="provider_type">Execution provider if applicable.</param>
/// <param name="mode">Current mode. Optimizer can be called in the context of transpose optimizations or during
/// layout transformations.</param>
/// <param name="layout_sensitive_ops">List of ops which are treated as layout sensitive by the ONNX standard
/// as well as any runtime specific ops. These ops should be provided when mode is set to OPTIMIZE_LAYOUT_TRANSFORM.
/// If these ops are not provided, transpose optimizer may convert the layout for these ops </param>
/// <returns>OptimizeResult. If error_msg is set the Optimize failed. If not set, graph_modified indicates whether
/// any changes were required during optimization.</returns>
OptimizeResult Optimize(api::GraphRef& graph, bool allow_extended_ops,
                        const std::string& provider_type = "",
                        OptimizerMode mode = OptimizerMode::OPTIMIZE_TRANSPOSE,
                        const std::unordered_set<std::string_view>& layout_sensitive_ops = {});

/* Layout Transformation Tools
 * These methods help change the channel ordering of layout sensitive ops (like Conv). ONNX currently only supports
 * channel first ordering for ops, so this requires changing the op type and domain to a contrib op supporting
 * the new ordering. The existence of a robust transpose optimizer means that we can freely add transpose ops during
 * conversion and then call Optimize to remove as many as possible. To change the channel ordering of some/all ops
 * in a model, a user of this tool should do the following:
 *
 * 1. Iterate over the graph nodes and identify nodes to convert. For each one:
 *    a. Change the op type and domain (and possibly attributes) to the op/contrib op with the desired ordering.
 *    b. The model is now invalid since the input tensors are in the original ordering (and all consumers
 *       expect the original ordering). Use WrapTransposesAroundNode helper to insert transposes around the
 *       inputs/outputs of the op to correct this.
 * 2. The model is now correct but has many unnecessary Transpose ops. Call Optimize on the graph.
 *
 * After step 1, the Transpose ops will wrap converted ops in a similar manner to q/dq ops in quantization.
 * The perm attributes essentially encode the information about which ops are being reordered.
 */

/// <summary>
/// Inserts transposes around op inputs/outputs. Alternatively transposes initializers or uses existing Transpose
/// nodes if possible. Populates shape information on affected node inputs/outputs to reflect the change.
///
/// Ex:
///   * -> NhwcConv -> **
///   becomes
///   * -> Transpose -> NhwcConv -> Transpose -> **
///   Conv inputs/outputs have new shape. Shapes of * and ** are unchanged (carrying NCHW data).
///
/// input_perms/output_perms are matched with node inputs/outputs positionally. Their lengths must be at most equal to
/// the number of inputs/outputs, respectively. nullptr entires indicate an input or output should not be transposed.
/// </summary>
/// <param name="graph">Graph containing the node</param>
/// <param name="node">Node to modify</param>
/// <param name="input_perms">Input permutations. nullptr entries indicate to skip corresponding input.</param>
/// <param name="output_perms">Output permutations. nullptr entries indicate to skip corresponding output.</param>
void WrapTransposesAroundNode(api::GraphRef& graph, api::NodeRef& node,
                              const std::vector<const std::vector<int64_t>*>& input_perms,
                              const std::vector<const std::vector<int64_t>*>& output_perms);

/// <summary>
/// Computes the perm attribute needed to transpose a tensor from channel-first ordering (NCHW or NCD...D) to
/// channel-last ordering (NHWC or ND...DC). rank must be >= 2.
/// </summary>
/// <param name="rank">Rank of the tensor</param>
/// <returns>perm attribute to transpose from channel first to channel last. Ex: [0, 2, 3, 1]</returns>
std::vector<int64_t> ChannelFirstToLastPerm(size_t rank);

/// <summary>
/// Computes the perm attribute needed to transpose a tensor from channel-last ordering (NHWC or ND...DC) to
/// channel-last ordering (NCHW or NCD...D). rank must be >= 2.
/// </summary>
/// <param name="rank">Rank of the tensor</param>
/// <returns>perm attribute to transpose from channel last to channel first. Ex: [0, 3, 1, 2]</returns>
std::vector<int64_t> ChannelLastToFirstPerm(size_t rank);

/// <summary>
/// Swaps out a node for a new copy of that node with the specified op type and domain. Current API does not all nodes
/// to have their op types or domains changed, so a new node is needed. All attributes, inputs, and outputs are moved
/// to the new node. The old node is removed from the graph and should no longer be accessed.
/// </summary>
/// <param name="graph">Graph containing the node</param>
/// <param name="node">Node to copy and remove</param>
/// <param name="op_type">New node op_type</param>
/// <param name="domain">New node domain. "" for the default domain.</param>
/// <returns>The newly created node.</returns>
std::unique_ptr<api::NodeRef> SwapNodeOpTypeAndDomain(api::GraphRef& graph, api::NodeRef& node,
                                                      std::string_view op_type, std::string_view domain);

}  // namespace onnx_layout_transformation
