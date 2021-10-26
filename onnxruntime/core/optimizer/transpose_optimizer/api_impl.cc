// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/transpose_optimizer/api_impl.h"
#include "core/optimizer/transpose_optimizer/ort_transpose_optimizer.h"
#include "core/providers/cpu/tensor/transpose.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

class ApiValueInfo final : public api::ValueInfoRef {
 private:
  NodeArg& node_arg_;

 public:
  explicit ApiValueInfo(NodeArg& node_arg) : node_arg_(node_arg){};
  std::string_view Name() const override;
  std::optional<std::vector<int64_t>> Shape() const override;
  api::DataType DType() const override;

  void SetShape(const std::vector<int64_t>* shape) override;
  void PermuteDims(const std::vector<int64_t>& perm) override;
  void UnsqueezeDims(const std::vector<int64_t>& axes) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiValueInfo);
};

class ApiTensor final : public api::TensorRef {
 private:
  const onnx::TensorProto& tensor_proto_;
  const Graph& graph_;
  AllocatorPtr cpu_allocator_;

 public:
  explicit ApiTensor(const onnx::TensorProto& tensor_proto, const Graph& graph, AllocatorPtr cpu_allocator) 
      : tensor_proto_(tensor_proto), graph_(graph), cpu_allocator_(std::move(cpu_allocator)){};

  const onnx::TensorProto& TensorProto() {
    return tensor_proto_;
  }

  std::unique_ptr<onnxruntime::Tensor> MakeTensor() const;
  std::vector<int64_t> Shape() const override;
  api::DataType DType() const override;
  std::vector<int64_t> DataInt64() const override;
  std::vector<int32_t> DataInt32() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiTensor);
};

class ApiGraph;

class ApiNode final : public api::NodeRef {
 private:
  onnxruntime::Node& node_;
  Graph& graph_;

 public:
  explicit ApiNode(onnxruntime::Node& node, Graph& graph) : node_(node), graph_(graph){};

  onnxruntime::Node& Node() {
    return node_;
  }

  std::string_view OpType() const override {
    return node_.OpType();
  }
  std::string_view Domain() const override {
    return node_.Domain();
  }
  std::vector<std::string_view> Inputs() const override;
  std::vector<std::string_view> Outputs() const override;
  std::optional<int64_t> GetAttributeInt(std::string_view name) const override;
  std::optional<std::vector<int64_t>> GetAttributeInts(std::string_view name) const override;
  void SetAttributeInt(std::string_view name, int64_t value) override;
  void SetAttributeInts(std::string_view name, const std::vector<int64_t>& value) override;
  void CopyAttributes(const api::NodeRef& node) override;
  void ClearAttribute(std::string_view name) override;
  void SetInput(size_t i, std::string_view name) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiNode);
};

class ApiGraph final : public api::GraphRef {
 private:
  onnxruntime::Graph& graph_;
  AllocatorPtr cpu_allocator_;
  const logging::Logger& logger_;
  const char* new_node_ep_;

 public:
  explicit ApiGraph(onnxruntime::Graph& graph, AllocatorPtr cpu_allocator, const logging::Logger& logger,
                    const char* new_node_ep) : graph_(graph), cpu_allocator_(std::move(cpu_allocator)),
                    logger_(logger), new_node_ep_(new_node_ep){};

  onnxruntime::Graph& Graph() {
    return graph_;
  }

  std::optional<int64_t> Opset(std::string_view domain = "") const override;
  std::vector<std::unique_ptr<api::NodeRef>> Nodes() const override;
  std::unique_ptr<api::TensorRef> GetConstant(std::string_view name) const override;
  std::unique_ptr<api::ValueInfoRef> GetValueInfo(std::string_view name) const override;
  std::unique_ptr<api::ValueConsumers> GetValueConsumers(std::string_view name) const override;
  std::unique_ptr<api::NodeRef> GetNodeProducingOutput(std::string_view name) const override;
  void TransposeInitializer(std::string_view name, const std::vector<int64_t>& perm) override;
  void ReshapeInitializer(std::string_view name, const std::vector<int64_t>& shape) override;
  std::unique_ptr<api::NodeRef> AddNode(std::string_view op_type, const std::vector<std::string_view>& inputs,
                                     size_t num_outputs = 1, std::string_view domain = "") override;
  void RemoveNode(api::NodeRef& node) override;
  void RemoveInitializer(std::string_view name) override;
  std::string_view AddInitializerInt64(const std::vector<int64_t>& shape,
                                             const std::vector<int64_t>& values) override;
  std::string_view AddInitializerInt32(const std::vector<int64_t>& shape,
                                             const std::vector<int32_t>& values) override;
  void MoveOutput(api::NodeRef& src_node, size_t src_idx, api::NodeRef& dst_node, size_t dst_idx) override;
  void CopyValueInfo(std::string_view src_name, std::string_view dst_name) override;
  bool HasValueConsumers(std::string_view name) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiGraph);
};

// <ApiValueInfo>
std::string_view ApiValueInfo::Name() const {
  return node_arg_.Name();
}

const onnx::TensorShapeProto* GetNodeArgShape(const NodeArg* node_arg) {
  if (node_arg == nullptr) {
    return nullptr;
  }

  const auto* type = node_arg->TypeAsProto();
  if (type == nullptr || !utils::HasShape(*type)) {
    return nullptr;
  }

  return &utils::GetShape(*type);
}

std::optional<std::vector<int64_t>> ApiValueInfo::Shape() const {
  const auto* shape_proto = GetNodeArgShape(&node_arg_);
  if (shape_proto == nullptr) {
    return std::nullopt;
  }

  TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
  return shape.GetDims();
}

api::DataType ApiValueInfo::DType() const {
  const auto* type = node_arg_.TypeAsProto();
  if (!utils::HasTensorType(*type)) {
    return api::DataType::UNDEFINED;
  }

  if (!utils::HasElementType(*type)) {
    return api::DataType::UNDEFINED;
  }

  return gsl::narrow_cast<api::DataType>(type->tensor_type().elem_type());
}

void ApiValueInfo::SetShape(const std::vector<int64_t>* shape) {
  if (shape == nullptr) {
    node_arg_.ClearShape();
    return;
  }

  TensorShapeProto new_shape;
  for (int64_t d : *shape) {
    auto* dim = new_shape.add_dim();
    if (d > 0) {
      dim->set_dim_value(d);
    }
  }

  node_arg_.SetShape(new_shape);
}

void ApiValueInfo::PermuteDims(const std::vector<int64_t>& perm) {
  const auto* shape_proto = GetNodeArgShape(&node_arg_);
  if (shape_proto == nullptr) {
    return;
  }

  ORT_ENFORCE(perm.size() == gsl::narrow_cast<size_t>(shape_proto->dim_size()),
              "Permutation length ", perm.size(), " does not match rank ", shape_proto->dim_size());
  TensorShapeProto new_shape;
  for (int64_t p : perm) {
    int p_int = gsl::narrow_cast<int>(p);
    ORT_ENFORCE(0 <= p && p_int < shape_proto->dim_size(),
                "Permutation entry ", p, " out of bounds for shape ", shape_proto->dim_size());
    auto& dim = *new_shape.add_dim();
    const auto& src_dim = shape_proto->dim(p_int);
    dim = src_dim;
  }

  node_arg_.SetShape(new_shape);
}

void ApiValueInfo::UnsqueezeDims(const std::vector<int64_t>& axes) {
  const auto* shape_proto = GetNodeArgShape(&node_arg_);
  if (shape_proto == nullptr) {
    return;
  }

  size_t rank = shape_proto->dim_size();
  TensorShapeProto new_shape;
  int j = 0;
  int64_t i = 0;
  while (true) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      new_shape.add_dim()->set_dim_value(1);
    } else if (gsl::narrow_cast<size_t>(j) < rank) {
      auto& dim = *new_shape.add_dim();
      const auto& src_dim = shape_proto->dim(j);
      dim = src_dim;
      ++j;
    } else {
      break;
    }
    ++i;
  }

  node_arg_.SetShape(new_shape);
}
// </ApiValueInfo>

// <ApiTensor>
std::vector<int64_t> ApiTensor::Shape() const {
  std::vector<int64_t> shape;
  shape.reserve(tensor_proto_.dims_size());
  for (int64_t d : tensor_proto_.dims()) {
    shape.push_back(d);
  }

  return shape;
}

api::DataType ApiTensor::DType() const {
  return gsl::narrow_cast<api::DataType>(tensor_proto_.data_type());
}

// Reading tensor values from tensor_proto requires some work because of external storage and special/raw_data fields
std::unique_ptr<onnxruntime::Tensor> ApiTensor::MakeTensor() const {
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto_.data_type())->GetElementType();
  auto tensor_shape_dims = utils::GetTensorShapeFromTensorProto(tensor_proto_);
  TensorShape tensor_shape{std::move(tensor_shape_dims)};
  auto tensor = onnxruntime::Tensor::Create(tensor_dtype, tensor_shape, cpu_allocator_);
  ORT_THROW_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), graph_.ModelPath().ToPathString().c_str(),
                                                tensor_proto_, *tensor));
  return tensor;
}

template<typename T>
std::vector<T> TensorDataToVector(const onnxruntime::Tensor& tensor) {
  const T* data = tensor.Data<T>();
  size_t num_elements = gsl::narrow_cast<size_t>(tensor.Shape().Size());
  std::vector<T> int_data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    int_data[i] = *data++;
  }

  return int_data;
}

std::vector<int64_t> ApiTensor::DataInt64() const {
  auto tensor = MakeTensor();
  return TensorDataToVector<int64_t>(*tensor);
}

std::vector<int32_t> ApiTensor::DataInt32() const {
  auto tensor = MakeTensor();
  return TensorDataToVector<int32_t>(*tensor);
}
// </ApiTensor>

// <ApiNode>
std::vector<std::string_view> NodeArgsToStrings(ConstPointerContainer<std::vector<NodeArg*>> node_args) {
  std::vector<std::string_view> result;
  result.reserve(node_args.size());
  for (const auto* arg : node_args) {
    result.push_back(arg->Name());
  }

  return result;
}

std::vector<std::string_view> ApiNode::Inputs() const {
  return NodeArgsToStrings(node_.InputDefs());
}

std::vector<std::string_view> ApiNode::Outputs() const {
  return NodeArgsToStrings(node_.OutputDefs());
}

std::optional<int64_t> ApiNode::GetAttributeInt(std::string_view name) const {
  const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(node_, std::string(name));
  if (attr == nullptr || attr->type() != onnx::AttributeProto_AttributeType_INT) {
    return std::nullopt;
  }

  return attr->i();
}

std::optional<std::vector<int64_t>> ApiNode::GetAttributeInts(std::string_view name) const {
  const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(node_, std::string(name));
  if (attr == nullptr || attr->type() != onnx::AttributeProto_AttributeType_INTS) {
    return std::nullopt;
  }

  std::vector<int64_t> value;
  const auto& ints = attr->ints();
  value.reserve(ints.size());
  for (int64_t x : ints) {
    value.push_back(x);
  }

  return value;
}

void ApiNode::SetAttributeInt(std::string_view name, int64_t value) {
  node_.AddAttribute(std::string(name), value);
}

void ApiNode::SetAttributeInts(std::string_view name, const std::vector<int64_t>& value) {
  node_.AddAttribute(std::string(name), value);
}

void ApiNode::CopyAttributes(const api::NodeRef& node) {
  const ApiNode& ort_node = static_cast<const ApiNode&>(node);
  const NodeAttributes& attributes = ort_node.node_.GetAttributes();
  for (const auto& pair : attributes) {
    node_.AddAttribute(pair.first, pair.second);
  }
}

void ApiNode::ClearAttribute(std::string_view name) {
  node_.ClearAttribute(std::string(name));
}

void ApiNode::SetInput(size_t i, std::string_view name) {
  // name could be empty to represent a missing optional.
  const std::string name_str(name);
  NodeArg* new_node_arg = &graph_.GetOrCreateNodeArg(name_str, nullptr);
  auto& mutable_input_defs = node_.MutableInputDefs();

  // Pad with optionals if needed
  while (i >= mutable_input_defs.size()) {
    NodeArg& node_arg = graph_.GetOrCreateNodeArg("", nullptr);
    mutable_input_defs.push_back(&node_arg);

    std::vector<int32_t>& args_count = node_.MutableInputArgsCount();
    size_t j = mutable_input_defs.size() - 1;
    if (j < args_count.size() && args_count[j] == 0) {
      // New input fills missing optional
      args_count[j] = 1;
    } else {
      // Append 1. Technically wrong if last input is variadic (but it never is)
      args_count.push_back(1);
    }

  }

  NodeArg* old_node_arg = mutable_input_defs[i];
  if (old_node_arg->Exists()) {
    // Input may be referenced multiple times. Only remove from consumers if all references are gone.
    size_t usages = 0;
    for (const auto* node_arg : mutable_input_defs) {
      if (node_arg == old_node_arg) {
        ++usages;
      }
    }

    if (usages == 1) {
      graph_.RemoveConsumerNode(old_node_arg->Name(), &node_);
    }

    const auto* old_node = graph_.GetProducerNode(old_node_arg->Name());
    if (old_node != nullptr) {
      int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*old_node, old_node_arg->Name());
      graph_.RemoveEdge(old_node->Index(), node_.Index(), inp_node_out_index, gsl::narrow_cast<int>(i));
    }
  }

  mutable_input_defs[i] = new_node_arg;
  if (new_node_arg->Exists()) {
    graph_.AddConsumerNode(name_str, &node_);
    const auto* inp_node = graph_.GetProducerNode(name_str);
    if (inp_node != nullptr) {
      int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*inp_node, name_str);
      graph_.AddEdge(inp_node->Index(), node_.Index(), inp_node_out_index, gsl::narrow_cast<int>(i));
    }
  }
}
// </ApiNode>

std::optional<int64_t> ApiGraph::Opset(std::string_view domain) const {
  const auto& version_map = graph_.DomainToVersionMap();
  auto match = version_map.find(std::string(domain));
  if (match == version_map.end()) {
    return std::nullopt;
  }

  return match->second;
}

std::vector<std::unique_ptr<api::NodeRef>> ApiGraph::Nodes() const {
  GraphViewer graph_viewer(graph_);
  std::vector<std::unique_ptr<api::NodeRef>> nodes;
  const auto& sorted_nodes = graph_viewer.GetNodesInTopologicalOrder();
  nodes.reserve(sorted_nodes.size());
  for (NodeIndex index : sorted_nodes) {
    auto& node = *graph_.GetNode(index);
    nodes.push_back(std::unique_ptr<api::NodeRef>(new ApiNode(node, graph_)));
  }

  return nodes;
}

std::unique_ptr<api::TensorRef> ApiGraph::GetConstant(std::string_view name) const {
  // TODO: make this work for initializers in parent graphs. See api.h for requirements.
  const auto* tensor = graph_.GetConstantInitializer(std::string(name), false);
  if (tensor == nullptr) {
    return nullptr;
  }

  return std::unique_ptr<api::TensorRef>(new ApiTensor(*tensor, graph_, cpu_allocator_));
}

std::unique_ptr<api::ValueInfoRef> ApiGraph::GetValueInfo(std::string_view name) const {
  NodeArg* node_arg_ = graph_.GetNodeArg(std::string(name));
  ORT_ENFORCE(node_arg_ != nullptr, "No NodeArg found for name ", name);
  return std::unique_ptr<api::ValueInfoRef>(new ApiValueInfo(*node_arg_));
}

std::unique_ptr<api::ValueConsumers> ApiGraph::GetValueConsumers(std::string_view name) const {
  auto consumers = std::make_unique<api::ValueConsumers>();
  consumers->comprehensive = true;
  // Consumers from GetConsumerNodes can be normal (explicit) inputs or implicit inputs used in subgraphs
  auto nodes = graph_.GetConsumerNodes(std::string(name));
  for (const auto* node : nodes) {
    // An input can technically be both an implicit input and an explicit inputs if used statically in a loop subgraph
    // and passed as an initial value for an input to that subgraph.
    for (const auto* input : node->ImplicitInputDefs()) {
      if (input->Exists() && input->Name() == name) {
        consumers->comprehensive = false;
        break;
      }
    }

    for (const auto* input : node->InputDefs()) {
      if (input->Exists() && input->Name() == name) {
        consumers->nodes.push_back(std::unique_ptr<api::NodeRef>(new ApiNode(*graph_.GetNode(node->Index()), graph_)));
        break;
      }
    }
  }

  const auto& graph_outputs = graph_.GetOutputs();
  for (const auto* output : graph_outputs) {
    if (output->Name() == name) {
      consumers->comprehensive = false;
    }
  }

  return consumers;
}

bool ApiGraph::HasValueConsumers(std::string_view name) const {
  auto nodes = graph_.GetConsumerNodes(std::string(name));
  if (nodes.size() > 0) {
    return true;
  }

  const auto& graph_outputs = graph_.GetOutputs();
  for (const auto* output : graph_outputs) {
    if (output->Name() == name) {
      return true;
    }
  }

  return false;
}

std::unique_ptr<api::NodeRef> ApiGraph::GetNodeProducingOutput(std::string_view name) const {
  auto* node = graph_.GetMutableProducerNode(std::string(name));
  if (node == nullptr) {
    return std::unique_ptr<api::NodeRef>(nullptr);
  }

  return std::unique_ptr<api::NodeRef>(new ApiNode(*node, graph_));
}

void ApiGraph::TransposeInitializer(std::string_view name, const std::vector<int64_t>& perm) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  const std::string name_str(name);
  bool success = graph_.GetInitializedTensor(name_str, tensor_proto);
  ORT_ENFORCE(success, "Failed to find initializer for name: ", name_str);
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto->data_type())->GetElementType();
  auto tensor_shape_dims = utils::GetTensorShapeFromTensorProto(*tensor_proto);
  TensorShape tensor_shape{tensor_shape_dims};
  std::unique_ptr<Tensor> in_tensor = Tensor::Create(tensor_dtype, tensor_shape, cpu_allocator_);

  std::vector<int64_t> new_tensor_shape_dims;
  std::vector<size_t> permutations;
  permutations.reserve(perm.size());
  new_tensor_shape_dims.reserve(perm.size());
  for (int64_t p : perm) {
    size_t p_size_t = gsl::narrow_cast<size_t>(p);
    permutations.push_back(p_size_t);
    new_tensor_shape_dims.push_back(tensor_shape_dims[p_size_t]);
  }

  TensorShape new_tensor_shape(new_tensor_shape_dims);

  std::unique_ptr<Tensor> out_tensor = Tensor::Create(tensor_dtype, new_tensor_shape, cpu_allocator_);

  ORT_THROW_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), graph_.ModelPath().ToPathString().c_str(),
                                                *tensor_proto, *in_tensor));

  ORT_THROW_IF_ERROR(Transpose::DoTranspose(permutations, *in_tensor, *out_tensor));

  auto& node_arg = *graph_.GetNodeArg(name_str);
  TensorShapeProto new_shape;
  for (int64_t d : new_tensor_shape_dims) {
    new_shape.add_dim()->set_dim_value(d);
  }

  node_arg.SetShape(new_shape);

  ONNX_NAMESPACE::TensorProto new_tensor_proto = utils::TensorToTensorProto(*out_tensor, name_str);
  graph_.RemoveInitializedTensor(name_str);
  graph_.AddInitializedTensor(new_tensor_proto);
}

void ApiGraph::ReshapeInitializer(std::string_view name, const std::vector<int64_t>& shape) {
  const std::string name_str(name);
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  bool success = graph_.GetInitializedTensor(name_str, tensor_proto);
  ORT_ENFORCE(success, "Failed to find initializer to reshape with name ", name);
  int64_t new_num_elts = 1;
  for (int64_t d : shape) {
    new_num_elts *= d;
  }

  int64_t old_num_elts = 1;
  for (int64_t d : tensor_proto->dims()) {
    old_num_elts *= d;
  }

  ORT_ENFORCE(new_num_elts == old_num_elts, "Cannot reshape initializer ", name,
              " to have different number of elements");

  auto new_tensor_proto = ONNX_NAMESPACE::TensorProto(*tensor_proto);
  new_tensor_proto.clear_dims();
  for (int64_t d : shape) {
    new_tensor_proto.add_dims(d);
  }

  graph_.RemoveInitializedTensor(name_str);
  graph_.AddInitializedTensor(new_tensor_proto);

  auto* node_arg = graph_.GetNodeArg(name_str);
  TensorShapeProto new_shape;
  for (int64_t d : shape) {
    new_shape.add_dim()->set_dim_value(d);
  }

  node_arg->SetShape(new_shape);
}

std::unique_ptr<api::NodeRef> ApiGraph::AddNode(std::string_view op_type,
                                             const std::vector<std::string_view>& inputs, size_t num_outputs, 
                                             std::string_view domain) {
  const std::string op_type_str(op_type);
  std::string name = graph_.GenerateNodeName(op_type_str);
  std::vector<NodeArg*> input_args;
  std::vector<NodeArg*> output_args;

  input_args.reserve(inputs.size());
  for (const auto& input : inputs) {
    NodeArg* arg;
    if (input == "") {
      arg = &graph_.GetOrCreateNodeArg("", nullptr);
    } else {
      arg = graph_.GetNodeArg(std::string(input));
    }
    input_args.push_back(arg);
  }

  output_args.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    std::string output = graph_.GenerateNodeArgName(name + "_out" + std::to_string(i));
    NodeArg* arg = &graph_.GetOrCreateNodeArg(output, nullptr);
    output_args.push_back(arg);
  }

  std::vector<NodeArg*> outputs;
  Node& node = graph_.AddNode(name, op_type_str, "Added in transpose optimizer", input_args, output_args, nullptr,
                              std::string(domain));

  if (new_node_ep_ != nullptr) {
    node.SetExecutionProviderType(new_node_ep_);
  }

  for (size_t i = 0; i < input_args.size(); ++i) {
    NodeArg* arg = input_args[i];
    if (arg->Exists()) {
      const std::string& name_str = arg->Name();
      graph_.AddConsumerNode(name_str, &node);
      const auto* inp_node = graph_.GetProducerNode(name_str);
      if (inp_node != nullptr) {
        int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*inp_node, name_str);
        graph_.AddEdge(inp_node->Index(), node.Index(), inp_node_out_index, gsl::narrow_cast<int>(i));
      }
    }
  }

  for (NodeArg* arg : output_args) {
    graph_.UpdateProducerNode(arg->Name(), node.Index());
  }

  return std::unique_ptr<api::NodeRef>(new ApiNode(node, graph_));
}

void ApiGraph::RemoveNode(api::NodeRef& node) {
  Node& ort_node = static_cast<ApiNode&>(node).Node();
  for (const auto* node_arg : ort_node.InputDefs()) {
    if (node_arg->Exists()) {
      graph_.RemoveConsumerNode(node_arg->Name(), &ort_node);
    }
  }

  graph_.RemoveNode(ort_node.Index());
}

void ApiGraph::RemoveInitializer(std::string_view name) {
  graph_.RemoveInitializedTensor(std::string(name));
}

template<typename T, onnx::TensorProto_DataType DType>
inline ONNX_NAMESPACE::TensorProto TensorProtoFromInts(std::string& name, const std::vector<int64_t>& shape,
                                                       const std::vector<T>& values) {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_data_type(DType);
  tensor_proto.set_name(name);
  tensor_proto.set_raw_data(values.data(), values.size() * sizeof(T));
  for (int64_t dim : shape) {
    tensor_proto.add_dims(dim);
  }

  return tensor_proto;
}

std::string_view ApiGraph::AddInitializerInt64(const std::vector<int64_t>& shape,
                                                     const std::vector<int64_t>& values) {
  std::string name = graph_.GenerateNodeArgName("const_transpose_optimizer");
  ONNX_NAMESPACE::TensorProto tensor_proto =
      TensorProtoFromInts<int64_t, ONNX_NAMESPACE::TensorProto_DataType_INT64>(name, shape, values);
  const auto& node_arg = graph_utils::AddInitializer(graph_, tensor_proto);
  return node_arg.Name();
}

std::string_view ApiGraph::AddInitializerInt32(const std::vector<int64_t>& shape,
                                                     const std::vector<int32_t>& values) {
  std::string name = graph_.GenerateNodeArgName("const_transpose_optimizer");
  ONNX_NAMESPACE::TensorProto tensor_proto =
      TensorProtoFromInts<int32_t, ONNX_NAMESPACE::TensorProto_DataType_INT32>(name, shape, values);
  const auto& node_arg = graph_utils::AddInitializer(graph_, tensor_proto);
  return node_arg.Name();
}

void ApiGraph::MoveOutput(api::NodeRef& src_node, size_t src_idx, api::NodeRef& dst_node, size_t dst_idx) {
  Node& src_ort_node = static_cast<ApiNode&>(src_node).Node();
  Node& dst_ort_node = static_cast<ApiNode&>(dst_node).Node();

  std::vector<NodeArg*>& src_output_defs = src_ort_node.MutableOutputDefs();
  std::vector<NodeArg*>& dst_output_defs = dst_ort_node.MutableOutputDefs();
  const NodeArg* node_arg = src_output_defs[src_idx];
  const std::string& node_arg_name = node_arg->Name();
  dst_output_defs[dst_idx] = src_output_defs[src_idx];
  NodeIndex dst_node_idx = dst_ort_node.Index();
  NodeIndex src_node_idx = src_ort_node.Index();
  graph_.UpdateProducerNode(node_arg_name, dst_node_idx);

  auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(src_ort_node, src_idx);
  int dst_idx_int = gsl::narrow_cast<int>(dst_idx);
  for (auto cur = output_edges.cbegin(), end = output_edges.cend(); cur != end; ++cur) {
    graph_.AddEdge(dst_node_idx, cur->dst_node, dst_idx_int, gsl::narrow_cast<int>(cur->dst_arg_index));
  }

  graph_utils::GraphEdge::RemoveGraphEdges(graph_, output_edges);

  std::string new_name = graph_.GenerateNodeArgName(src_ort_node.Name());
  src_output_defs[src_idx] = &graph_.GetOrCreateNodeArg(new_name, nullptr);
  graph_.UpdateProducerNode(new_name, src_node_idx);
}

void ApiGraph::CopyValueInfo(std::string_view src_name, std::string_view dst_name) {
  NodeArg* src_arg = graph_.GetNodeArg(std::string(src_name));
  if (src_arg != nullptr) {
    NodeArg& dst_arg = graph_.GetOrCreateNodeArg(std::string(dst_name), src_arg->TypeAsProto());
    const TensorShapeProto* shape = src_arg->Shape();
    if (shape == nullptr) {
      dst_arg.ClearShape();
    } else {
      dst_arg.SetShape(*shape);
    }

    ORT_THROW_IF_ERROR(dst_arg.UpdateTypeAndShape(*src_arg, /*strict*/ false, /*override_types*/ false, logger_));
  }
}

std::unique_ptr<api::GraphRef> MakeApiGraph(onnxruntime::Graph& graph, AllocatorPtr cpu_allocator,
                                         const logging::Logger& logger, const char* new_node_ep) {
  return std::unique_ptr<api::GraphRef>(new ApiGraph(graph, std::move(cpu_allocator), logger, new_node_ep));
}

onnxruntime::Graph& GraphFromApiGraph(onnx_layout_transformation::api::GraphRef& graph) {
  return static_cast<ApiGraph&>(graph).Graph();
}

onnxruntime::Node& NodeFromApiNode(onnx_layout_transformation::api::NodeRef& node) {
  return static_cast<ApiNode&>(node).Node();
}

}  // namespace onnxruntime
