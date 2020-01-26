// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/generic_registry.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "core/framework/data_types.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "onnx/defs/attr_proto_util.h"
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace training {

template <class T>
ONNX_NAMESPACE::TensorProto CreateTensorProto(
    const std::string& name,
    T val,
    const std::vector<int64_t>& dims = {1}) {
  size_t count = static_cast<size_t>(std::accumulate(dims.begin(), dims.end(), int64_t(1), std::multiplies<int64_t>{}));
  std::vector<T> values(count, val);
  ONNX_NAMESPACE::TensorProto tensor_proto = ONNX_NAMESPACE::ToTensor<T>(values);
  tensor_proto.set_name(name);
  std::for_each(dims.begin(), dims.end(), [&](auto dim) { tensor_proto.add_dims(dim); });
  return tensor_proto;
}

template <class T>
ONNX_NAMESPACE::TensorProto CreateTensorProto(
    const std::string& name,
    const std::vector<T>& values,
    const std::vector<int64_t>& dims = {1}) {
  size_t count = static_cast<size_t>(std::accumulate(dims.begin(), dims.end(), int64_t(1), std::multiplies<int64_t>{}));
  ORT_ENFORCE(values.size() == count);
  ONNX_NAMESPACE::TensorProto tensor_proto = ONNX_NAMESPACE::ToTensor<T>(values);
  tensor_proto.set_name(name);
  std::for_each(dims.begin(), dims.end(), [&](auto dim) { tensor_proto.add_dims(dim); });
  return tensor_proto;
}

class OptimizerBuilder {
 public:
  OptimizerBuilder(const std::string& name, const std::vector<std::string>& attribute_names = {})
      : name_(name),
        attr_names_(attribute_names) {}

  virtual ~OptimizerBuilder() {}

  /**
   * Adds the optimizer node to the graph.
   * This component may be placed into an If node subgraph to enable
   * conditional execution.
   *
   * @param weight_argdef The ArgDef of the weight to optimize.
   * @param gradient_argdef The ArgDef of the gradient of the weight to
            optimize.
   * @param gradient_norm_argdef (Optional) The ArgDef of gradient norm.
   * @param gradient_norm_finite_argdef (Optional) The ArgDef indicates whether
            the passed-in gradient norm is finite.
   * @param opt_config The optimizer configuration.
   * @param[out] graph_defs The GraphDefs corresponding to the graph (possibly
   *             a subgraph) that the component is to be added to.
   * @param[out] new_external_initializers Any initializers that should be
   *             placed in the parent graph, if there is one.
   *             Other initializers are treated as local to the current
   *             (sub)graph.
   * @param[out] output_weight_argdefs The output weight ArgDef. All optimizers
                 should have this output.
   * @param[out] output_gradient_argdefs The output gradient ArgDef. All optimizers
                 should have this output.
   *
   * @return The status of the optimizer node addition.
   */
  virtual Status Build(
      const std::vector<ArgDef>& weight_argdefs,
      const std::vector<ArgDef>& gradient_argdefs,
      const ArgDef* gradient_norm_argdef,
      const ArgDef* gradient_norm_finite_argdef,
      const std::vector<OptimizerNodeConfig>& opt_configs,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ONNX_NAMESPACE::TensorProto>& new_external_initializers,
      std::vector<ArgDef>& output_weight_argdefs,
      std::vector<ArgDef>& output_gradient_argdefs) const = 0;

  const std::string& OpType() const { return name_; }

 protected:
  const std::string OptimizerNodeName(const std::string& weight_name) const {
    return name_ + "_" + weight_name;
  }

  std::vector<ONNX_NAMESPACE::AttributeProto> BuildAttributeProto(const OptimizerNodeConfig& opt_config) const {
    std::vector<ONNX_NAMESPACE::AttributeProto> attribute_protos;
    for (auto attr_name : attr_names_) {
      if (opt_config.attributes.count(attr_name)) {
        attribute_protos.push_back(ONNX_NAMESPACE::MakeAttribute(attr_name, opt_config.attributes.at(attr_name)));
      }
    }
    return attribute_protos;
  }

 private:
  std::string name_;
  std::vector<std::string> attr_names_;
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

}  // namespace training
}  // namespace onnxruntime
