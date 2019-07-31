// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "core/graph/training/loss_func/mean_squared_error.h"

namespace onnxruntime {
namespace training {

GraphAugmenter::GraphDefs MeanSquaredError::operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) {
  const std::string& loss_name = loss_func_info.loss_name;
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 2, " Invalid loss_func_info for MeanSquaredError.");
  const std::string& prediction_name = args[0];
  const std::string& label_name = args[1];

  GraphAugmenter::GraphDefs graph_defs;

  graph_defs.AddGraphOutputs({loss_name});

  std::vector<NodeDef> new_nodes;
  // Sub
  {
    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_name);
    ORT_ENFORCE(prediction_arg != nullptr,
                "Prediction arg ", prediction_name, " is not found in the graph.");
    TypeProto* label_type_proto = graph_defs.CopyTypeProto(prediction_arg);

    new_nodes.emplace_back(NodeDef("Sub",  // Op
                                   {
                                       ArgDef(prediction_name),
                                       ArgDef(label_name, label_type_proto)},
                                   {
                                       ArgDef("MeanSquaredError_diff")  // Outputs
                                   },
                                   NodeAttributes(),
                                   "MeanSquaredError_diff"  // name
                                   ));
  }
  // Pow
  {
    onnx::TensorProto tensor_proto;
    tensor_proto.add_dims(1);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_proto.add_float_data(2.f);
    tensor_proto.set_name("MeanSquaredError_exponent");
    graph_defs.AddInitializers({tensor_proto});

    new_nodes.emplace_back(NodeDef("Pow",  // Op
                                   {
                                       ArgDef("MeanSquaredError_diff"),
                                       ArgDef("MeanSquaredError_exponent")},
                                   {
                                       ArgDef("MeanSquaredError_diff_square")  // Outputs
                                   },
                                   NodeAttributes(),
                                   "MeanSquaredError_pow"  // name
                                   ));
  }
  // ReduceMean
  {
    NodeAttributes attributes;
    {
      onnx::AttributeProto att;
      att.set_name("axes");
      att.set_type(onnx::AttributeProto::INTS);
      att.add_ints(1);
      attributes["axes"] = att;
    }
    {
      onnx::AttributeProto att;
      att.set_name("keepdims");
      att.set_type(onnx::AttributeProto::INT);
      att.set_i(0);
      attributes["keepdims"] = att;
    }
    new_nodes.emplace_back(NodeDef("ReduceMean",  // Op
                                   {
                                       ArgDef("MeanSquaredError_diff_square")},
                                   {
                                       ArgDef(loss_name)  // Outputs
                                   },
                                   attributes,
                                   "MeanSquaredError_reduce_mean"  // name
                                   ));
  }

  graph_defs.AddNodeDefs(new_nodes);

  return graph_defs;
}
}  // namespace training
}  // namespace onnxruntime
