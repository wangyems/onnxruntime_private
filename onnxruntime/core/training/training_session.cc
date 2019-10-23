﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/graph/training/loss_function_builder.h"
#include "core/graph/training/training_optimizer.h"
#include "core/training/gradient_graph_builder.h"
#include "core/training/optimizer_graph_builder.h"
#include "core/training/training_session.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/insert_output_rewriter.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/graph/training/mixed_precision_transformer.h"
#include "core/graph/training/tensorboard_transformer.h"
#include "core/graph/training/gradient_builder_base.h"

//Gist Encoding
#include "core/optimizer/gist_encode_decode.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_allocator.h"
#endif

using namespace std;

namespace onnxruntime {
namespace training {

static Status AddLossFunctionInternal(Graph& graph,
                                      ILossFunction& loss_graph_builder,
                                      const LossFunctionInfo& loss_func_info,
                                      const std::string& loss_scale_input_name,
                                      std::string& actual_loss_name) {
  auto loss_function_graph_defs = loss_graph_builder(graph, loss_func_info);

  if (!loss_scale_input_name.empty()) {
    // add node to scale by loss_scale_input_name
    TypeProto* loss_type_proto = loss_function_graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    actual_loss_name = graph.GenerateNodeArgName("scaled_loss");
    loss_function_graph_defs.AddNodeDefs(
        {NodeDef{
            "Mul",
            {ArgDef{loss_func_info.loss_name}, ArgDef{loss_scale_input_name, loss_type_proto}},
            {ArgDef{actual_loss_name, loss_type_proto}}}});
  } else {
    actual_loss_name = loss_func_info.loss_name;
  }

  return GraphAugmenter::AugmentGraph(graph, loss_function_graph_defs);
}

static Status BuildGradientGraphInternal(Graph& graph,
                                         const string& loss_function_output_name,
                                         const unordered_set<string>& node_arg_names_to_train,
                                         const bool set_gradient_as_graph_output = false) {
  // Compute the gradient graph def.
  GradientGraphBuilder grad_graph_builder(&graph,
                                          {loss_function_output_name},
                                          node_arg_names_to_train,
                                          loss_function_output_name,
                                          set_gradient_as_graph_output);
  return grad_graph_builder.Build();
}

static Status BuildOptimizerInternal(Graph& graph,
                                     const OptimizerGraphConfig& opt_graph_config,
                                     const unordered_map<string, OptimizerNodeConfig>& opt_configs,
                                     std::unordered_map<std::string, std::string>& opt_graph_outputs) {
  OptimizerGraphBuilder optimizer_graph_builder{
      OptimizerBuilderRegistry::GetInstance(), opt_graph_config, opt_configs};

  ORT_RETURN_IF_ERROR(optimizer_graph_builder.Build(graph, opt_graph_outputs));

  return Status::OK();
}

static Status AddGradientAccumulationNodes(Graph& graph,
                                           const NodeArgNameGeneratorFn& nodearg_name_generator,
                                           const std::vector<std::string> gradient_names,
                                           bool add_accumulate_as_graph_output) {
  GraphAugmenter::GraphDefs graph_defs{};

  std::vector<ArgDef> gradient_argdefs{};
  ORT_RETURN_IF_ERROR(GetArgDefsFromGraph(graph, gradient_names, gradient_argdefs));
  std::vector<ArgDef> gradient_accumulation_buffers;
  gradient_accumulation_buffers.resize(gradient_argdefs.size());
  std::vector<std::string> grad_acc_outputs;
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    grad_acc_outputs.push_back(BuildGradientAccumulationNode(
                                   nodearg_name_generator, gradient_argdefs[i], gradient_accumulation_buffers[i], graph_defs, false)
                                   .name);
  }
  if (add_accumulate_as_graph_output)
    graph_defs.AddGraphOutputs(grad_acc_outputs);
  return GraphAugmenter::AugmentGraph(graph, graph_defs);
}

common::Status TrainingSession::ApplyTransformationsToMainGraph() {
  try {
    Graph& graph = model_->MainGraph();

    GraphTransformerManager graph_transformation_mgr{1};

    // MUST be empty here, because this is called before partition, so the node's execution type is not decided yet.
    // If we give values here, the check in transformer will fail.
    std::unordered_set<std::string> compatible_eps = {};
    auto gelu_transformer = std::make_unique<GeluFusion>(compatible_eps);
    graph_transformation_mgr.Register(std::move(gelu_transformer), TransformerLevel::Level2);

    auto status = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2);
    return status;
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to apply default optimization passes: ", exp.what());
  }
}

Status TrainingSession::AddGistEncoding() {
  try {
    Graph& graph = model_->MainGraph();

    auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleGistTransformer1");
    rule_transformer_L1->Register(std::make_unique<GistEncodeDecode>());
    onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
    graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

    ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1));
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add Gist Encoding:", exp.what());
  }
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::BuildLossScalingFactorInput(const float loss_scale, std::string& loss_scale_input_name) {
  const std::string input_name = model_->MainGraph().GenerateNodeArgName("loss_scale");
  GraphAugmenter::GraphDefs defs{};
  defs.AddInitializers({CreateTensorProto(input_name, loss_scale, {1})});
  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(model_->MainGraph(), defs));
  ORT_RETURN_IF_ERROR(DoPostLoadProcessing(*model_));
  loss_scale_input_name = input_name;
  return Status::OK();
}

Status TrainingSession::AddTensorboard(const std::string& summary_name,
                                       const std::vector<std::string>& scalar_nodes,
                                       const std::vector<std::string>& histogram_nodes,
                                       const std::vector<std::string>& norm_nodes,
                                       const bool dump_convergence_metrics) {
  ORT_RETURN_IF_ERROR(
    TransformGraphForTensorboard(
      model_->MainGraph(), summary_name, scalar_nodes, histogram_nodes, norm_nodes, dump_convergence_metrics));
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::BuildLossFunction(const LossFunctionInfo& loss_func_info,
                                          const std::string& loss_scale_input_name,
                                          std::string& actual_loss_name) {
  try {
    ORT_RETURN_IF(loss_func_info.op_def.type.empty() || loss_func_info.loss_name.empty(),
                  "BuildLossFunction's loss_function_info is invalid.");

    loss_func_info_ = loss_func_info;
    loss_graph_builder_ = LossFunctionBuilder::Build(loss_func_info_.op_def.type);
    loss_scale_input_name_ = loss_scale_input_name;

    ORT_RETURN_IF_NOT(loss_graph_builder_);
    ORT_RETURN_IF_ERROR(AddLossFunctionInternal(
        model_->MainGraph(), *loss_graph_builder_, loss_func_info_,
        loss_scale_input_name_, actual_loss_name));
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add loss function:", exp.what());
  }
  return DoPostLoadProcessing(*model_);
}

common::Status TrainingSession::EnableMixedPrecision(const std::unordered_set<std::string>& weights_to_train,
                                                     bool use_fp16_initializer,
                                                     std::unordered_map<std::string, NodeArg*>& fp16_weights_map) {
  ORT_RETURN_IF_ERROR(TransformGraphForMixedPrecision(model_->MainGraph(), weights_to_train, use_fp16_initializer, fp16_weights_map));
  return Status::OK();
}

Status TrainingSession::BuildGradientGraph(const unordered_set<string>& weights_to_train,
                                           const string& loss_function_output_name,
                                           const bool set_gradient_as_graph_output) {
  // Fill weights_to_train_ according to weights_to_train
  weights_to_train_ = weights_to_train;

  ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(model_->MainGraph(),
                                                 loss_function_output_name,
                                                 weights_to_train_,
                                                 set_gradient_as_graph_output));

  return DoPostLoadProcessing(*model_);
}

common::Status TrainingSession::BuildAccumulationNode(const std::unordered_set<std::string>& weights_to_train) {
  std::vector<std::string> gradient_names{};
  gradient_names.reserve(weights_to_train.size());
  std::transform(
      weights_to_train.begin(), weights_to_train.end(), std::back_inserter(gradient_names),
      GradientBuilderBase::GradientName);
  auto nodearg_name_generator = [](const std::string& base_name) {
    return base_name;
  };
  ORT_RETURN_IF_ERROR(AddGradientAccumulationNodes(model_->MainGraph(), nodearg_name_generator, gradient_names, false));
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::BuildOptimizer(
    const OptimizerGraphConfig& opt_graph_config,
    const unordered_map<string, OptimizerNodeConfig>& opt_configs,
    std::unordered_map<std::string, std::string>& opt_graph_outputs) {
  ORT_RETURN_IF_NOT(
      opt_configs.size() == weights_to_train_.size(),
      "Number of optimizer configurations does not match number of weights to train.")

  for (const auto& weight_name : weights_to_train_) {
    ORT_RETURN_IF_NOT(
        opt_configs.find(weight_name) != opt_configs.end(),
        "Optimizer configuration was not found for weight to train: ", weight_name);
  }

  opt_graph_config_ = opt_graph_config;
  opt_configs_ = opt_configs;

  ORT_RETURN_IF_ERROR(BuildOptimizerInternal(model_->MainGraph(),
                                             opt_graph_config_,
                                             opt_configs_,
                                             opt_graph_outputs));

  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::OverrideGraphOutputs(const std::vector<std::string>& outputs) {
  ORT_RETURN_IF_ERROR(GraphAugmenter::OverrideGraphOutputs(model_->MainGraph(), outputs));
  return DoPostLoadProcessing(*model_);
}

NameMLValMap TrainingSession::GetWeights() const {
  return session_state_.GetInitializedTensors(weights_to_train_);
}

Status TrainingSession::UpdateWeightsInSessionState(const NameMLValMap& new_weights) {
  session_state_.UpdateInitializedTensors(new_weights);
  VLOGS(*session_logger_, 1) << "Done updating weights";
  return Status::OK();
}

static Status UpdateWeightsBeforeSaving(Graph& graph, const NameMLValMap& weights) {
  // Store MLValue (either in CPU or CUDA) into TensorProto
  // TODO: support more types than float

  for (const auto& name_and_ml_value : weights) {
    // Set src_data pointer
    const auto& src_tensor = name_and_ml_value.second.Get<Tensor>();
    const void* src_data = src_tensor.DataRaw(src_tensor.DataType());

    // Set dst_data pointer
    const ONNX_NAMESPACE::TensorProto* old_tensor_proto = nullptr;
    if (!graph.GetInitializedTensor(name_and_ml_value.first, old_tensor_proto)) {
      continue;
    }
    ONNX_NAMESPACE::TensorProto new_tensor_proto = *old_tensor_proto;
    void* dst_data = nullptr;
    if (new_tensor_proto.has_raw_data()) {
      dst_data = const_cast<char*>(new_tensor_proto.mutable_raw_data()->data());
    } else {
      ORT_ENFORCE(new_tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT);
      dst_data = new_tensor_proto.mutable_float_data()->mutable_data();
    }

    // Copy from src_data to dst_data.
    auto data_size = src_tensor.SizeInBytes();
    if (strcmp(src_tensor.Location().name, CPU) == 0) {
      memcpy(dst_data, src_data, data_size);
    }
#ifdef USE_CUDA
    else if (strcmp(src_tensor.Location().name, CUDA) == 0) {
      ORT_RETURN_IF_NOT(cudaSuccess == cudaMemcpy(dst_data, src_data, data_size, cudaMemcpyDeviceToHost),
                        "cudaMemcpy returns error");
    }
#endif
    else {
      ORT_THROW("Device is not supported:", src_tensor.Location().name);
    }

    // Replace the TensorProto in the model.
    graph.RemoveInitializedTensor(old_tensor_proto->name());
    graph.AddInitializedTensor(new_tensor_proto);
  }
  return Status::OK();
}

Status TrainingSession::Save(const string& model_uri, TrainingSession::SaveOption opt) {
  // Delete the old file before saving.
  std::remove(model_uri.c_str());

  if (opt == TrainingSession::SaveOption::NO_RELOAD) {
    return Model::Save(*model_, model_uri);
  }

  // Have to load the original model again.
  // Because after Initialize(), the model has been optimized and the saved graph doesn't look like what we expect.
  shared_ptr<Model> new_model;
  ORT_RETURN_IF_ERROR(Model::Load(model_location_, new_model));
  ORT_RETURN_IF_ERROR(UpdateWeightsBeforeSaving(new_model->MainGraph(), GetWeights()));

  std::string actual_loss_name{};
  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC /* with weights and loss func*/ ||
      opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS /*with everything*/) {
    ORT_RETURN_IF_NOT(loss_graph_builder_);
    ORT_RETURN_IF_ERROR(AddLossFunctionInternal(
        new_model->MainGraph(),
        *loss_graph_builder_, loss_func_info_,
        loss_scale_input_name_, actual_loss_name));
  }

  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS) {
    ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(new_model->MainGraph(),
                                                   actual_loss_name,
                                                   weights_to_train_,
                                                   false));

    std::unordered_map<std::string, std::string> opt_graph_outputs;
    ORT_RETURN_IF_ERROR(BuildOptimizerInternal(new_model->MainGraph(),
                                               opt_graph_config_,
                                               opt_configs_,
                                               opt_graph_outputs));
  }

  auto status = Model::Save(*new_model, model_uri);

  if (!status.IsOK()) {
    LOGS(*session_logger_, WARNING) << "Error when saving model " << model_uri << " : " << status.ErrorMessage();
  }

  return status;
}

std::unordered_set<std::string> TrainingSession::GetModelInputNames() const {
  return model_input_names_;
}

std::unordered_set<std::string> TrainingSession::GetModelOutputNames() const {
  return model_output_names_;
}

bool TrainingSession::IsUntrainable(const Node* node, const std::string& initializer_name,
                                    const logging::Logger* logger) {
  auto it = STOP_GRADIENT_EDGES.find(node->OpType());
  if (it != STOP_GRADIENT_EDGES.end()) {
    for (auto input_idx : it->second) {
      if (input_idx < node->InputDefs().size() &&
          node->InputDefs()[input_idx]->Name() == initializer_name) {
        if (logger) {
          VLOGS(*logger, 1) << "Excluding " << node->Name() << "'s input " << input_idx
                            << " initializer: " << initializer_name;
        }
        return true;
      }
    }
  }
  return false;
}

bool TrainingSession::IsImmutableWeight(const ImmutableWeights& immutable_weights,
                                        const Node* node, const TensorProto* tensor,
                                        const logging::Logger* logger) {
  auto it = immutable_weights.find(node->OpType());
  if (it == immutable_weights.end()) {
    return false;
  }

  for (auto pair : it->second) {
    size_t& input_idx = pair.first;
    float& value = pair.second;

    if (input_idx < node->InputDefs().size() &&
        node->InputDefs()[input_idx]->Name() == tensor->name()) {
      if (tensor->data_type() == TensorProto_DataType_FLOAT && tensor->dims_size() == 0) {
        float tensor_value;
        if (tensor->has_raw_data()) {
          memcpy(&tensor_value, tensor->raw_data().data(), sizeof(float));
        } else {
          tensor_value = *(tensor->float_data().data());
        }
        if (tensor_value == value) {
          if (logger) {
            VLOGS(*logger, 1) << "Excluding " << node->Name() << "'s input " << input_idx
                              << " initializer: " << tensor->name() << " with value " << tensor_value;
          }
          return true;
        }
      }
    }
  }

  return false;
}

std::unordered_set<std::string> TrainingSession::GetTrainableModelInitializers(
    const ImmutableWeights& immutable_weights) const {
  const Graph& graph = model_->MainGraph();
  const auto& initialized_tensors = graph.GetAllInitializedTensors();
  std::unordered_set<std::string> model_initializers;
  std::transform(initialized_tensors.begin(),
                 initialized_tensors.end(),
                 std::inserter(model_initializers, model_initializers.end()),
                 [](const auto& pair) { return pair.first; });

  std::unordered_set<std::string> trainable_initializers(model_initializers);
  for (const string& initializer_name : model_initializers) {
    const auto& nodes = graph.GetConsumerNodes(initializer_name);
    for (const Node* node : nodes) {
      if (IsUntrainable(node, initializer_name, session_logger_) ||
          IsImmutableWeight(immutable_weights, node, initialized_tensors.at(initializer_name), session_logger_)) {
        trainable_initializers.erase(initializer_name);
      }
    }
  }

  return trainable_initializers;
}

common::Status TrainingSession::UpdateTrainableWeightsInfoInGraph() {
  Graph& graph = model_->MainGraph();
  const auto& graph_inputs = graph.GetInputsIncludingInitializers();
  std::unordered_set<const NodeArg*> inputs_to_add{};
  std::transform(
      weights_to_train_.begin(), weights_to_train_.end(), std::inserter(inputs_to_add, inputs_to_add.end()),
      [&graph](const std::string& node_name) {
        return graph.GetNodeArg(node_name);
      });
  for (const NodeArg* graph_input : graph_inputs) {
    inputs_to_add.erase(graph_input);
  }
  std::vector<const NodeArg*> new_graph_inputs(graph_inputs);
  new_graph_inputs.insert(new_graph_inputs.end(), inputs_to_add.begin(), inputs_to_add.end());
  graph.SetInputs(new_graph_inputs);
  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
