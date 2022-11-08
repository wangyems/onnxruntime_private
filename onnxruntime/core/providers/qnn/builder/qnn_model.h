// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared/node_unit/node_unit.h"

namespace onnxruntime {
namespace qnn {
class QnnModel {
 public:
  QnnModel(const logging::Logger& logger,
           QnnBackendManager* qnn_backend_manager,
           const onnxruntime::AllocatorPtr& cpu_allocator,
           bool is_quantized_model = true)
      : cpu_allocator_(cpu_allocator),
        logger_(logger),
        qnn_backend_manager_(qnn_backend_manager),
        is_quantized_model_(is_quantized_model) {
  }

  ~QnnModel() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnModel);

  Status ComposeGraph(const GraphViewer& graph_viewer,
                      const onnxruntime::Node& fused_node,
                      bool debug = false);

  Status FinalizeGraphs();

  Status SetupQnnInputOutput();

  Status ExecuteGraph(Ort::CustomOpApi& ort, OrtKernelContext* context);

  const std::unordered_map<std::string, size_t>& GetInputs() const { return model_input_index_map_; }

  const std::unordered_map<std::string, size_t>& GetOutputs() const { return model_output_index_map_; }
  const OnnxTensorInfo* GetOutputInfo(const std::string& name) const {
    auto it = outputs_info_.find(name);
    if (it == outputs_info_.end()) {
      LOGS_DEFAULT(ERROR) << "GetOutputInfo, output: " << name << "not exist!";
      return nullptr;
    }
    return &(it->second);
  }

  Status SetGraphInputOutputInfo(const GraphViewer& graph_viewer,
                                 const onnxruntime::Node& fused_node);
  Status ParseGraphInputOrOutput(ConstPointerContainer<std::vector<NodeArg*>>& input_output_defs,
                                 std::unordered_map<std::string, OnnxTensorInfo>& input_output_info_table,
                                 std::unordered_map<std::string, size_t>& input_output_index,
                                 std::unordered_map<std::string, size_t>& input_output_index_without_initializers,
                                 bool is_input = false);

  const std::unordered_set<std::string>& GetInitializerInputs() const { return initializer_inputs_; }
  bool IsGraphInitializerInput(const std::string input_name) {
    return initializer_inputs_.find(input_name) != initializer_inputs_.end();
  }

  size_t GetInputIndex(const std::string& name) const {
    return GetInputOutputIndex(name, inputs_info_);
  }

  size_t GetOutputIndex(const std::string& name) const {
    return GetInputOutputIndex(name, outputs_info_);
  }

 private:
  const NodeUnit& GetNodeUnit(const Node* node,
                              const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map) const;
  bool GetGraphInfoFromModel(QnnModelWrapper* model_wrapper);

  onnxruntime::AllocatorPtr GetAllocator() {
    if (cpu_allocator_ == nullptr) {
      LOGS_DEFAULT(ERROR) << "cpu_allocator is null!";
    }
    return cpu_allocator_;
  }

  Status GetQnnTensorDataLength(uint32_t* data_dimensions,
                                uint32_t rank,
                                Qnn_DataType_t data_type,
                                size_t& data_length);

  Status SetupTensors(std::vector<Qnn_Tensor_t>& tensors, const std::vector<QnnTensorWrapper>& tensor_wrappers, bool is_input = true);

 private:

  size_t GetInputOutputIndex(const std::string& name, const std::unordered_map<std::string, OnnxTensorInfo>& io_info) const {
    auto it = io_info.find(name);
    ORT_ENFORCE(it != io_info.end(), "Input/Output name not found.");
    return it->second.index_;
  }

  onnxruntime::AllocatorPtr cpu_allocator_;
  const logging::Logger& logger_;
  std::unique_ptr<GraphInfo> graph_info_;
  QnnBackendManager* qnn_backend_manager_ = nullptr;
  // <input_name, input_index>, initializer inputs are excluded, keep the input index here
  std::unordered_map<std::string, size_t> model_input_index_map_;
  std::unordered_map<std::string, size_t> model_input_index_map_without_initializers_;
  std::unordered_map<std::string, size_t> model_output_index_map_;
  std::unordered_set<std::string> initializer_inputs_;
  bool is_quantized_model_ = false;
  std::unordered_map<std::string, OnnxTensorInfo> inputs_info_;
  std::unordered_map<std::string, OnnxTensorInfo> outputs_info_;
  std::vector<Qnn_Tensor_t> qnn_inputs_;
  std::vector<Qnn_Tensor_t> qnn_outputs_;
};

}  // namespace qnn
}  // namespace onnxruntime
