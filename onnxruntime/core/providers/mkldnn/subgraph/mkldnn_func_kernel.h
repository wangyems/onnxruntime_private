// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
struct SubgraphParams {
  NodeAttributes attributes;
  MKLDNNExecutionProvider* provider;
  std::shared_ptr<Subgraph> subgraph;
  std::shared_ptr<MKLContext> mkl_context;
  std::string subgraph_id;
  std::string subgraph_key;

  SubgraphParams() {}
};
}  // namespace

template <typename T>
class MkldnnFuncKernel {
 public:
  explicit MkldnnFuncKernel(const ComputeContext* context,
                          const NodeAttributes& attributes,
                          MKLDNNExecutionProvider* provider) {
    params_.provider = provider;
    params_.attributes = attributes;
    params_.mkl_context.reset(new MKLContext(context->allocate_func, context->release_func, context->allocator_handle));

    auto sub_it = attributes.find("subgraph_id");
    if (sub_it->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
      params_.subgraph_id = sub_it->second.s();
      params_.subgraph = provider->GetMklDnnSubgraph(params_.subgraph_id);
      std::ostringstream key_os;
      key_os << params_.subgraph_id << "-" << params_.subgraph->mklnodes.back().name << "-" << params_.subgraph->mklnodes.back().output_name;
      params_.subgraph_key = key_os.str();
    }
  }

  Status Compute(const ONNXRunTimeTensor* input_tensors, const size_t num_inputs,
                 ONNXRunTimeTensor* const output_tensors, const size_t num_outputs) const;

 private:
  SubgraphParams params_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime