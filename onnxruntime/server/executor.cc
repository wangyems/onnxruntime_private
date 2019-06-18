// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdio.h>
#include <onnx/onnx_pb.h>
#include "core/common/logging/logging.h"
#include "core/framework/data_types.h"
#include "core/framework/environment.h"
#include "core/framework/framework_common.h"
#include "serializing/mem_buffer.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "serializing/tensorprotoutils.h"
#include "core/common/callback.h"

#include "onnx-ml.pb.h"
#include "predict.pb.h"

#include "converter.h"
#include "executor.h"
#include "util.h"

namespace onnxruntime {
namespace server {

namespace protobufutil = google::protobuf::util;

protobufutil::Status Executor::SetMLValue(const onnx::TensorProto& input_tensor,
                                          MemBufferArray& buffers,
                                          OrtAllocatorInfo* cpu_allocator_info,
                                          /* out */ Ort::Value& ml_value) {
  auto logger = env_->GetLogger(request_id_);

  size_t cpu_tensor_length = 0;
  auto status = onnxruntime::server::GetSizeInBytesFromTensorProto<0>(input_tensor, &cpu_tensor_length);
  if (!status.IsOK()) {
    LOGS(*logger, ERROR) << "GetSizeInBytesFromTensorProto() failed. Error Message: " << status.ToString();
    return GenerateProtobufStatus(status, "GetSizeInBytesFromTensorProto() failed: " + status.ToString());
  }

  auto* buf = buffers.AllocNewBuffer(cpu_tensor_length);
  status = onnxruntime::server::TensorProtoToMLValue(input_tensor,
                                                    onnxruntime::server::MemBuffer(buf, cpu_tensor_length, *cpu_allocator_info),
                                                    ml_value);
  if (!status.IsOK()) {
    LOGS(*logger, ERROR) << "TensorProtoToMLValue() failed. Message: " << status.ToString();
    return GenerateProtobufStatus(status, "TensorProtoToMLValue() failed:" + status.ToString());
  }

  return protobufutil::Status::OK;
}

protobufutil::Status Executor::SetNameMLValueMap(std::vector <std::string>& input_names,
                                                 std::vector <Ort::Value>& input_values,
                                                 const onnxruntime::server::PredictRequest& request,
                                                 MemBufferArray& buffers) {
  auto logger = env_->GetLogger(request_id_);

  OrtAllocatorInfo* allocator_info = nullptr;
  auto ort_status = OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info);

  if (ort_status != nullptr || allocator_info == nullptr) {
    LOGS(*logger, ERROR) << "OrtCreateAllocatorInfo failed";
    return protobufutil::Status(protobufutil::error::Code::RESOURCE_EXHAUSTED, "OrtCreateAllocatorInfo() failed");
  }

  // Prepare the MLValue object
  for (const auto& input : request.inputs()) {
    using_raw_data_ = using_raw_data_ && input.second.has_raw_data();

    Ort::Value ml_value {nullptr};
    auto status = SetMLValue(input.second, buffers, allocator_info, ml_value);
    if (status != protobufutil::Status::OK) {
      OrtReleaseAllocatorInfo(allocator_info);
      LOGS(*logger, ERROR) << "SetMLValue() failed! Input name: " << input.first;
      return status;
    }

    input_names.push_back(input.first);
    input_values.push_back(std::move(ml_value));
    }

  OrtReleaseAllocatorInfo(allocator_info);
  return protobufutil::Status::OK;
}




std::vector<Ort::Value> Run(const Ort::Session& session, const Ort::RunOptions& options, const std::vector <std::string>& input_names, const std::vector <Ort::Value>& input_values, const std::vector<std::string>& output_names){
  size_t input_count = input_names.size();
  size_t output_count = output_names.size();

  std::vector<const char*> input_ptrs{input_count};
  for (auto const& input: input_names){
    input_ptrs.push_back(input.data());
  }
  std::vector<const char *> output_ptrs{output_count};
  for (auto const& output: output_names){
    output_ptrs.push_back(output.data());
  }

  return session.Run(options, input_ptrs.data(), input_values.data(), input_count, output_ptrs.data(), output_count); 

}


protobufutil::Status Executor::Predict(const std::string& model_name,
                                       const std::string& model_version,
                                       onnxruntime::server::PredictRequest& request,
                                       /* out */ onnxruntime::server::PredictResponse& response) {
  auto logger = env_->GetLogger(request_id_);

  // Convert PredictRequest to NameMLValMap
  MemBufferArray buffer_array;
  std::vector <std::string> input_names;
  std::vector <Ort::Value> input_values;
  auto conversion_status = SetNameMLValueMap(input_names, input_values, request, buffer_array);
  if (conversion_status != protobufutil::Status::OK) {
    return conversion_status;
  }

  Ort::RunOptions run_options{};
  run_options.SetRunLogVerbosityLevel(static_cast<unsigned int>(env_->GetLogSeverity()));
  run_options.SetRunTag(request_id_.c_str());


  // Prepare the output names
  std::vector<std::string> output_names;

  if (!request.output_filter().empty()) {
    output_names.reserve(request.output_filter_size());
    for (auto const& name: request.output_filter()){
      output_names.push_back(name);
    }
  } else {
    output_names = env_->GetModelOutputNames();
  }

  

//TODO Add exception handling.
  auto outputs = Run(env_->GetSession(), run_options, input_names, input_values, output_names);

/* 
  if (!status.IsOK()) {
    LOGS(*logger, ERROR) << "Run() failed."
                         << ". Error Message: " << status.ToString();
    return GenerateProtobufStatus(status, "Run() failed: " + status.ToString());
  }
*/

  // Build the response
  for (size_t i = 0, sz = outputs.size(); i < sz; ++i) {
    onnx::TensorProto output_tensor{};
    auto status = MLValueToTensorProto(outputs[i], using_raw_data_, std::move(logger), output_tensor);
    logger = env_->GetLogger(request_id_);

    if (!status.IsOK()) {
      LOGS(*logger, ERROR) << "MLValueToTensorProto() failed. Output name: " << output_names[i] << ". Error Message: " << status.ToString();
      return GenerateProtobufStatus(status, "MLValueToTensorProto() failed: " + status.ToString());
    }

    auto insertion_result = response.mutable_outputs()->insert({output_names[i], output_tensor});

    if (!insertion_result.second) {
      LOGS(*logger, ERROR) << "SetNameMLValueMap() failed. Output name: " << output_names[i] << " Trying to overwrite existing output value";
      return protobufutil::Status(protobufutil::error::Code::INVALID_ARGUMENT, "SetNameMLValueMap() failed: Cannot have two outputs with the same name");
    }
  }

  return protobufutil::Status::OK;
}

}  // namespace server
}  // namespace onnxruntime