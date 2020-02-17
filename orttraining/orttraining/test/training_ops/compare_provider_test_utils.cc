// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/util/include/default_providers.h"
#include "orttraining/test/training_ops/compare_provider_test_utils.h"

#include "test/compare_ortvalue.h"

using namespace std;

namespace onnxruntime {
namespace test {

std::unique_ptr<IExecutionProvider> GetExecutionProvider(const std::string& provider_type) {
  std::unique_ptr<IExecutionProvider> execution_provider;
  if (provider_type == onnxruntime::kCpuExecutionProvider)
    execution_provider = DefaultCpuExecutionProvider();
  else if (provider_type == onnxruntime::kCudaExecutionProvider)
    execution_provider = DefaultCudaExecutionProvider();
  else if (provider_type == onnxruntime::kDnnlExecutionProvider)
    execution_provider = DefaultDnnlExecutionProvider();
  else if (provider_type == onnxruntime::kNGraphExecutionProvider)
    execution_provider = DefaultNGraphExecutionProvider();
  else if (provider_type == onnxruntime::kNupharExecutionProvider)
    execution_provider = DefaultNupharExecutionProvider();
  else if (provider_type == onnxruntime::kTensorrtExecutionProvider)
    execution_provider = DefaultTensorrtExecutionProvider();
  else if (provider_type == onnxruntime::kOpenVINOExecutionProvider)
    execution_provider = DefaultOpenVINOExecutionProvider();
  else if (provider_type == onnxruntime::kNnapiExecutionProvider)
    execution_provider = DefaultNnapiExecutionProvider();
  else if (provider_type == onnxruntime::kAclExecutionProvider)
    execution_provider = DefaultAclExecutionProvider();
  // skip if execution provider is disabled
  if (execution_provider == nullptr) {
    return nullptr;
  }
  return execution_provider;
}

void CompareOpTester::CompareWithCPU(const std::string& target_provider_type,
                                     double per_sample_tolerance,
                                     double relative_per_sample_tolerance) {
#ifndef NDEBUG
  run_called_ = true;
#endif

  std::unique_ptr<IExecutionProvider> target_execution_provider = GetExecutionProvider(target_provider_type);
  ASSERT_TRUE(target_execution_provider != nullptr) << "provider_type " << target_provider_type << " is not supported.";

  auto p_model = BuildGraph();
  auto& graph = p_model->MainGraph();

  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    return;
  }

  // Hookup the inputs and outputs
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  // Run the model
  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;

  InferenceSession cpu_session_object{so};

  // first run with cpu
  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::istringstream model_proto_str(s1);

  status = cpu_session_object.Load(model_proto_str);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = cpu_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
    return;
  }

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;

  std::vector<MLValue> cpu_fetches;
  status = cpu_session_object.Run(run_options, feeds, output_names, &cpu_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
    return;
  }

  // run with target provider

  InferenceSession target_session_object{so};
  EXPECT_TRUE(target_session_object.RegisterExecutionProvider(std::move(target_execution_provider)).IsOK());

  std::istringstream model_proto_str1(s1);
  status = target_session_object.Load(model_proto_str1);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = target_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
    return;
  }

  std::vector<MLValue> target_fetches;
  status = target_session_object.Run(run_options, feeds, output_names, &target_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  //compare
  ASSERT_TRUE(cpu_fetches.size() == target_fetches.size());
  for (auto i = 0; i < cpu_fetches.size(); i++) {
    auto ret = CompareOrtValue(target_fetches[i], cpu_fetches[i], per_sample_tolerance, relative_per_sample_tolerance, false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

}  // namespace test
}  // namespace onnxruntime
