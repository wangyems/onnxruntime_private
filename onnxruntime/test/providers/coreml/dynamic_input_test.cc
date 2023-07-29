// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/providers/coreml/coreml_execution_provider.h"
#include "test/providers/model_tester.h"
#include "test/common/random_generator.h"
#include "test/util/include/test_utils.h"

namespace onnxruntime::test {

TEST(CoreMLExecutionProviderDynamicInputShapeTest, MatMul) {
  constexpr auto model_path = ORT_TSTR("testdata/matmul_with_dynamic_input_shape.onnx");

  auto test = [&](const size_t M) {
    SCOPED_TRACE(MakeString("M=", M));

    auto coreml_ep = std::make_unique<CoreMLExecutionProvider>(0);

    const auto ep_verification_params = EPVerificationParams{
        ExpectedEPNodeAssignment::All,
        2e-3f,
    };

    RandomValueGenerator gen{1234};
    const auto A_shape = std::vector<int64_t>{static_cast<int64_t>(M), 2};
    const auto A_data = gen.Uniform<float>(A_shape, 0.0f, 1.0f);

    OrtValue A;
    CreateInputOrtValueOnCPU(A_shape, A_data, &A);

    RunAndVerifyOutputsWithEP(model_path, "CoreMLEPDynamicInputShape.MatMul",
                              std::move(coreml_ep),
                              {{"A", A}},
                              ep_verification_params);
  };

  for (size_t i = 1; i <= 5; ++i) {
    test(i);
  }
}

TEST(CoreMLExecutionProviderDynamicInputShapeTest, MobileNetExcerpt) {
  constexpr auto model_path = ORT_TSTR("testdata/mobilenet_v3_small_excerpt.onnx");

  auto test = [&](const size_t batch_size) {
    SCOPED_TRACE(MakeString("batch_size=", batch_size));

    auto coreml_ep = std::make_unique<CoreMLExecutionProvider>(0);

    const auto ep_verification_params = EPVerificationParams{
        ExpectedEPNodeAssignment::Some,
        5e-2f,
    };

    RandomValueGenerator gen{1234};
    const auto input_shape = std::vector<int64_t>{static_cast<int64_t>(batch_size), 3, 224, 224};
    const auto input_data = gen.Uniform<float>(input_shape, 0.0f, 1.0f);

    OrtValue input;
    CreateInputOrtValueOnCPU(input_shape, input_data, &input);

    RunAndVerifyOutputsWithEP(model_path, "CoreMLEPDynamicInputShape.MobileNet",
                              std::move(coreml_ep),
                              {{"input", input}},
                              ep_verification_params);
  };

  for (size_t i = 1; i <= 5; ++i) {
    test(i);
  }
}

TEST(CoreMLExecutionProviderDynamicInputShapeTest, EmptyInputFails) {
  constexpr auto model_path = ORT_TSTR("testdata/matmul_with_dynamic_input_shape.onnx");

  ModelTester tester("CoreMLEPDynamicInputShape.EmptyInputFails", model_path);

  tester.AddInput<float>("A", {0, 2}, {});
  tester.AddOutput<float>("Y", {0, 4}, {});

  auto eps = std::vector<std::unique_ptr<IExecutionProvider>>{};
  eps.emplace_back(std::make_unique<CoreMLExecutionProvider>(0));

  tester
      .Config(ModelTester::ExpectResult::kExpectFailure,
              "the runtime shape ([ 0 2 ]) has zero elements. This is not supported by the CoreML EP.")
      .ConfigEps(std::move(eps))
      .RunWithConfig();
}

}  // namespace onnxruntime::test
