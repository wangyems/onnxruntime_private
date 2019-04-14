// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(Identity, FloatType) {
  OpTester test("Identity", 9, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run();
}

TEST(Identity, StringType) {
  OpTester test("Identity", 10, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<std::string>("X", dims, {"a" , "b", "x", "y"});
  test.AddOutput<std::string>("Y", dims, {"a" , "b", "x", "y"});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});//TensorRT: unsupported data type
}

}  // namespace test
}  // namespace onnxruntime
