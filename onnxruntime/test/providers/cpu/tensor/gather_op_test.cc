// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

// Some of the tests can't run on TensorrtExecutionProvider because of unsupported data types.
// Those tests will fallback to other EPs

TEST(GatherOpTest, Gather_axis0) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1}, {1LL});
  test.AddOutput<float>("output", {1, 3, 4},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_negative_axis) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", -3LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1}, {1LL});
  test.AddOutput<float>("output", {1, 3, 4},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_invalid_axis) {
  OpTester test("Gather");
  // Invalid axis not in range [-r, r-1]
  test.AddAttribute<int64_t>("axis", -10LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1}, {1LL});
  test.AddOutput<float>("output", {1, 3, 4},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "axis must be in [-r, r-1]");
}

TEST(GatherOpTest, Gather_invalid_index_cpu) {
  OpTester test("Gather", 11);  // added check in opset 11
  // Invalid index 3. data[3] does not exist.
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<float>("data", {3, 4},
                       {0.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 7.0f,
                        8.0f, 9.0f, 10.0f, 11.0f});
  test.AddInput<int32_t>("indices", {3}, {0LL, 1L, 1000L});
  test.AddOutput<float>("output", {1}, {1.0f});

  test.Run(OpTester::ExpectResult::kExpectFailure, "Mismatch between number of source and target dimensions.");
}

#ifdef USE_CUDA
TEST(GatherOpTest, Gather_invalid_index_gpu) {
  OpTester test("Gather");
  // Invalid index 3. data[3] does not exist.
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<float>("data", {3, 4},
                       {0.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 7.0f,
                        8.0f, 9.0f, 10.0f, 11.0f});
  test.AddInput<int32_t>("indices", {3}, {0LL, 1LL, 1000LL});
  test.AddOutput<float>("output", {3, 4},
                        {0.0f, 1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f, 7.0f,
                         0.0f, 0.0f, 0.0f, 0.0f});

  //On GPU, just set the value to 0 instead of report error. exclude all other providers
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kDnnlExecutionProvider, kNupharExecutionProvider, kTensorrtExecutionProvider});
}
#endif

TEST(GatherOpTest, Gather_axis1) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {2}, {2LL, 0LL});
  test.AddOutput<float>("output", {2, 2, 4},
                        {2.0f, 2.1f, 2.2f, 2.3f,
                         0.0f, 0.1f, 0.2f, 0.3f,
                         12.0f, 12.1f, 12.2f, 12.3f,
                         10.0f, 10.1f, 10.2f, 10.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_axis2) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 2LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {3}, {1LL, 0LL, 2LL});
  test.AddOutput<float>("output", {2, 3, 3},
                        {0.1f, 0.0f, 0.2f,
                         1.1f, 1.0f, 1.2f,
                         2.1f, 2.0f, 2.2f,
                         10.1f, 10.0f, 10.2f,
                         11.1f, 11.0f, 11.2f,
                         12.1f, 12.0f, 12.2f});
  test.Run();
}

const std::vector<MLFloat16> FloatToMLFloat16(const std::vector<float>& float_data) {
  std::vector<MLFloat16> new_data;
  for (const auto& f : float_data) {
    new_data.push_back(MLFloat16(math::floatToHalf(f)));
  }
  return new_data;
}

#ifdef USE_CUDA
#if __CUDA_ARCH__ >= 700
//TODO: Currently this cannot pass CI, due to GPU architecture problem
TEST(GatherOpTest, Gather_axis0_indices2d_half) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<MLFloat16>("data", {3, 3},
                           FloatToMLFloat16({0.0f, 0.1f, 0.2f,
                                             1.0f, 1.1f, 1.2f,
                                             2.0f, 2.1f, 2.2f}));
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {1LL, 0LL,
                          2LL, 1LL});
  test.AddOutput<MLFloat16>("output", {2, 2, 3},
                            FloatToMLFloat16({1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f,
                                              2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f}));
  test.Run();
}

TEST(GatherOpGradTest, GatherGrad_axis0_indices2d_half) {
  OpTester test("GatherGrad", 9);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         {3, 3});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {0LL, 1LL,
                          0LL, 1LL});

  test.AddInput<MLFloat16>("grad", {2, 2, 3},
                           FloatToMLFloat16({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));
  test.AddOutput<MLFloat16>("output", {3, 3},
                            FloatToMLFloat16({0, 2, 4, 6, 8, 10, 0, 0, 0}));
  test.Run();
}
#endif
#endif

TEST(GatherOpGradTest, GatherGrad_axis0_indices2d_float) {
  OpTester test("GatherGrad", 9);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         {3, 3});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {0LL, 1LL,
                          0LL, 1LL});

  test.AddInput<float>("grad", {2, 2, 3},
                       {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
  test.AddOutput<float>("output", {3, 3},
                        {0, 2, 4, 6, 8, 10, 0, 0, 0});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<float>("data", {3, 3},
                       {0.0f, 0.1f, 0.2f,
                        1.0f, 1.1f, 1.2f,
                        2.0f, 2.1f, 2.2f});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {1LL, 0LL,
                          2LL, 1LL});
  test.AddOutput<float>("output", {3, 2, 2},
                        {0.1f, 0.0f, 0.2f, 0.1f,
                         1.1f, 1.0f, 1.2f, 1.1f,
                         2.1f, 2.0f, 2.2f, 2.1f});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d_int32) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<int32_t>("data", {3, 3},
                         {0, 1, 2,
                          10, 11, 12,
                          20, 21, 22});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<int32_t>("output", {3, 2, 2},
                          {1, 0, 2, 1,
                           11, 10, 12, 11,
                           21, 20, 22, 21});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Input batch size is inconsistent
}

TEST(GatherOpTest, Gather_axis1_indices2d_uint32) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<uint32_t>("data", {3, 3},
                          {0, 1, 2,
                           10, 11, 12,
                           20, 21, 22});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<uint32_t>("output", {3, 2, 2},
                           {1, 0, 2, 1,
                            11, 10, 12, 11,
                            21, 20, 22, 21});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d_int16) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<int16_t>("data", {3, 3},
                         {0, 1, 2,
                          10, 11, 12,
                          20, 21, 22});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<int16_t>("output", {3, 2, 2},
                          {1, 0, 2, 1,
                           11, 10, 12, 11,
                           21, 20, 22, 21});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d_uint16) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<uint16_t>("data", {3, 3},
                          {0, 1, 2,
                           10, 11, 12,
                           20, 21, 22});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<uint16_t>("output", {3, 2, 2},
                           {1, 0, 2, 1,
                            11, 10, 12, 11,
                            21, 20, 22, 21});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d_int8) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<int8_t>("data", {3, 3},
                        {0, 1, 2,
                         10, 11, 12,
                         20, 21, 22});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<int8_t>("output", {3, 2, 2},
                         {1, 0, 2, 1,
                          11, 10, 12, 11,
                          21, 20, 22, 21});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Assertion `regionRanges != nullptr' failed
}

TEST(GatherOpTest, Gather_axis1_indices2d_string) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<std::string>("data", {3, 3},
                             {"0", "1", "2",
                              "10", "11", "12",
                              "20", "21", "22"});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<std::string>("output", {3, 2, 2},
                              {"1", "0", "2", "1",
                               "11", "10", "12", "11",
                               "21", "20", "22", "21"});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d_bool) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<bool>("data", {3, 3},
                      {true, false, true,
                       true, true, false,
                       false, true, false});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<bool>("output", {3, 2, 2},
                       {false, true, true, false,
                        true, true, false, true,
                        true, false, false, true});
  test.Run();
}

TEST(GatherOpTest, Gather_perf) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  std::vector<int32_t> input(50000 * 100, 1);

  std::vector<int32_t> indices(800, 5);

  std::vector<int32_t> output(800 * 100, 1);

  test.AddInput<int32_t>("data", {50000, 100}, input);
  test.AddInput<int32_t>("indices", {800, 1}, indices);
  test.AddOutput<int32_t>("output", {800, 1, 100}, output);
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_neg_indices2d_int8) {
  OpTester test("Gather", 11);
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<int8_t>("data", {3, 3},
                        {0, 1, 2,
                         10, 11, 12,
                         20, 21, 22});
  test.AddInput<int32_t>("indices", {2, 2},
                         {-2, -3,
                          -1, -2});
  test.AddOutput<int8_t>("output", {3, 2, 2},
                         {1, 0, 2, 1,
                          11, 10, 12, 11,
                          21, 20, 22, 21});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Assertion `regionRanges != nullptr' failed
}

template <typename T>
void CalculateOutput(const int64_t stride, const int64_t num_input_before_gather_axis,
                     const int64_t num_input_from_gather_axis, const std::vector<T>& grad, const std::vector<int64_t>& indices, std::vector<T>& output) {
  std::map<int64_t, std::vector<T> > indices_grad;
  for (int64_t t = 0; t < num_input_before_gather_axis; ++t) {
    auto offset1 = t * num_input_from_gather_axis;
    for (size_t i = 0; i < indices.size(); ++i) {
      auto offset2 = (t * indices.size() + i) * stride;
      auto index = offset1 + indices[i];
      if (indices_grad.count(index)) {
        for (int64_t j = 0; j < stride; ++j) {
          indices_grad[index][j] += grad[offset2 + j];
        }
      } else {
        for (int64_t j = 0; j < stride; ++j) {
          indices_grad[index].push_back(grad[offset2 + j]);
        }
      }
    }
  }
  for (auto& itr : indices_grad) {
    for (int64_t i = 0; i < stride; ++i) {
      output[itr.first * stride + i] = itr.second[i];
    }
  }
}

TEST(GatherOpTest, Gather_axis1_float_impl2) {
  OpTester test("GatherGrad", 9);
  int64_t axis_0 = 3;
  int64_t axis_1 = 6;
  int64_t axis_2 = 128;
  int64_t output_shape = 4;
  RandomValueGenerator random{};
  std::vector<float> grad(random.Uniform<float>({axis_0 * axis_1 * axis_2}, 1.0f, 1.0f));
  std::vector<int64_t> indices(random.Uniform<int64_t>({axis_1 * axis_2}, (int64_t)0, (int64_t)3));

  std::vector<int64_t> shape{axis_0, output_shape};
  std::vector<float> output(axis_0 * output_shape);

  CalculateOutput(1, axis_0, 4, grad, indices, output);

  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<int64_t>("shape", {2},
                         shape);
  test.AddInput<int64_t>("indices", {axis_1, axis_2},
                         indices);
  test.AddInput<float>("grad", {axis_0, axis_1, axis_2},
                       grad);
  test.AddOutput<float>("output", shape,
                        output);
  test.Run();
}

TEST(GatherOpTest, Gather_axis0_float_impl2) {
  OpTester test("GatherGrad", 9);
  int64_t axis_0 = 3;
  int64_t axis_1 = 6;
  int64_t axis_2 = 128;
  int64_t output_shape = 4;
  RandomValueGenerator random{};
  std::vector<float> grad(random.Uniform<float>({axis_1 * axis_2 * output_shape}, 1.0f, 1.0f));
  std::vector<int64_t> indices(random.Uniform<int64_t>({axis_1 * axis_2}, (int64_t)0, (int64_t)3));
  std::vector<int64_t> shape{axis_0, output_shape};
  std::vector<float> output(axis_0 * output_shape);

  CalculateOutput(output_shape, 1, 12, grad, indices, output);

  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         shape);
  test.AddInput<int64_t>("indices", {axis_1, axis_2},
                         indices);
  test.AddInput<float>("grad", {axis_1, axis_2, output_shape},
                       grad);
  test.AddOutput<float>("output", shape,
                        output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
