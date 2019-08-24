// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#ifdef USE_NUPHAR
#include "core/providers/providers.h"
#endif // USE_NUPHAR
#include "core/util/math_cpuonly.h"

#include <random>

namespace onnxruntime {

#ifdef USE_NUPHAR
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool, int device_id, const char*);
#endif // USE_NUPHAR

namespace test {

TEST(MatmulIntegerOpTest, MatMulInteger1) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
  test.AddInput<uint8_t>("T2", {3, 2}, {1, 4, 2, 5, 3, 6});
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {4, 2}, {-38, -83, -44, -98, -50, -113, -56, -128});
  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {1, 1}, {11});
  test.AddInput<uint8_t>("T2", {1, 1}, {13});
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {12});
  test.AddOutput<int32_t>("T3", {1, 1}, {-1});
  test.Run();
}

template <typename T>
std::vector<T> ToVector(const int* value, int size) {
  std::vector<T> data(size);
  for (int i = 0; i < size; i++)
    data[i] = static_cast<T>(value[i]);
  return data;
}

// [M x N] = [M x K] x [K x N] = [batch_seq x input_dim] x [input_dim x embed_dim]
void RunMatMulIntegerU8S8Test(const int M, const int N, const int K) {
  OpTester test("MatMulInteger", 10);
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n_unsigned(0, 127);
  static std::uniform_int_distribution<int> n_signed(-128, 127);
  Eigen::MatrixXi T1 = Eigen::MatrixXi::Random(K, M)
                           .unaryExpr([](int) { return n_unsigned(e); });
  Eigen::MatrixXi T2 = Eigen::MatrixXi::Random(N, K)
                           .unaryExpr([](int) { return n_signed(e); });
  Eigen::MatrixXi T3 = (T2 * T1).eval();

  test.AddInput<uint8_t>("T1", {M, K},
                         ToVector<uint8_t>(T1.data(), M * K));
  test.AddInput<int8_t>("T2", {K, N},
                        ToVector<int8_t>(T2.data(), K * N), /*is_initializer*/ true);
  test.AddOutput<int32_t>("T3", {M, N},
                          ToVector<int32_t>(T3.data(), M * N));
  test.Run();

#ifdef NUPHAR_USE_MKL
  // Make sure nuphar's MKL path works as expected
  const char* nuphar_setting = "nuphar_imatmul_force_mkl:1";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(
                        CreateExecutionProviderFactory_Nuphar(/*allow_unaligned_buffers*/ true,
                                                              /*device_id*/ 0,
                                                              nuphar_setting)->CreateProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess,
           /*expected_failure_string*/ "",
           /*excluded_provider_types*/ {},
           /*run_options*/ nullptr,
           &execution_providers);
#endif // NUPHAR_USE_MKL
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8) {
  // GEMV
  RunMatMulIntegerU8S8Test(1, 1, 32);
  RunMatMulIntegerU8S8Test(1, 1, 260);
  RunMatMulIntegerU8S8Test(1, 1, 288);
  RunMatMulIntegerU8S8Test(1, 2, 16);
  RunMatMulIntegerU8S8Test(1, 2, 64);
  // GEMM
  RunMatMulIntegerU8S8Test(2, 2, 40);
  RunMatMulIntegerU8S8Test(2, 48, 33);
  RunMatMulIntegerU8S8Test(2, 51, 40);
  RunMatMulIntegerU8S8Test(6, 10, 34);
  RunMatMulIntegerU8S8Test(8, 16, 64);
}

}  // namespace test
}  // namespace onnxruntime
