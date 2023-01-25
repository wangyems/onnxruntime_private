/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_halfgemm.h

Abstract:

    Tests for MLAS half precision GEMM.

--*/

#pragma once

#include "test_util.h"
#include "mlas_float16.h"


//
// Define our own fp16 type to avoid dragging in big dependencies
//
struct MLFp16 {
  uint16_t val{0};

  MLFp16() = default;
  explicit constexpr MLFp16(uint16_t x) : val(x) {}
  explicit constexpr MLFp16(int32_t x) : val((uint16_t)x) {}
  explicit MLFp16(float ff) : val(MLAS_Float2Half(ff)) {}

  float ToFloat() const {
    return MLAS_Half2Float(val);
  }

  operator float() const { return ToFloat(); }

  MLFp16& operator=(float ff) {
    val = MLAS_Float2Half(ff);
    return *this;
  }
};

inline bool
operator==(const MLFp16& left, const MLFp16& right) {
  return left.val == right.val;
}

inline bool
operator!=(const MLFp16& left, const MLFp16& right) {
  return left.val != right.val;
}

template<typename T>
void SmallFloatFill(T* start, size_t size) {
  constexpr float MinimumFillValue = -11.0f;
  auto* FillAddress = start;
  size_t offset = size % 23;

  for (size_t i = 0; i < size; i++) {
    offset = (offset + 21) % 23;
    *FillAddress++ = T((MinimumFillValue + offset) / 16.0f);
  }
}


/**
 * @brief Test class for half precision GEMM
 * @tparam AType  Data type of A matrix, can be either float or MLFp16
 * @tparam BType  Data type of b matrix, can be either float or MLFp16
*/
template <typename AType, typename BType, bool Packed, bool Threaded>
class MlasHalfGemmTest : public MlasTestBase {

private:
  MatrixGuardBuffer<uint8_t> BufferBPacked;
  MatrixGuardBuffer<AType> BufferA;
  MatrixGuardBuffer<BType> BufferB;
  MatrixGuardBuffer<MLFp16> BufferBias;
  MatrixGuardBuffer<MLFp16> BufferC;
  MatrixGuardBuffer<float> BufferCReference;
  MLAS_THREADPOOL* threadpool_;

  void* PackB(size_t N, size_t K, const BType* B, size_t ldb) {
    size_t PackedBSize = MlasHalfGemmPackBSize(N, K, std::is_same<BType, float>::value);
    if (PackedBSize == 0) {
      return nullptr;
    }
    void* PackedB = BufferBPacked.GetBuffer(PackedBSize);
    if (std::is_same<BType, float>::value) {
      MlasHalfGemmConvertPackB(N, K, (const float*)B, ldb, PackedB);
    } else {
      MlasHalfGemmPackB(N, K, (const MLAS_FP16*)B, ldb, PackedB);
    }
    return PackedB;
  }

  void CallGemm(size_t M,
                size_t N,
                size_t K,
                size_t BatchSize,
                const AType* A,
                size_t lda,
                const BType* B,
                size_t ldb,
                const MLFp16* Bias,
                MLFp16* C,
                size_t ldc) {

    std::vector<MLAS_HALF_GEMM_DATA_PARAMS> GemmParameters(BatchSize);

    for (size_t i = 0; i < GemmParameters.size(); i++) {
      auto& params = GemmParameters[i];
      params.A = A + (M * lda * i);
      params.lda = lda;
      if (nullptr != Bias) {
        params.Bias = reinterpret_cast<const MLAS_FP16*>(Bias + N * i);
      } else {
        params.Bias = nullptr;
      }
      params.C = reinterpret_cast<MLAS_FP16*>(C + (M * ldc * i));
      params.ldc = ldc;

      if (Packed) {
        ASSERT_EQ(BatchSize, size_t(1)) << "Packing B not supported in batching yet!";
        params.B = PackB(N, K, B, ldb);
        params.ldb = 0;
      } else {
        params.B = B + (K * N * i);
        params.ldb = ldb;
      }
      params.AIsfp32 = std::is_same<AType, float>::value;
      params.BIsfp32 = std::is_same<BType, float>::value;
    }

    MlasHalfGemmBatch(M, N, K, BatchSize, GemmParameters.data(), threadpool_);
  }

  void ReferenceQgemm(size_t M,
                      size_t N,
                      size_t K,
                      size_t BatchSize,
                      const AType* A,
                      const BType* B,
                      const MLFp16* Bias,
                      float* C) {
    for (size_t batch = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
          const AType* a = A + M * K * batch + m * K;
          const BType* b = B + K * N * batch + n;
          float* c = C + (M * N * batch) + (m * N) + n;
          float sum = Bias == nullptr ? 0.0f : float(Bias[n]);

          for (size_t k = 0; k < K; k++) {
            MLFp16 down(float(*b) * float(*a) + sum);
            sum = float(down);
            b += N;
            a += 1;
          }

          *c = sum;
        }
      }
      if (Bias) {
        Bias += N;
      }
    }
  }

public:
  MlasHalfGemmTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void Test(size_t M, size_t N, size_t K, size_t BatchSize, bool withBias) {
    const AType* A = BufferA.GetFilledBuffer(K * M * BatchSize, SmallFloatFill<AType>);
    const BType* B = BufferB.GetFilledBuffer(N * K * BatchSize, SmallFloatFill<BType>);
    const MLFp16* Bias = withBias ? BufferBias.GetFilledBuffer(N * BatchSize, SmallFloatFill<MLFp16>) : nullptr;
    MLFp16* C = BufferC.GetFilledBuffer(N * M * BatchSize, SmallFloatFill<MLFp16>);
    float* CReference = BufferCReference.GetFilledBuffer(
        N * M * BatchSize,
        [](float* start, size_t size) {
          std::fill_n(start, size, -1.0f);
        });

    this->CallGemm(M, N, K, BatchSize, A, K, B, N, Bias, C, N);
    ReferenceQgemm(M, N, K, BatchSize, A, B, Bias, CReference);

    for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++, f++) {
          ASSERT_EQ(float(C[f]), CReference[f]) << "@[" << batch << "x" << m << "x" << n << "], "
                                                << "Batch=" << BatchSize << "M=" << M << ", N=" << N << ", K=" << K;
        }
      }
    }
  }

 private:

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("HalfGemmFP") +
                                    (std::is_same<AType, float>::value ? "32" : "16") +
                                    (std::is_same<BType, float>::value ? "32" : "16") +
                                    (Packed ? "_Packed" : "_NoPack") +
                                    (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }

  void ExecuteLong(void) override {
    for (size_t M = 16; M < 160; M += 32) {
      for (size_t N = 16; N < 160; N += 32) {
        static const size_t ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320};
        for (size_t k = 0; k < _countof(ks); k++) {
          size_t K = ks[k];

          Test(M, N, K, 1, false);
          Test(M, N, K, 1, true);
          Test(M + 1, N, K, 1, false);
          Test(M, N + 1, K, 1, true);
          Test(M + 1, N + 1, K, 1, false);
          Test(M + 3, N + 2, K, 1, true);
          Test(M + 4, N, K, 1, false);
          Test(M, N + 4, K, 1, true);
          Test(M + 4, N + 4, K, 1, false);
          Test(M + 3, N + 7, K, 1, true);
          Test(M + 8, N, K, 1, false);
          Test(M, N + 8, K, 1, true);
          Test(M + 12, N + 12, K, 1, false);
          Test(M + 13, N, K, 1, true);
          Test(M, N + 15, K, 1, false);
          Test(M + 15, N + 15, K, 1, false);
          if (!Packed) {
            Test(M, N, K, 7, false);
            Test(M + 3, N, K, 8, true);
            Test(M, N + 1, K, 9, false);
            Test(M + 12, N, K, 10, true);
            Test(M, N + 15, K, 11, false);
            Test(M + 15, N + 15, K, 12, true);
          }
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 1; M < 160; M++) {
      for (size_t N = 1; N < 160; N++) {
        for (size_t K = 1; K < 160; K++) {
          Test(M, N, K, 1, true);
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
      for (size_t N = 112; N < 320; N += 24) {
        for (size_t K = 1; K < 16; K++) {
          Test(M, N, K, 1, true);
        }
        for (size_t K = 16; K < 160; K += 32) {
          Test(M, N, K, 1, false);
        }
      }
      printf("M %zd\n", M);
    }
  }


};


