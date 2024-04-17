/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx512vnni.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"

#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx_common_fp32.h"

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32(
  size_t BlkLen,
  const float* A,
  const std::byte* QuantBData,
  const float* QuantBScale,
  const std::byte* QuantBZeroPoint,
  float* C,
  size_t CountN,
  size_t CountK,
  size_t BlockStrideQuantB,
  const float* Bias
)
{
  if (BlkLen >= 32)
  {
    if (QuantBZeroPoint != nullptr) {
      MlasQ4GemmKernelBlkLen32PlusAvx512f<true>(
        BlkLen,
        A,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        1,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias,
        0,
        0
      );
    }
    else {
      MlasQ4GemmKernelBlkLen32PlusAvx512f<false>(
        BlkLen,
        A,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        1,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias,
        0,
        0
      );
    }
  }
  else {
    if (QuantBZeroPoint != nullptr) {
      MlasQ4GemmKernelBlkLen16Avx512f<true>(
        A,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        1,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias,
        0,
        0
      );
    }
    else {
      MlasQ4GemmKernelBlkLen16Avx512f<false>(
        A,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        1,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias,
        0,
        0
      );
    }
  }
}

void MLASCALL
MlasQ80BlkQuantRow_avx512(
  size_t BlkLen,
  const float* A,
  size_t CountK,
  std::byte* QuantA
);

//
// Kernel dispatch structure definition.
//
const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx512vnni = []() {
  MLAS_SQNBIT_GEMM_DISPATCH d;

  d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
  d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;

  d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32;
  d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32_avx2;

  d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8_avx512vnni;
  d.QuantizeARow_CompInt8 = MlasQ80BlkQuantRow_avx512;

  return d;
  }();
