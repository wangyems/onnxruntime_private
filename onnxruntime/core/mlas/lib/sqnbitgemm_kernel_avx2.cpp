/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx2.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"

#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx_common_int8.h"


MLAS_FORCEINLINE
__m256
load_float_n_avx2(const float* data, int n)
{
    assert(n <= 8);
    if (n <= 0) {
        return _mm256_setzero_ps();
    }
    static const int32_t mask_buffer[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
    const __m256i load_mask = _mm256_loadu_si256((const __m256i*)(mask_buffer + 8 - n));
    return _mm256_maskload_ps(data, load_mask);
}

MLAS_FORCEINLINE
void
SQ4BitGemmM1Kernel_CompInt8_avx2(
  size_t BlkLen,
  const std::byte* QuantA,
  const std::byte* QuantBData,
  const float* QuantBScale,
  const std::byte* QuantBZeroPoint,
  float* C,
  size_t CountN,
  size_t CountK,
  size_t BlockStrideQuantB,
  const float* Bias
) {
  if (QuantBZeroPoint != nullptr) {
    constexpr bool HasZeroPoint = true;
    if (BlkLen == 16) {
      SQ4BitGemmM1Kernel_BlkLen16_CompInt8_Impl<HasZeroPoint>(
        QuantA,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
      );
    } else if (BlkLen == 32) {
      SQ4BitGemmM1Kernel_BlkLen32_CompInt8_Impl<HasZeroPoint, accumulate_mul_sum_avx2<HasZeroPoint>>(
        QuantA,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        BlockStrideQuantB,
        Bias
      );
    } else {
      SQ4BitGemmM1Kernel_BlkLen64Plus_CompInt8_Impl<HasZeroPoint, dot_quad_avx2>(
        BlkLen,
        QuantA,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
      );
    }
  } else {
    constexpr bool HasZeroPoint = false;
    if (BlkLen == 16) {
      SQ4BitGemmM1Kernel_BlkLen16_CompInt8_Impl<HasZeroPoint>(
        QuantA,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
      );
    } else if (BlkLen == 32) {
      SQ4BitGemmM1Kernel_BlkLen32_CompInt8_Impl<HasZeroPoint, accumulate_mul_sum_avx2<HasZeroPoint>>(
        QuantA,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        BlockStrideQuantB,
        Bias
      );
    } else {
      SQ4BitGemmM1Kernel_BlkLen64Plus_CompInt8_Impl<HasZeroPoint, dot_quad_avx2>(
        BlkLen,
        QuantA,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
      );
    }
  }
}

template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkLen16_CompFp32_avx2(
  size_t BlkLen,
  const float* ARowPtr,
  const std::byte* QuantBDataColPtr,
  const float* QuantBScaleColPtr,
  const std::byte* QuantBZeroPointColPtr,
  float* sum_ptr,
  size_t CountK,
  size_t StrideQuantBData,
  size_t StrideQuantBScale,
  size_t StrideQuantBZeroPoint,
  const float* bias_ptr
)
{
  if constexpr (!HasZeroPoint) {
    // Suppress unused variable warnings
    (void)QuantBZeroPointColPtr;
    (void)StrideQuantBZeroPoint;
  }

  constexpr size_t BlkBitWidth4 = 4;
  constexpr size_t SubBlkLen16 = 16;
  constexpr size_t SubBlkStep8 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, SubBlkLen16);
  static_assert(SubBlkStep8 == 8);  // 16 * 4 / 8

  __m256 acc[NCols];
  UnrolledLoop<NCols>([&](size_t i) {
    acc[i] = _mm256_setzero_ps();
    });

  const std::byte* b_blk_data_ptr = QuantBDataColPtr;
  const float* s = QuantBScaleColPtr;

  [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
  // only used if HasZeroPoint == true

  for (size_t k = 0; k < CountK; k += BlkLen) {
    size_t ck = std::min(CountK - k, BlkLen);

    float scale_v[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      scale_v[i] = *(s + StrideQuantBScale * i);
      });

    std::byte* b_blk_data_col_ptr[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      b_blk_data_col_ptr[i] = (std::byte*)(b_blk_data_ptr + StrideQuantBData * i);
      });

    [[maybe_unused]] uint8_t offset[NCols];
    // not ready for "Manual conversion to float" in neon yet. following neon to unpack to uint8_t.
    if constexpr (HasZeroPoint) {
      UnrolledLoop<NCols>([&](size_t i) {
        const std::byte zp_packed =
          QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
          ? (zp_packed >> 4)
          : (zp_packed & std::byte{ 0x0F });
        offset[i] = std::to_integer<uint8_t>(zp);
        });
    }

    for (size_t kk = 0; kk < ck; kk += SubBlkLen16) {
      int kklen = std::min((int)SubBlkLen16, (int)(ck - kk));

      // Load A row vectors
      int n_to_read = std::min(kklen, 8);
      __m256 av_lo = load_float_n_avx2(ARowPtr + k + kk, n_to_read);
      n_to_read = std::min(kklen - 8, 8);
      __m256 av_hi = load_float_n_avx2(ARowPtr + k + kk + 8, n_to_read);

      UnrolledLoop<NCols>([&](size_t i) {
        // SubBlkLen = 16: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
        // SubBlkLen = 32: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
        // Load B col vectors. get SubBlkLen(16) 4 bits quantized features from each column
        __m128i bvi4 = _mm_loadl_epi64(b_blk_data_col_ptr[i]);
        b_blk_data_col_ptr[i] += SubBlkStep8;

        // TODO: avoid _mm_set1_epi8
        //__m128i lower_mask_epi8 = _mm_cmpeq_epi16(bvi4, bvi4); // can use any __m128i
        //lower_mask_epi8 = _mm_srli_epi16(lower_mask_epi8, 13);
        //lower_mask_epi8 = _mm_packus_epi16(lower_mask_epi8, lower_mask_epi8);
        __m128i lower_mask_epi8 = _mm_set1_epi8(0x0F);  // Mask to isolate the lower 4 bits

        const __m128i lower = _mm_and_si128(bvi4, lower_mask_epi8);
        const __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi4, 4), lower_mask_epi8), 8);
        __m256i bv_epi16 = _mm256_cvtepu8_epi16(_mm_add_epi8(upper, lower)); // unpacked 16 weights of epi16

        // Subtract zero-point from the integers
        if constexpr (HasZeroPoint) {
          // Subtract zero-point from the integers
          __m256i zp = _mm256_set1_epi16(offset[i]);
          bv_epi16 = _mm256_sub_epi16(bv_epi16, zp);
        } else {
          // Subtract 8 from the integers
          const __m256i eight = _mm256_set1_epi16(8);
          bv_epi16 = _mm256_sub_epi16(bv_epi16, eight);
        }

        // Convert to 16 epi16 to 16 float32
        const __m128i bv_lo = _mm256_extractf128_si256(bv_epi16, 0);
        const __m128i bv_hi = _mm256_extractf128_si256(bv_epi16, 1);

        __m256 bvf_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_lo));
        __m256 bvf_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_hi));

        // multiply by scale
        __m256 scale_ps = _mm256_set1_ps(scale_v[i]);
        bvf_lo = _mm256_mul_ps(bvf_lo, scale_ps);
        bvf_hi = _mm256_mul_ps(bvf_hi, scale_ps);

        // c[m,n] += a[m,k] * b[k,n]
        acc[i] = _mm256_fmadd_ps(bvf_lo, av_lo, acc[i]);
        acc[i] = _mm256_fmadd_ps(bvf_hi, av_hi, acc[i]);
        });
    } // kk

    b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    s++;

    if constexpr (HasZeroPoint) {
      QuantBZeroPointIdx += 1;
    }
  } // k

  if constexpr (NCols == 4) {
    __m128 acc_x = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
    if (bias_ptr != nullptr) {
      acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
    }
    _mm_storeu_ps(sum_ptr, acc_x);
  } else {
    UnrolledLoop<NCols>([&](size_t i) {
      __m128 vlow = _mm256_castps256_ps128(acc[i]);
      __m128 vhigh = _mm256_extractf128_ps(acc[i], 1); // Extract high 128 bit

      // Add the two 128-bit vectors together
      __m128 vsum = _mm_add_ps(vlow, vhigh);
      // Horizontally add the elements of the resulting 128-bit vector
      vsum = _mm_hadd_ps(vsum, vsum);
      vsum = _mm_hadd_ps(vsum, vsum);

      _mm_store_ss(&sum_ptr[i], vsum);
      sum_ptr[i] += bias_ptr == nullptr ? 0.0f : bias_ptr[i];
      });
  }
}

// TODO: flow MlasQ4GemmKernelBlkLen16Avx512f to improve perf
template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_BlkLen16_CompFp32_avx2(
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
  constexpr size_t BlkLen16 = 16;
  constexpr size_t BlkBitWidth4 = 4;
  constexpr size_t NCols4 = 4;

  const float* ARowPtr = A;
  float* CRowPtr = C;

  const size_t BlockCountK = BlockStrideQuantB;

  const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
  const size_t StrideQuantBScale = BlockCountK;
  const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

  const float* BiasPtr = Bias;

  const std::byte* QuantBDataColPtr = QuantBData;
  const float* QuantBScaleColPtr = QuantBScale;
  const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

  float* SumPtr = CRowPtr;

  int64_t nblk = static_cast<int64_t>(CountN) - NCols4;

  while (nblk >= 0) {
    ComputeDotProducts_BlkLen16_CompFp32_avx2<NCols4, HasZeroPoint>(
      BlkLen16,
      ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
      StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
      BiasPtr
    );

    // move to next `NCols` columns

    QuantBDataColPtr += NCols4 * StrideQuantBData;
    QuantBScaleColPtr += NCols4 * StrideQuantBScale;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointColPtr += NCols4 * StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
    SumPtr += NCols4;

    nblk -= NCols4;
  }

  // left over columns less than `NCols`?
  nblk += NCols4;
  for (int64_t n = 0; n < nblk; ++n) {
    ComputeDotProducts_BlkLen16_CompFp32_avx2<1, HasZeroPoint>(
      BlkLen16,
      ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
      StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
      BiasPtr
    );

    // move to next column

    QuantBDataColPtr += StrideQuantBData;
    QuantBScaleColPtr += StrideQuantBScale;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointColPtr += StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? 1 : 0;
    SumPtr += 1;
  }
}

// TODO: flow MlasQ4GemmKernelBlkLen32PlusAvx512f to improve perf
template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkLen32Plus_CompFp32_avx2(
  size_t BlkLen,
  const float* ARowPtr,
  const std::byte* QuantBDataColPtr,
  const float* QuantBScaleColPtr,
  const std::byte* QuantBZeroPointColPtr,
  float* sum_ptr,
  size_t CountK,
  size_t StrideQuantBData,
  size_t StrideQuantBScale,
  size_t StrideQuantBZeroPoint,
  const float* bias_ptr
)
{
  if constexpr (!HasZeroPoint) {
    // Suppress unused variable warnings
    (void)QuantBZeroPointColPtr;
    (void)StrideQuantBZeroPoint;
  }

  constexpr size_t BlkBitWidth4 = 4;
  constexpr size_t SubBlkLen32 = 32;
  constexpr size_t SubBlkStep16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, SubBlkLen32);
  static_assert(SubBlkStep16 == 16);  // 32 * 4 / 8

  __m256i lowMask = _mm256_set1_epi8(0x0F);

  __m256 acc[NCols];
  UnrolledLoop<NCols>([&](size_t i) {
    acc[i] = _mm256_setzero_ps();
    });

  const std::byte* b_blk_data_ptr = QuantBDataColPtr;
  const float* s = QuantBScaleColPtr;

  [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
  // only used if HasZeroPoint == true

  for (size_t k = 0; k < CountK; k += BlkLen) {
    size_t ck = std::min(CountK - k, BlkLen);

    float scale_v[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      scale_v[i] = *(s + StrideQuantBScale * i);
      });

    std::byte* b_blk_data_col_ptr[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      b_blk_data_col_ptr[i] = (std::byte*)(b_blk_data_ptr + StrideQuantBData * i);
      });

    [[maybe_unused]] uint8_t offset[NCols];
    // not ready for "Manual conversion to float" in neon yet.
    if constexpr (HasZeroPoint) {
      UnrolledLoop<NCols>([&](size_t i) {
        const std::byte zp_packed =
          QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
          ? (zp_packed >> 4)
          : (zp_packed & std::byte{ 0x0F });
        offset[i] = std::to_integer<uint8_t>(zp);
        });
    }

    for (size_t kk = 0; kk < ck; kk += SubBlkLen32) {
      int kklen = std::min((int)SubBlkLen32, (int)(ck - kk));

      // Load 4 float8 from A
      int n_to_read = std::min(kklen, 8);
      __m256 av0_8_ps = load_float_n_avx2(ARowPtr + k + kk, n_to_read);

      n_to_read = std::min(kklen - 8, 8);
      __m256 av1_8_ps = load_float_n_avx2(ARowPtr + k + kk + 8, n_to_read);

      n_to_read = std::min(kklen - 16, 8);
      __m256 av2_8_ps = load_float_n_avx2(ARowPtr + k + kk + 16, n_to_read);

      n_to_read = std::min(kklen - 24, 8);
      __m256 av3_8_ps = load_float_n_avx2(ARowPtr + k + kk + 24, n_to_read);

      UnrolledLoop<NCols>([&](size_t i) {
        // SubBlkLen = 16: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
        // SubBlkLen = 32: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
        // Load B col vectors. get SubBlkLen32 4b quantized weights from each column
        __m128i bvi4 = _mm_loadu_si128((const __m128i*)(b_blk_data_col_ptr[i]));
        b_blk_data_col_ptr[i] += SubBlkStep16;

        __m256i bv_32_epi8 = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
        bv_32_epi8 = _mm256_and_si256(lowMask, bv_32_epi8);

        // Subtract zero-point from the integers
        if constexpr (HasZeroPoint) {
          // Subtract zero-point from the integers
          __m256i zp = _mm256_set1_epi8(offset[i]);
          bv_32_epi8 = _mm256_sub_epi8(bv_32_epi8, zp);
        } else {
          // Subtract 8 from the integers
          const __m256i eight = _mm256_set1_epi8(8);
          bv_32_epi8 = _mm256_sub_epi8(bv_32_epi8, eight);
        }

        // Convert to 16 float32
        const __m256i bv0_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv_32_epi8, 0));
        const __m256i bv1_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv_32_epi8, 1));

        __m256 bv0_8_ps =
          _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bv0_16_epi16, 0)));
        __m256 bv1_8_ps =
          _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bv0_16_epi16, 1)));
        __m256 bv2_8_ps =
          _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bv1_16_epi16, 0)));
        __m256 bv3_8_ps =
          _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bv1_16_epi16, 1)));

        // multiply by scale
        __m256 scale_ps = _mm256_set1_ps(scale_v[i]);
        bv0_8_ps = _mm256_mul_ps(bv0_8_ps, scale_ps);
        bv1_8_ps = _mm256_mul_ps(bv1_8_ps, scale_ps);
        bv2_8_ps = _mm256_mul_ps(bv2_8_ps, scale_ps);
        bv3_8_ps = _mm256_mul_ps(bv3_8_ps, scale_ps);

        // c[m,n] += a[m,k] * b[k,n]
        acc[i] = _mm256_fmadd_ps(bv0_8_ps, av0_8_ps, acc[i]);
        acc[i] = _mm256_fmadd_ps(bv1_8_ps, av1_8_ps, acc[i]);
        acc[i] = _mm256_fmadd_ps(bv2_8_ps, av2_8_ps, acc[i]);
        acc[i] = _mm256_fmadd_ps(bv3_8_ps, av3_8_ps, acc[i]);
        });
    } // kk

    b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    s++;

    if constexpr (HasZeroPoint) {
      QuantBZeroPointIdx += 1;
    }
  } // k

  if constexpr (NCols == 4) {
    __m128 acc_x = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
    if (bias_ptr != nullptr) {
      acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
    }
    _mm_storeu_ps(sum_ptr, acc_x);
  } else {
    UnrolledLoop<NCols>([&](size_t i) {
      __m128 vlow = _mm256_castps256_ps128(acc[i]);
      __m128 vhigh = _mm256_extractf128_ps(acc[i], 1); // Extract high 128 bit

      // Add the two 128-bit vectors together
      __m128 vsum = _mm_add_ps(vlow, vhigh);
      // Horizontally add the elements of the resulting 128-bit vector
      vsum = _mm_hadd_ps(vsum, vsum);
      vsum = _mm_hadd_ps(vsum, vsum);

      _mm_store_ss(&sum_ptr[i], vsum);
      sum_ptr[i] += bias_ptr == nullptr ? 0.0f : bias_ptr[i];
      });
  }
}

// TODO: flow MlasQ4GemmKernelBlkLen16Avx512f to improve perf
template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_avx2(
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
  constexpr size_t BlkBitWidth4 = 4;
  constexpr size_t NCols4 = 4;

  const float* ARowPtr = A;
  float* CRowPtr = C;

  const size_t BlockCountK = BlockStrideQuantB;

  const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
  const size_t StrideQuantBScale = BlockCountK;
  const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

  const float* BiasPtr = Bias;

  const std::byte* QuantBDataColPtr = QuantBData;
  const float* QuantBScaleColPtr = QuantBScale;
  const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

  float* SumPtr = CRowPtr;

  int64_t nblk = static_cast<int64_t>(CountN) - NCols4;

  while (nblk >= 0) {
    ComputeDotProducts_BlkLen32Plus_CompFp32_avx2<NCols4, HasZeroPoint>(
      BlkLen,
      ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
      StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
      BiasPtr
    );

    // move to next `NCols` columns

    QuantBDataColPtr += NCols4 * StrideQuantBData;
    QuantBScaleColPtr += NCols4 * StrideQuantBScale;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointColPtr += NCols4 * StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
    SumPtr += NCols4;

    nblk -= NCols4;
  }

  // left over columns less than `NCols`?
  nblk += NCols4;
  for (int64_t n = 0; n < nblk; ++n) {
    ComputeDotProducts_BlkLen32Plus_CompFp32_avx2<1, HasZeroPoint>(
      BlkLen,
      ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
      StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
      BiasPtr
    );

    // move to next column

    QuantBDataColPtr += StrideQuantBData;
    QuantBScaleColPtr += StrideQuantBScale;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointColPtr += StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? 1 : 0;
    SumPtr += 1;
  }
}

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32_avx2(
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
) {
  if (BlkLen == 16) {
    if (QuantBZeroPoint != nullptr) {
      SQ4BitGemmM1Kernel_BlkLen16_CompFp32_avx2<true>(
        A,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
      );
    } else {
      SQ4BitGemmM1Kernel_BlkLen16_CompFp32_avx2<false>(
        A,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
      );
    }
  } else {
    if (QuantBZeroPoint != nullptr) {
      SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_avx2<true>(
        BlkLen,
        A,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
      );
    } else {
      SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_avx2<false>(
        BlkLen,
        A,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
      );
    }
  }
}

MLAS_FORCEINLINE __m128i
convert_2_ps_to_epi8(__m256 v0, __m256 v1)
{
  __m256i v0_8_epi32 = _mm256_cvtps_epi32(v0);
  __m256i v1_8_epi32 = _mm256_cvtps_epi32(v1);

  __m128i v0_8_epi16 = _mm_packs_epi32(_mm256_extractf128_si256(v0_8_epi32, 0), _mm256_extractf128_si256(v0_8_epi32, 1));
  __m128i v1_8_epi16 = _mm_packs_epi32(_mm256_extractf128_si256(v1_8_epi32, 0), _mm256_extractf128_si256(v1_8_epi32, 1));

  return _mm_packs_epi16(v0_8_epi16, v1_8_epi16);
}

void MLASCALL
QuantizeARow_CompInt8_avx2(
  size_t BlkLen,
  const float* A,
  size_t CountK,
  std::byte* QuantA
) {
  // port from MlasQ80BlkQuantRow
  assert(BlkLen % 16 == 0);
  const __m256 signBit = _mm256_set1_ps(-0.0f);
  int8_t* blob = reinterpret_cast<int8_t*>(QuantA);
  for (size_t k = 0; k < CountK; k += BlkLen) {
    const size_t step = std::min(BlkLen, CountK - k);

    __m256 maxAbs = _mm256_setzero_ps();
    for (size_t kk = 0; kk < step; kk += 8) {
      const int klen = std::min(8, (int)(step - kk));

      __m256 v0 = load_float_n_avx2(A + k + kk, klen);

      // Compute max(abs(e)) for the block
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v0));
    }

    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_shuffle_ps(max4, max4, 1));
    const float maxScalar = _mm_cvtss_f32(max4);

    // Quantize these floats
    const float scale = maxScalar / 127.f;
    *reinterpret_cast<float*>(blob) = scale;
    blob += sizeof(float);

    const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
    const __m256 mul = _mm256_set1_ps(inverse_scale);
    __m128i* dst = reinterpret_cast<__m128i*>(blob);

    for (size_t kk = 0; kk < step; kk += 16) {
      const int klen = std::min(16, (int)(step - kk));

      int n_to_read = std::min(klen, 8);
      __m256 v0 = load_float_n_avx2(A + k + kk, n_to_read);
      v0 = _mm256_mul_ps(v0, mul);
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);

      __m256 v1;
      n_to_read = std::min(klen - 8, 8);
      if (n_to_read <= 0)
      {
        v1 = _mm256_setzero_ps();
      } else {
        v1 = load_float_n_avx2(A + k + kk + 8, n_to_read);
        v1 = _mm256_mul_ps(v1, mul);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      }

      __m128i i_8 = convert_2_ps_to_epi8(v0, v1);
      _mm_storeu_si128(dst++, i_8);
    }
    if (step < BlkLen) {
      memset(blob + step, 0, BlkLen - step);
    }
    blob += BlkLen;
  }
}

//
// Kernel dispatch structure definition.
//
const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx2 = []() {
  MLAS_SQNBIT_GEMM_DISPATCH d;

  d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
  d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;

  d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32_avx2;
  d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32_avx2;

  d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8_avx2;
  d.QuantizeARow_CompInt8 = QuantizeARow_CompInt8_avx2;

  return d;
  }();
