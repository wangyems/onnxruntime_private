// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

#include "core/common/common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

#ifdef USE_HIPBLASLT

typedef enum _ActivationType {
  NONE = 0,
  RELU = 1,
  GELU = 2,
} ActivationType;

template <typename T>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const T*) {
  static_assert(sizeof(T) == 0, "Unsupported data type for hipBLASLt operation.");
  // Compiler will complain if we don't return something.
  return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const float*) {
  return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const half*) {
  return HIPBLAS_R_16F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const BFloat16*) {
  return HIPBLAS_R_16B;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const double*) {
  return HIPBLAS_R_64F;
}

template <typename T, typename ParamsT>
Status HipBlasLtMatMul(const ParamsT* params, int64_t batch, ActivationType activation_type = ActivationType::NONE,
                       bool enable_bias = false, const T* d_bias = nullptr,
                       bool enable_scaleD = false, const T* d_scaleD = nullptr) {
  hipblasLtHandle_t handle;
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtCreate(&handle));

  // Note: properties of original matrices A and B are swapped.
  int64_t lda = (params->opb == BlasOp::N) ? params->n : params->k;
  int64_t ldb = (params->opa == BlasOp::N) ? params->k : params->m;
  int64_t ldc = params->n;
  int64_t stride_a = params->n * params->k;
  int64_t stride_b = params->k * params->m;
  int64_t stride_c = params->n * params->m;
  float alpha = static_cast<float>(params->alpha);
  float beta = static_cast<float>(params->beta);
  int row_a, col_a, row_b, col_b, row_c, col_c;
  row_a = lda;
  col_a = (params->opb == BlasOp::N) ? params->k : params->n;
  row_b = ldb;
  col_b = (params->opa == BlasOp::N) ? params->m : params->k;
  row_c = ldc;
  col_c = params->m;

  hipblasDatatype_t in_out_datatype = HipBlasDataTypeFor(params->a);
  hipblasLtMatrixLayout_t mat_a, mat_b, mat_c;
  hipblasLtMatmulDesc_t matmul;
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutCreate(&mat_a, in_out_datatype, row_a, col_a, lda));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutCreate(&mat_b, in_out_datatype, row_b, col_b, ldb));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutCreate(&mat_c, in_out_datatype, row_c, col_c, ldc));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLASLT_COMPUTE_F32, HIPBLAS_R_32F));

  if (batch > 1) {
    HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
        mat_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
        mat_a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
        mat_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
        mat_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
        mat_c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
        mat_c, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
  }

  hipblasOperation_t trans_a = (params->opb == BlasOp::N) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t trans_b = (params->opa == BlasOp::N) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

  hipblasLtEpilogue_t epilogue;
  switch (activation_type) {
    case ActivationType::NONE:
      epilogue = enable_bias ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
      break;
    case ActivationType::RELU:
      epilogue = enable_bias ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_RELU;
      break;
    case ActivationType::GELU:
      epilogue = enable_bias ? HIPBLASLT_EPILOGUE_GELU_BIAS : HIPBLASLT_EPILOGUE_GELU;
      break;
  }
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  if (enable_bias) {
    HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void*)));
  }
  if (enable_scaleD) {
    HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scaleD, sizeof(void*)));
  }

  hipblasLtMatmulPreference_t pref;
  void* workspace;
  size_t max_workspace_size = 32 * 1024 * 1024;
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size)));

  const int heuristic_result_count = 3;
  hipblasLtMatmulHeuristicResult_t heuristic_result[heuristic_result_count] = {0};
  int ret_algo_count = 0;
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                            matmul,
                                                            mat_a,
                                                            mat_b,
                                                            mat_c,
                                                            mat_c,
                                                            pref,
                                                            heuristic_result_count,
                                                            heuristic_result,
                                                            &ret_algo_count));

  size_t workspace_size = heuristic_result[0].workspaceSize;
  if (workspace_size > 0) {
    HIP_RETURN_IF_ERROR(hipMallocAsync(&workspace, workspace_size, params->stream));
  }

  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmul(handle,
                                            matmul,
                                            &alpha,
                                            params->b,
                                            mat_a,
                                            params->a,
                                            mat_b,
                                            &beta,
                                            params->c,
                                            mat_c,
                                            params->c,
                                            mat_c,
                                            &heuristic_result[0].algo,
                                            workspace,
                                            workspace_size,
                                            params->stream));

  if (workspace > 0) {
    HIP_RETURN_IF_ERROR(hipFreeAsync(workspace, params->stream));
  }

  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescDestroy(matmul));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutDestroy(mat_a));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutDestroy(mat_b));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutDestroy(mat_c));
  HIPBLASLT_RETURN_IF_ERROR(hipblasLtDestroy(handle));
  return Status::OK();
}

template <typename T>
Status HipBlasLtGemmOp(const GemmParams<T>* params) {
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF((std::is_same_v<T, double>), "hipBLASLt does not support double inputs");
  return HipBlasLtMatMul<T, GemmParams<T>>(params, /*batch=*/1);
}

template <typename T>
Status HipBlasLtStridedBatchedGemmOp(const StridedBatchedGemmParams<T>* params) {
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF((std::is_same_v<T, double>), "hipBLASLt does not support double inputs");
  return HipBlasLtMatMul<T, StridedBatchedGemmParams<T>>(params, params->batch);
};

#endif  // USE_HIPBLASLT

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
