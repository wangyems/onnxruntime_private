// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include <pybind11/pybind11.h>

#include "core/providers/rocm/rocm_common.h"
#include "device_array.h"
#include "operator.h"

#include "core/providers/rocm/shared_inc/fpgeneric.h"

namespace py = pybind11;

namespace onnxruntime {

enum class BlasOp {
  N,
  T,
};

// assume row majored
template <typename T>
class GemmBase : public Operator {
 public:
  GemmBase(
      BlasOp opa, BlasOp opb,
      int64_t m, int64_t n, int64_t k,
      double alpha,
      DeviceArray& a, int64_t lda,
      DeviceArray& b, int64_t ldb,
      double beta,
      DeviceArray& c, int64_t ldc)
      : Operator(),
        opa_(opa),
        opb_(opb),
        m_(m),
        n_(n),
        k_(k),
        alpha_(alpha),
        a_(reinterpret_cast<T*>(a.ptr())),
        lda_(lda),
        b_(reinterpret_cast<T*>(b.ptr())),
        ldb_(ldb),
        beta_(beta),
        c_(reinterpret_cast<T*>(c.ptr())),
        ldc_(ldc) {}

 protected:
  BlasOp opa_;
  BlasOp opb_;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  T alpha_;
  T* a_;
  int64_t lda_;
  T* b_;
  int64_t ldb_;
  T beta_;
  T* c_;
  int64_t ldc_;
};

template <typename T>
class RocBlasGemm : public GemmBase<T> {
 public:
  RocBlasGemm(BlasOp opa, BlasOp opb,
              int64_t m, int64_t n, int64_t k,
              double alpha,
              DeviceArray& a, int64_t lda,
              DeviceArray& b, int64_t ldb,
              double beta,
              DeviceArray& c, int64_t ldc)
      : GemmBase<T>(opa, opb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        opa_(opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose),
        opb_(opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose) {
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
  }

  ~RocBlasGemm() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() {
    ROCBLAS_CALL_THROW(
        rocblasGemmHelper(this->rocblas_handle_, this->opa_, this->opb_,
                          this->m_, this->n_, this->k_,
                          &(this->alpha_),
                          this->a_, this->lda_,
                          this->b_, this->ldb_,
                          &(this->beta_),
                          this->c_, this->ldc_));
  }

 private:
  rocblas_handle rocblas_handle_;
  rocblas_operation opa_;
  rocblas_operation opb_;
};

void InitGemm(py::module mod) {
  auto blas_op = mod.def_submodule("blas_op");

  py::enum_<BlasOp>(blas_op, "BlasOp")
      .value("N", BlasOp::N, "Passthrough")
      .value("T", BlasOp::T, "Transpose")
      .export_values();

  // float
  py::class_<RocBlasGemm<float>>(mod, "RocblasGemm_float")
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, double, DeviceArray&, int64_t, DeviceArray&, int64_t, double, DeviceArray&, int64_t>())
      .def("SetRepeats", &RocBlasGemm<float>::SetRepeats)
      .def("Profile", &RocBlasGemm<float>::Profile)
      .def("Run", &RocBlasGemm<float>::Run);

  // half
  py::class_<RocBlasGemm<half>>(mod, "RocblasGemm_half")
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, double, DeviceArray&, int64_t, DeviceArray&, int64_t, double, DeviceArray&, int64_t>())
      .def("SetRepeats", &RocBlasGemm<half>::SetRepeats)
      .def("Profile", &RocBlasGemm<half>::Profile)
      .def("Run", &RocBlasGemm<half>::Run);
}

}  // namespace onnxruntime
