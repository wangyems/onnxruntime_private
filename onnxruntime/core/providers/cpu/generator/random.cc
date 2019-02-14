/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// NOTE: License applies to the multinomial implementation only.
// Portions Copyright (c) Microsoft Corporation

#include "core/providers/cpu/generator/random.h"

// build\windows\debug\external\eigen3\unsupported\eigen\cxx11\src/Tensor/Tensor.h(76):
// warning C4554: '&': check operator precedence for possible error; use parentheses to clarify precedence
// build\windows\relwithdebinfo\eigen\src\eigen\eigen-eigen-5a0156e40feb\unsupported\eigen\cxx11\src/Tensor/TensorChipping.h(52)
// warning C4100: 'dim': unreferenced formal parameter
#ifdef _WIN32
#pragma warning(disable : 4554 4100)
#endif

#include <algorithm>
#include <chrono>
#include <random>
#include "core/util/math_cpuonly.h"
#include "core/util/eigen_common_wrapper.h"
#include "gsl/span"
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    RandomNormal,
    1,
    KernelDefBuilder().TypeConstraint("T", std::vector<MLDataType>{
                                               DataTypeImpl::GetTensorType<float>(),
                                               DataTypeImpl::GetTensorType<double>()}),
    RandomNormal);

ONNX_CPU_OPERATOR_KERNEL(
    RandomUniform,
    1,
    KernelDefBuilder().TypeConstraint("T", std::vector<MLDataType>{
                                               DataTypeImpl::GetTensorType<float>(),
                                               DataTypeImpl::GetTensorType<double>()}),
    RandomUniform);

ONNX_CPU_OPERATOR_KERNEL(
    RandomNormalLike,
    1,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::AllTensorTypes()).TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    RandomNormalLike);

ONNX_CPU_OPERATOR_KERNEL(
    RandomUniformLike,
    1,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::AllTensorTypes()).TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    RandomUniformLike);

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#multinomial
ONNX_CPU_OPERATOR_KERNEL(
    Multinomial,
    7,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()).TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Multinomial);

template <typename T, typename TDistribution>
void GenerateData(std::default_random_engine& generator, TDistribution distribution, Tensor& tensor);

static Status RandomNormalCompute(float mean, float scale, std::default_random_engine& generator, TensorProto::DataType dtype, Tensor& Y);
static Status RandomUniformCompute(float high, float low, std::default_random_engine& generator, TensorProto::DataType dtype, Tensor& Y);

// Leaving in case we need to change to this approach
//static Status CreateOutputTensorFromTensorValues(OpKernelContext* ctx, const Tensor& X,Tensor** Y);
static Status CreateOutputTensorFromTensorShape(OpKernelContext* ctx, const Tensor& X, Tensor** Y);
static TensorProto::DataType InferDataType(const Tensor& tensor);

Status RandomNormal::Compute(OpKernelContext* ctx) const {
  Tensor& Y = *ctx->Output(0, shape_);

  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  auto status = RandomNormalCompute(mean_, scale_, generator_, dtype_, Y);

  return status;
}

Status RandomUniform::Compute(OpKernelContext* ctx) const {
  Tensor& Y = *ctx->Output(0, shape_);

  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  auto status = RandomUniformCompute(low_, high_, generator_, dtype_, Y);

  return status;
}

Status RandomNormalLike::Compute(OpKernelContext* ctx) const {
  const Tensor* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  Tensor* Y = nullptr;

  auto status = CreateOutputTensorFromTensorShape(ctx, X, &Y);
  ORT_RETURN_IF_ERROR(status);

  auto dtype = dtype_ != TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(X);

  if (dtype == TensorProto_DataType_UNDEFINED)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Could not infer data type from input tensor with data type ",
                           X.DataType());

  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  status = RandomNormalCompute(mean_, scale_, generator_, dtype, *Y);

  return status;
}

Status RandomUniformLike::Compute(OpKernelContext* ctx) const {
  const Tensor* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  Tensor* Y = nullptr;

  auto status = CreateOutputTensorFromTensorShape(ctx, X, &Y);
  ORT_RETURN_IF_ERROR(status);

  auto dtype = dtype_ != TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(X);

  if (dtype == TensorProto_DataType_UNDEFINED)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Could not infer data type from input tensor with data type ",
                           X.DataType());
  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  status = RandomUniformCompute(low_, high_, generator_, dtype, *Y);

  return status;
}

// Rank-2 tensor (matrix) of scalar type T.
template <typename T, typename IndexType = int64_t>
using Matrix = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>>;

template <typename T, typename IndexType = int64_t>
using ConstMatrix = Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>>;

template <typename T, typename IndexType = int64_t>
using EigenVector = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>>;

template <typename OutputType>
static Status MultinomialCompute(OpKernelContext* ctx,
                                 const Tensor& X,
                                 const int64_t batch_size,
                                 const int64_t num_classes,
                                 const int64_t num_samples,
                                 std::default_random_engine& generator,
                                 Tensor& Y) {
  // implementation copied from Tensorflow with some changes such as using the std::uniform_real_distribution
  // instead of the Philox RNG.
  Eigen::array<int64_t, 2> X_dims = {{batch_size, num_classes}};
  ConstMatrix<float> logits = ConstMatrix<float>(X.template Data<float>(), X_dims);

  Eigen::array<int64_t, 2> Y_dims = {{batch_size, num_samples}};
  Matrix<OutputType> output = Matrix<OutputType>(Y.template MutableData<OutputType>(), Y_dims);

  // TODO (perf optimization) - the idea behind making this a lambda is so that we can parallelize across batches.
  // When we do that this lamdba will act as one task given to a thread
  auto DoWork = [ctx, num_samples, num_classes, &generator, &logits, &output](int64_t start_row,
                                                                              int64_t limit_row) {
    std::default_random_engine generator_copy = generator;
    // BEGIN create temporary tensor
    AllocatorPtr alloc;
    ctx->GetTempSpaceAllocator(&alloc);
    auto cdf_data = static_cast<double*>(alloc->Alloc(sizeof(double) * num_classes));
    BufferUniquePtr cdf_buffer(cdf_data, BufferDeleter(alloc));
    Eigen::array<int64_t, 1> cdf_dims = {{num_classes}};
    auto cdf = EigenVector<double>(cdf_data, cdf_dims);
    // END create temporary tensor

    std::uniform_real_distribution<double> dist(0.0, 1.0);  // TODO: should this be initialized per batch?
    for (int64_t b = start_row; b < limit_row; ++b) {
      const float* logits_row = &(logits(b, 0));
      // Takes an along-class maximum (for numerical stability).
      float maxx = std::numeric_limits<float>::lowest();
      for (int64_t j = 0; j < num_classes; ++j) {
        if (Eigen::numext::isfinite(logits_row[j])) {
          maxx = std::max(maxx, logits_row[j]);
        }
      }
      const double max_logit = static_cast<double>(maxx);

      // Precompute cumulative probability distribution across classes.
      // Note: This isn't normalized.
      cdf = (logits.chip<0>(b).cast<double>() - max_logit).exp();
      double running_total = 0;
      for (int64_t j = 0; j < num_classes; ++j) {
        if (Eigen::numext::isfinite(logits_row[j])) {
          running_total += cdf(j);
        }
        cdf(j) = running_total;
      }
      // Generate each sample.
      const double* cdf_begin = cdf.data();
      const double* cdf_end = cdf.data() + num_classes;
      for (int64_t j = 0; j < num_samples; ++j) {
        const double to_find = dist(generator_copy) * running_total;
        auto found_iter = std::upper_bound(cdf_begin, cdf_end, to_find);
        output(b, j) = static_cast<OutputType>(std::distance(cdf_begin, found_iter));
      }
    }
  };
  DoWork(0, batch_size);
  return Status::OK();
}

Status Multinomial::Compute(OpKernelContext* ctx) const {
  const Tensor* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  auto& X_dims = X.Shape().GetDims();

  if (X_dims.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Empty dimensions for input tensor");
  }

  const auto batch_size = X_dims[0];
  const auto num_classes = X_dims[1];

  // validate inputs
  if (batch_size < 1) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "batch_size is < 1");
  }
  if (num_classes < 1) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "num_classes is < 1");
  }
  if (num_samples_ < 1) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "num_samples is < 1");
  }

  Tensor* Y = ctx->Output(0, TensorShape({batch_size, num_samples_}));

  Status status = Status::OK();
  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  switch (output_dtype_) {
    case TensorProto::INT32: {
      status = MultinomialCompute<int32_t>(ctx, X, batch_size, num_classes, num_samples_, generator_, *Y);
      break;
    }
    case TensorProto::INT64: {
      status = MultinomialCompute<int64_t>(ctx, X, batch_size, num_classes, num_samples_, generator_, *Y);
      break;
    }
    default:
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid data type of ", output_dtype_);
  }

  return status;
}

/*
alternative interpretation of the spec is that the input tensor contains the dimensions as ints.
Keeping this temporarily in case we go back to that.

// read shape information from input tensor and create output tensor with it
static Status CreateOutputTensorFromTensorValues(OpKernelContext* ctx, const Tensor& X, Tensor** Y) {
  const TensorShape& shape = X.Shape();
  auto size = shape.Size();
  auto num_dims = shape.NumDimensions();

  if (num_dims != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expected 1 dimension tensor with shape information. Dimensions=", num_dims);
  }

  std::vector<int64_t> dims;
  dims.reserve(shape.Size());

  auto data = gsl::make_span(tensor.template Data<int64_t>(), shape.Size());
  dims.insert(dims.cbegin(), data.cbegin(), data.cend());

  *Y = ctx->Output(0, TensorShape(dims));

  return Status::OK();
}
*/

// create output tensor using shape of input tensor
static Status CreateOutputTensorFromTensorShape(OpKernelContext* ctx, const Tensor& X, Tensor** Y) {
  const TensorShape& shape = X.Shape();

  *Y = ctx->Output(0, shape);

  return Status::OK();
}

static TensorProto::DataType InferDataType(const Tensor& tensor) {
  auto tensor_type = tensor.DataType();
  TensorProto::DataType dtype = TensorProto_DataType_UNDEFINED;

  if (tensor_type == DataTypeImpl::GetType<float>())
    dtype = TensorProto_DataType_FLOAT;
  else if (tensor_type == DataTypeImpl::GetType<double>())
    dtype = TensorProto_DataType_DOUBLE;
  else {
    // unsupported. return UNDEFINED
  }

  return dtype;
}

static Status RandomNormalCompute(float mean, float scale,
                                  std::default_random_engine& generator,
                                  TensorProto::DataType dtype, Tensor& Y) {
  switch (dtype) {
    case TensorProto::FLOAT: {
      GenerateData<float, std::normal_distribution<float>>(
          generator, std::normal_distribution<float>{mean, scale}, Y);
      break;
    }
    case TensorProto::FLOAT16: {
      ORT_NOT_IMPLEMENTED("FLOAT16 is not supported");
    }
    case TensorProto::DOUBLE: {
      GenerateData<double, std::normal_distribution<double>>(
          generator, std::normal_distribution<double>{mean, scale}, Y);
      break;
    }
    default:
      ORT_THROW("Invalid data type of ", dtype);
  }

  return Status::OK();
}

static Status RandomUniformCompute(float low, float high,
                                   std::default_random_engine& generator,
                                   TensorProto::DataType dtype,
                                   Tensor& Y) {
  switch (dtype) {
    case TensorProto::FLOAT: {
      GenerateData<float, std::uniform_real_distribution<float>>(
          generator, std::uniform_real_distribution<float>{low, high}, Y);
      break;
    }
    case TensorProto::FLOAT16: {
      ORT_NOT_IMPLEMENTED("FLOAT16 is not supported");
    }
    case TensorProto::DOUBLE: {
      GenerateData<double, std::uniform_real_distribution<double>>(
          generator, std::uniform_real_distribution<double>{low, high}, Y);
      break;
    }
    default:
      ORT_THROW("Invalid data type of ", dtype);
  }

  return Status::OK();
}

template <typename T, typename TDistribution>
void GenerateData(std::default_random_engine& generator, TDistribution distribution, Tensor& tensor) {
  auto out = gsl::make_span(tensor.template MutableData<T>(), tensor.Shape().Size());

  std::for_each(out.begin(), out.end(), [&generator, &distribution](T& value) { value = distribution(generator); });
}

}  // namespace onnxruntime
