// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <random>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class RandomNormal final : public OpKernel {
 public:
  RandomNormal(const OpKernelInfo& info) : OpKernel(info) {
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("mean", &mean_).IsOK());
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("scale", &scale_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (!info.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t dtype;
    ONNXRUNTIME_ENFORCE(info.GetAttr<int64_t>("dtype", &dtype).IsOK());
    dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
    ONNXRUNTIME_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) && dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                "Invalid dtype of ", dtype_);

    std::vector<int64_t> shape;
    ONNXRUNTIME_ENFORCE(info.GetAttrs<int64_t>("shape", shape).IsOK());
    shape_ = TensorShape(shape);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float mean_;
  float scale_;
  std::default_random_engine generator_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_;
  TensorShape shape_;
};

class RandomNormalLike final : public OpKernel {
 public:
  RandomNormalLike(const OpKernelInfo& info) : OpKernel(info) {
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("mean", &mean_).IsOK());
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("scale", &scale_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (!info.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t dtype;
    if (info.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
      dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
      ONNXRUNTIME_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) && dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                  "Invalid dtype of ", dtype_);
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float mean_;
  float scale_;
  std::default_random_engine generator_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_ = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;  //optional and may be inferred
};

class RandomUniform final : public OpKernel {
 public:
  RandomUniform(const OpKernelInfo& info) : OpKernel(info) {
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("high", &high_).IsOK());
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("low", &low_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (!info.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t dtype;
    ONNXRUNTIME_ENFORCE(info.GetAttr<int64_t>("dtype", &dtype).IsOK());
    dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
    ONNXRUNTIME_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) && dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                "Invalid dtype of ", dtype_);

    std::vector<int64_t> shape;
    ONNXRUNTIME_ENFORCE(info.GetAttrs<int64_t>("shape", shape).IsOK());
    shape_ = TensorShape(shape);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float high_;
  float low_;
  std::default_random_engine generator_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_;
  TensorShape shape_;
};

class RandomUniformLike final : public OpKernel {
 public:
  RandomUniformLike(const OpKernelInfo& info) : OpKernel(info) {
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("high", &high_).IsOK());
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("low", &low_).IsOK());
    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (!info.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t dtype;
    if (info.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
      dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
      ONNXRUNTIME_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) && dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                  "Invalid dtype of ", dtype_);
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float high_;
  float low_;
  std::default_random_engine generator_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_ = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;  //optional and may be inferred
};

class Multinomial final : public OpKernel {
 public:
  Multinomial(const OpKernelInfo& info) : OpKernel(info) {
    ONNXRUNTIME_ENFORCE(info.GetAttr<int64_t>("sample_size", &num_samples_).IsOK());

    float seed = 0.f;
    if (!info.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t output_dtype_tmp;
    if (!info.GetAttr<int64_t>("dtype", &output_dtype_tmp).IsOK()) {
      output_dtype_ = ONNX_NAMESPACE::TensorProto_DataType_INT32;  // default is INT32 as per spec
    } else {
      output_dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(output_dtype_tmp);
    }
    ONNXRUNTIME_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(output_dtype_) && output_dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                "Invalid dtype of ", output_dtype_);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  int64_t num_samples_;
  std::default_random_engine generator_;
  ONNX_NAMESPACE::TensorProto::DataType output_dtype_;
};
}  // namespace onnxruntime
