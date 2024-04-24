// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"

#include <utility>
#include "core/common/safeint.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/framework/utils.h"

namespace onnxruntime {

#ifdef ENABLE_STRIDED_TENSORS
namespace {
int64_t GetSizeFromStrides(const TensorShape& shape, gsl::span<const int64_t> strides) {
  SafeInt<int64_t> size = 1;
  for (size_t dim = 0; dim < shape.NumDimensions(); ++dim) {
    if (shape[dim] == 0) {
      size = 0;
      break;
    }
    size += strides[dim] * (shape[dim] - 1);
  }
  return size;
}
}  // namespace
#endif

size_t Tensor::CalculateTensorStorageSize(MLDataType elt_type, const TensorShape& shape) {
  int64_t shape_size = shape.Size();
  if (shape_size < 0)
    ORT_THROW("shape.Size() must >=0");

  if (shape_size > 0) {
    SafeInt<size_t> len = 0;

    // TODO(adrianlizarraga): Handle more cleanly.
    if (utils::IsPrimitiveDataType<Int4x2>(elt_type) || utils::IsPrimitiveDataType<UInt4x2>(elt_type)) {
      shape_size = (shape_size + 1) / 2;
    }

    if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(shape_size), elt_type->Size(), &len))
      ORT_THROW("tensor failed memory size calculation");

    return len;
  }

  return 0;
}

Tensor::Tensor(MLDataType elt_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& location,
               ptrdiff_t offset, gsl::span<const int64_t> strides)
    : alloc_info_(location) {
  ORT_ENFORCE(elt_type != nullptr);
  Init(elt_type, shape, p_data, nullptr, offset, strides);
}

Tensor::Tensor(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator)
    : alloc_info_(allocator->Info()) {
  ORT_ENFORCE(elt_type != nullptr);
  size_t len = Tensor::CalculateTensorStorageSize(elt_type, shape);

  void* p_data = nullptr;
  if (len > 0) {
    p_data = allocator->Alloc(len);
  }
  Init(elt_type, shape, p_data, allocator, 0L);
}

Tensor::Tensor(MLDataType elt_type, const TensorShape& shape, void* p_data, std::shared_ptr<IAllocator> deleter,
               ptrdiff_t offset, gsl::span<const int64_t> strides)
    : alloc_info_(deleter->Info()) {
  ORT_ENFORCE(elt_type != nullptr);
  Init(elt_type, shape, p_data, deleter, offset, strides);
}

void Tensor::InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator,
                          OrtValue& ort_value) {
  auto p_tensor = std::make_unique<Tensor>(elt_type, shape, std::move(allocator));
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

void Tensor::InitOrtValue(MLDataType elt_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& location,
                          OrtValue& ort_value, ptrdiff_t offset, gsl::span<const int64_t> strides) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = std::make_unique<Tensor>(elt_type, shape, p_data, location, offset, strides);
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

void Tensor::InitOrtValue(MLDataType elt_type, const TensorShape& shape,
                          void* p_data, std::shared_ptr<IAllocator> allocator,
                          OrtValue& ort_value, ptrdiff_t offset,
                          gsl::span<const int64_t> strides) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = std::make_unique<Tensor>(elt_type, shape, p_data, std::move(allocator), offset, strides);
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

void Tensor::InitOrtValue(Tensor&& tensor, OrtValue& ort_value) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = std::make_unique<Tensor>(std::move(tensor));
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

size_t Tensor::SizeInBytes() const {
#ifdef ENABLE_STRIDED_TENSORS
  int64_t size = IsContiguous() ? shape_.Size() : GetSizeFromStrides(shape_, strides_);
#else
  int64_t size = shape_.Size();
#endif
  size_t ret = 0;

  // TODO(adrianlizarraga): Handle more cleanly.
  if (utils::IsPrimitiveDataType<Int4x2>(dtype_) || utils::IsPrimitiveDataType<UInt4x2>(dtype_)) {
    size = (size + 1) / 2;
  }

  if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(size), dtype_->Size(), &ret)) {
    ORT_THROW("tensor size overflow");
  }
  return ret;
}

void Tensor::Init(MLDataType elt_type, const TensorShape& shape, void* p_raw_data, AllocatorPtr deleter,
                  ptrdiff_t offset, gsl::span<const int64_t> strides) {
  int64_t shape_size = shape.Size();
  if (shape_size < 0)
    ORT_THROW("shape.Size() must >=0");

  dtype_ = elt_type->AsPrimitiveDataType();
  ORT_ENFORCE(dtype_ != nullptr,
              "Tensor is expected to contain one of the primitive data types. Got: ",
              DataTypeImpl::ToString(elt_type));
  shape_ = shape;
  p_data_ = p_raw_data;
  // if caller passed in a deleter we now own p_data_ and must free it in the dtor
  buffer_deleter_ = std::move(deleter);
  // for string tensors, if this tensor own the buffer (caller passed in the deleter)
  // do the placement new for strings on pre-allocated buffer.
  if (buffer_deleter_ && IsDataTypeString()) {
    utils::ConstructStrings(p_data_, shape_size);
  }

  byte_offset_ = offset;

#ifdef ENABLE_STRIDED_TENSORS
  if (shape.NumDimensions() > 0 && !strides.empty()) {
    ORT_ENFORCE(shape.NumDimensions() == strides.size(), "Length of strides doesn't match tensor dimension size.");
    strides_.assign(strides.begin(), strides.end());
    is_contiguous_ = CheckIsContiguous();
  }
#else
  ORT_UNUSED_PARAMETER(strides);
#endif
}

Tensor::Tensor(Tensor&& other) noexcept
    : p_data_(other.p_data_),
      buffer_deleter_(other.buffer_deleter_),
      shape_(other.shape_),
#ifdef ENABLE_STRIDED_TENSORS
      strides_(other.strides_),
      is_contiguous_(other.is_contiguous_),
#endif
      dtype_(other.dtype_),
      alloc_info_(other.alloc_info_),
      byte_offset_(other.byte_offset_) {
  other.p_data_ = nullptr;
  other.buffer_deleter_ = nullptr;
  other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
  other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
#ifdef ENABLE_STRIDED_TENSORS
  other.strides_ = {};
  other.is_contiguous_ = true;
#endif
  other.byte_offset_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    ReleaseBuffer();

    p_data_ = other.p_data_;
    buffer_deleter_ = other.buffer_deleter_;
    shape_ = other.shape_;
#ifdef ENABLE_STRIDED_TENSORS
    strides_ = other.strides_;
    is_contiguous_ = other.is_contiguous_;
#endif
    dtype_ = other.dtype_;
    alloc_info_ = other.alloc_info_;
    byte_offset_ = other.byte_offset_;

    other.p_data_ = nullptr;
    other.buffer_deleter_ = nullptr;
    other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
#ifdef ENABLE_STRIDED_TENSORS
    other.strides_ = {};
    other.is_contiguous_ = true;
#endif
    other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
    other.byte_offset_ = 0;
  }

  return *this;
}

Tensor::~Tensor() {
  ReleaseBuffer();
}

void Tensor::ReleaseBuffer() {
  if (buffer_deleter_) {
    if (IsDataTypeString()) {
      utils::DestroyStrings(p_data_, shape_.Size());
    }
    buffer_deleter_->Free(p_data_);
  }
}

#ifdef ENABLE_STRIDED_TENSORS
bool Tensor::CheckIsContiguous() const {
  if (strides_.empty()) {
    return true;
  }

  int64_t running_size = 1;
  for (size_t i = shape_.NumDimensions(); i > 0; --i) {
    size_t j = i - 1;
    if (shape_[j] == 0) {
      return true;
    }

    if (shape_[j] != 1 && strides_[j] != running_size) {
      return false;
    }

    running_size *= shape_[j];
  }

  return true;
}

gsl::span<const int64_t> Tensor::Strides() const {
  if (shape_.NumDimensions() == 0) {
    return {};
  }

  if (strides_.empty()) {
    strides_.resize(shape_.NumDimensions());
    int64_t running_size = 1;
    for (size_t i = shape_.NumDimensions(); i > 0; --i) {
      strides_[i - 1] = running_size;
      running_size *= shape_[i - 1];
    }
  }

  return gsl::make_span(strides_);
}

void Tensor::SetShapeAndStrides(const TensorShape& new_shape, gsl::span<const int64_t> new_strides) {
  ORT_ENFORCE(new_shape.NumDimensions() == new_strides.size(),
              "Length of strides doesn't match with tensor dimension size.");
  shape_ = new_shape;
  strides_ = ToShapeVector(new_strides);
  is_contiguous_ = CheckIsContiguous();
}
#endif

}  // namespace onnxruntime
