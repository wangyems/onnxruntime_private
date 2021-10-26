// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <iosfwd>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <gsl/gsl>
#include "onnxruntime_config.h"

namespace onnxruntime {
#ifdef __GNUC__
#pragma GCC diagnostic push
#ifdef HAS_NULL_DEREFERENCE
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
#endif
class TensorShape {
  // We use negative numbers for unknown symbolic dimension. Each negative
  // number represents a unique symbolic dimension.
 public:
  TensorShape() = default;

  TensorShape(const TensorShape& other) : TensorShape(other.GetDims()) {}
  TensorShape& operator=(const TensorShape& other);

  TensorShape(TensorShape&& other) { operator=(std::move(other)); }
  TensorShape& operator=(TensorShape&& other);

  TensorShape(gsl::span<const int64_t> dims);
  TensorShape(const std::vector<int64_t>& dims) : TensorShape(gsl::make_span(dims)) {}
  TensorShape(const std::initializer_list<int64_t>& dims) : TensorShape(gsl::make_span(dims)) {}
  TensorShape(const int64_t* dimension_sizes, size_t dimension_count) : TensorShape(gsl::span<const int64_t>(dimension_sizes, dimension_count)) {}
  TensorShape(const std::vector<int64_t>& dims, size_t start, size_t end) : TensorShape(gsl::span<const int64_t>(&dims[start], end - start)) {}

  /**
     Return the dimension specified by <idx>.
  */
  const int64_t& operator[](size_t idx) const { return values_[idx]; }
  int64_t& operator[](size_t idx) { return values_[idx]; }

  bool operator==(const TensorShape& other) const noexcept { return GetDims() == other.GetDims(); }
  bool operator!=(const TensorShape& other) const noexcept { return GetDims() != other.GetDims(); }

  size_t NumDimensions() const noexcept {
    return size_;
  }

  /**
     Copy dims into an array with given size
  */
  void CopyDims(int64_t* dims, size_t num_dims) const {
    memcpy(dims, values_, sizeof(int64_t) * std::min(num_dims, NumDimensions()));
  }

  /**
     Copy dims from a specific start dim into an array with given size
     `start_dim` is expected to be in the inclusive range [0, NumDimensions() - 1]
     and this function does no checks to ensure that
  */
  void CopyDims(int64_t* dims, size_t start_dim, size_t num_dims) const {
    memcpy(dims, values_ + start_dim, sizeof(int64_t) * std::min(num_dims, NumDimensions() - start_dim));
  }

  /**
     Return underlying vector representation.
  */
  gsl::span<const int64_t> GetDims() const { return gsl::span<const int64_t>(values_, size_); }
  std::vector<int64_t> GetDimsAsVector() const { return std::vector<int64_t>(values_, values_ + size_); }

  /**
   * Return the total number of elements. Returns 1 for an empty (rank 0) TensorShape.
   *
   * May return -1
   */
  int64_t Size() const;

  /**
     Return the total number of elements up to the specified dimension.
     If the dimension interval is empty (dimension == 0), return 1.
     @param dimension Return size up to this dimension. Value must be between 0 and this->NumDimensions(), inclusive.
  */
  int64_t SizeToDimension(size_t dimension) const;

  /**
     Return the total number of elements from the specified dimension to the end of the tensor shape.
     If the dimension interval is empty (dimension == this->NumDimensions()), return 1.
     @param dimension Return size from this dimension to the end. Value must be between 0 and this->NumDimensions(),
                      inclusive.
  */
  int64_t SizeFromDimension(size_t dimension) const;

  /**
     Return a new TensorShape of the dimensions from dimstart to dimend.
  */
  TensorShape Slice(size_t dimstart, size_t dimend) const;

  /**
     Return a new TensorShape of the dimensions from dimstart to end.
  */
  TensorShape Slice(size_t dimstart) const { return Slice(dimstart, size_); }

  /**
     output dimensions nicely formatted
  */
  std::string ToString() const;

  /**
     Calculate size between start and end.
     Assumes start and end are between 0 and this->NumDimensions(), inclusive, and that
     start < end.
  */
  int64_t SizeHelper(size_t start, size_t end) const;

  /**
     empty shape or 1D shape (1) is regarded as scalar tensor
  */
  bool IsScalar() const {
    size_t len = size_;
    return len == 0 || (len == 1 && values_[0] == 1);
  }

 private:
  void Allocate(size_t size);

  int64_t* values_{};
  size_t size_{};
  int64_t small_buffer_[4];
  std::unique_ptr<int64_t[]> allocated_buffer_;
  //  std::vector<int64_t> m_vector;
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
// operator<< to nicely output to a stream
std::ostream& operator<<(std::ostream& out, const TensorShape& shape);

}  // namespace onnxruntime
