// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <memory>
#include <random>
#include <type_traits>
#include <variant>
#include <vector>

#include "synthetic_data_loader.h"

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

namespace {

void RandomFloats(std::vector<float>& rets, size_t num_element) {
  const float scale = 1.f;
  const float mean = 0.f;
  const float seed = 123.f;
  static std::default_random_engine generator{static_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};

  std::generate_n(std::back_inserter(rets), num_element,
                  [&distribution]() -> float { return distribution(generator); });
}

template <typename IntType>
void RandomInts(std::vector<IntType>& rets, size_t num_element, IntType low, IntType high) {
  static std::random_device rd;
  static std::mt19937 generator(rd());
  std::uniform_int_distribution<IntType> distribution(low, high);

  std::generate_n(std::back_inserter(rets), num_element,
                  [&distribution]() -> IntType { return distribution(generator); });
}

}  // namespace

template <typename T>
void SyntheticSampleBatch::AddIntInput(const std::vector<int64_t>& shape, T low, T high) {
  data_vector_.push_back(SyntheticInput(shape));

  std::vector<T> values;
  auto num_of_element = data_vector_.back().NumOfElements();
  values.reserve(num_of_element);
  RandomInts(values, num_of_element, low, high);

  SyntheticDataVector& data = data_vector_.back().GetData();
  data = values;
}

void SyntheticSampleBatch::AddInt64Input(const std::vector<int64_t>& shape, int64_t low, int64_t high) {
  AddIntInput(shape, low, high);
}

void SyntheticSampleBatch::AddInt32Input(const std::vector<int64_t>& shape, int32_t low, int32_t high) {
  AddIntInput(shape, low, high);
}

void SyntheticSampleBatch::AddBoolInput(const std::vector<int64_t>& shape) {
  // Use uint8_t to store the bool value by intention, because vector<bool> is specialized, we can not create a
  // Tensor leveraging C APIs to reuse the data buffer.
  AddIntInput(shape, static_cast<uint8_t>(0), static_cast<uint8_t>(1));
}

void SyntheticSampleBatch::AddFloatInput(const std::vector<int64_t>& shape) {
  data_vector_.push_back(SyntheticInput(shape));

  std::vector<float> values;
  auto num_of_element = data_vector_.back().NumOfElements();
  values.reserve(num_of_element);
  RandomFloats(values, num_of_element);

  SyntheticDataVector& data = data_vector_.back().GetData();
  data = values;
}

#define ORT_RETURN_ON_ERROR(expr)                              \
  do {                                                         \
    OrtStatus* onnx_status = (expr);                           \
    if (onnx_status != NULL) {                                 \
      auto code = ort_api->GetErrorCode(onnx_status);          \
      const char* msg = ort_api->GetErrorMessage(onnx_status); \
      printf("Run failed with error code :%d\n", code);        \
      printf("Error message :%s\n", msg);                      \
      ort_api->ReleaseStatus(onnx_status);                     \
      return false;                                            \
    }                                                          \
  } while (0);

bool SyntheticSampleBatch::GetBatch(std::vector<OrtValue*>& batches) {
  batches.clear();
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const auto* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  for (size_t i = 0; i < data_vector_.size(); ++i) {
    SyntheticInput& input = data_vector_[i];

    std::visit([&batches, &input, &ort_api, &memory_info](auto&& arg) -> bool {
      ONNXTensorElementDataType elem_data_type;
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
        elem_data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      } else {
        elem_data_type = Ort::TypeToTensorType<int32_t>::type;
      }

      void* p_data = arg.data();
      OrtValue* value = nullptr;
      const auto& shape_vector = input.ShapeVector();
      // Be noted: the created OrtValue won't clean the raw data after its lifetime ended.
      ORT_RETURN_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          memory_info,
          p_data, (input.NumOfElements() * sizeof(T)),
          shape_vector.data(), shape_vector.size(),
          elem_data_type,
          &value));

      batches.emplace_back(value);
      return true;
    },
               input.GetData());
  }

  return true;
}

bool SyntheticDataLoader::GetNextSampleBatch(std::vector<OrtValue*>& batches) {
  if (sample_batch_iter_index_ >= NumOfSampleBatches()) {
    return false;
  }

  auto& sample = sample_batch_collections_[sample_batch_iter_index_];
  sample.GetBatch(batches);
  sample_batch_iter_index_ += 1;
  return true;
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
