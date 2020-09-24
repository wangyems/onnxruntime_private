// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include <core/graph/basic_types.h>
#include <core/session/onnxruntime_c_api.h>

namespace flatbuffers {
class FlatBufferBuilder;

template <typename T>
struct Offset;

struct String;

template <typename T>
class Vector;
}  // namespace flatbuffers

namespace onnxruntime {

namespace experimental {

namespace fbs {
struct OperatorSetId;
struct ValueInfo;
}  // namespace fbs

namespace utils {

onnxruntime::common::Status SaveValueInfoOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::ValueInfoProto& value_info_proto,
    flatbuffers::Offset<fbs::ValueInfo>& fbs_value_info) ORT_MUST_USE_RESULT;

#if defined(ENABLE_ORT_FORMAT_LOAD)

void LoadStringFromOrtFormat(std::string& dst, const flatbuffers::String* fbs_string);

onnxruntime::common::Status LoadValueInfoOrtFormat(
    const fbs::ValueInfo& fbs_value_info, ONNX_NAMESPACE::ValueInfoProto& value_info_proto) ORT_MUST_USE_RESULT;

onnxruntime::common::Status LoadOpsetImportOrtFormat(
    const flatbuffers::Vector<flatbuffers::Offset<fbs::OperatorSetId>>* fbs_op_set_ids,
    std::unordered_map<std::string, int>& domain_to_version) ORT_MUST_USE_RESULT;

#endif

// check if filename ends in .ort
template <typename T>
bool IsOrtFormatModel(const std::basic_string<T>& filename) {
  auto len = filename.size();
  return len > 4 &&
         filename[len - 4] == '.' &&
         std::tolower(filename[len - 3]) == 'o' &&
         std::tolower(filename[len - 2]) == 'r' &&
         std::tolower(filename[len - 1]) == 't';
}

// check if bytes has the flatbuffer ORT identifier
bool IsOrtFormatModelBytes(const void* bytes, int num_bytes);

}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime
