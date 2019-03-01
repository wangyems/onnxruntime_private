// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensor_external_data_info.h"
#include "core/common/common.h"
#include "path_lib.h"

#ifdef _WIN32
#include <Windows.h>
#endif
using ::google::protobuf::RepeatedPtrField;
using ::ONNX_NAMESPACE::StringStringEntryProto;

namespace onnxruntime {
Status ExternalDataInfo::Create(const RepeatedPtrField<StringStringEntryProto>& input,
                                std::unique_ptr<ExternalDataInfo>& out) {
  out = std::make_unique<ExternalDataInfo>();
  const int input_size = input.size();
  for (int i = 0; i != input_size; ++i) {
    StringStringEntryProto stringmap = input[i];
    if (!stringmap.has_key())
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Need a key for the external data info");
    if (!stringmap.has_value())
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Need a value for the external data info");
    if (stringmap.key() == "location" && !stringmap.value().empty()) {
#ifndef _WIN32
      out->rel_path_ = stringmap.value();
#else
      const std::string& s = stringmap.value();
      if (s.size() >= std::numeric_limits<int>::max()) throw std::runtime_error("length overflow");
      const int src_len = static_cast<int>(s.size() + 1);
      const int len = MultiByteToWideChar(CP_ACP, 0, s.data(), src_len, nullptr, 0);
      assert(len > 0);
      std::wstring ret(static_cast<size_t>(len) - 1, '\0');
      const int r = MultiByteToWideChar(CP_ACP, 0, s.data(), src_len, (wchar_t*)ret.data(), len);
      assert(len == r);
      out->rel_path_ = ret;
#endif
    } else if (stringmap.key() == "offset" && !stringmap.value().empty()) {
      char* end;
      out->offset_ = OrtStrtoPtrDiff(stringmap.value().c_str(), &end, 10);
      if (end != stringmap.value().c_str() + stringmap.value().length())
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parsing ", stringmap.value(), " failed");
    } else if (stringmap.key() == "length" && !stringmap.value().empty()) {
      char* end;
      out->length_ = OrtStrtoPtrDiff(stringmap.value().c_str(), &end, 10);
      if (end != stringmap.value().c_str() + stringmap.value().length())
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parsing ", stringmap.value(), " failed");
    } else if (stringmap.key() == "checksum" && !stringmap.value().empty()) {
      out->checksum_ = stringmap.value();
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error!");
    }
  }
  if (out->rel_path_.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Missing 'location'");
  }
  return Status::OK();
}
}  // namespace onnxruntime