// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/path_string.h"
#include "core/common/status.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/data_types.h"
#include "core/framework/framework_common.h"
#include "core/common/path.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace onnxruntime {
namespace training {

/**
 * @brief Create OrtValues From TensorProto objects
 *
 * @param tensor_protos vector of TensorProto
 * @param name_to_ort_value saved results.
 * @return Status
 */
Status CreateOrtValuesFromTensorProtos(
    const std::vector<const ONNX_NAMESPACE::TensorProto*>& tensor_protos,
    NameMLValMap& name_to_ort_value) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  for (const auto tensor_proto : tensor_protos) {
    TensorShape tensor_shape{utils::GetTensorShapeFromTensorProto(*tensor_proto)};
    const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto->data_type())->GetElementType();
    auto p_tensor = std::make_unique<Tensor>(tensor_dtype, tensor_shape, cpu_allocator);
    ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), nullptr, *tensor_proto, *p_tensor));

    OrtValue ort_value;
    ort_value.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    name_to_ort_value.emplace(tensor_proto->name(), ort_value);
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime