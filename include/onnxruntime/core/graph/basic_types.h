// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstdint>
#include <memory>
#include <functional>

namespace ONNX_NAMESPACE {
class ValueInfoProto;
class TensorProto;
class SparseTensorProto;
class TypeProto;
class AttributeProto;
// define types that would come from the ONNX library if we were building against it.
#if defined(ORT_MINIMAL_BUILD)
using OperatorSetVersion = int;
#endif

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
using NodeIndex = size_t;
using Version = int64_t;
using NodeArgInfo = ONNX_NAMESPACE::ValueInfoProto;
using InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;
using ArgNameToTypeMap = std::unordered_map<std::string, ONNX_NAMESPACE::TypeProto>;
using ProviderType = const std::string&;
// TODO - Evaluate switching the types below to support transparent comparators and enable
// lookups based on gsl::cstring_span<> and std::string_view.  This would reduces allocations
// converting to std::string, but requires conversion to std::map<std::string, foo, std::less<>>
// instead of std::unordered_map<std::string, foo, [std::less<foo>]>.

using NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto>;
class IOnnxRuntimeOpSchemaCollection;
using IOnnxRuntimeOpSchemaCollectionPtr = std::shared_ptr<IOnnxRuntimeOpSchemaCollection>;
}  // namespace onnxruntime

namespace onnxruntime {
class OpKernel;
class OpKernelInfo;

using KernelCreateFn = std::function<OpKernel*(const OpKernelInfo& info)>;
}  // namespace onnxruntime
