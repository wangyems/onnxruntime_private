// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/framework/data_types.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include <type_traits>

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out) {
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<float>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out) {
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<int64_t>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
}

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out) {
  *out = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->InputCount();
  return nullptr;
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out) {
  *out = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->OutputCount();
  return nullptr;
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out) {
  *out = reinterpret_cast<const OrtValue*>(reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->GetInputMLValue(index));
  return nullptr;
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Out_ OrtValue** out) {
  onnxruntime::TensorShape shape(dim_values, dim_count);
  *out = reinterpret_cast<OrtValue*>(reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->OutputMLValue(index, shape));
  return nullptr;
};

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size) {
  std::string value;
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<std::string>(name, &value);
  if (status.IsOK()) {
    if (out == nullptr) {  // User is querying the true size of the attribute
      *size = value.size() + 1;
      return nullptr;
    } else if (*size >= value.size() + 1) {
      std::memcpy(out, value.data(), value.size());
      out[value.size()] = '\0';
      *size = value.size() + 1;
      return nullptr;
    } else {  // User has provided a buffer that is not large enough
      *size = value.size() + 1;
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Result buffer is not large enough");
    }
  }
  return onnxruntime::ToOrtStatus(status);
}

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetGPUComputeStream, _In_ const OrtKernelContext* context, _Outptr_ void** out) {
  *out = reinterpret_cast<const onnxruntime::OpKernelContext*>(context)->GetComputeStream();
  return nullptr;
};

template <typename T, typename std::enable_if<std::is_fundamental<T>::value, int>::type = 0>
static Status CopyDataFromVectorToMemory(const std::vector<T>& values, T* out, size_t* size) {
  if (out == nullptr) {  // User is querying the true size of the attribute
    *size = values.size();
    return Status::OK();
  } else if (*size >= values.size()) {
    std::memcpy(out, values.data(), values.size() * sizeof(T));
    *size = values.size();
  } else {  // User has provided a buffer that is not large enough
    *size = values.size();
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Result buffer is not large enough");
  }

  return Status::OK();
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttributeArray_float, _In_ const OrtKernelInfo* info, _In_ const char* name,
                    _Out_ float* out, _Inout_ size_t* size) {
  std::vector<float> values;
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttrs<float>(name, values);
  if (status.IsOK()) {
    status = CopyDataFromVectorToMemory<float>(values, out, size);
  }
  return onnxruntime::ToOrtStatus(status);
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttributeArray_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                    _Out_ int64_t* out, _Inout_ size_t* size) {
  std::vector<int64_t> values;
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttrs<int64_t>(name, values);
  if (status.IsOK()) {
    status = CopyDataFromVectorToMemory<int64_t>(values, out, size);
  }
  return onnxruntime::ToOrtStatus(status);
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#include "core/framework/customregistry.h"
namespace onnxruntime {

struct CustomOpKernel : OpKernel {
  CustomOpKernel(const OpKernelInfo& info, const OrtCustomOp& op) : OpKernel(info), op_(op) {
    if (op_.version > ORT_API_VERSION) {
      ORT_THROW("Unsupported version '" + std::to_string(op_.version) + "' in custom op '" + op.GetName(&op));
    }

    op_kernel_ = op_.CreateKernel(&op_, OrtGetApiBase()->GetApi(op_.version),
                                  reinterpret_cast<const OrtKernelInfo*>(&info));
  }

  ~CustomOpKernel() override { op_.KernelDestroy(op_kernel_); }

  Status Compute(OpKernelContext* ctx) const override {
    op_.KernelCompute(op_kernel_, reinterpret_cast<OrtKernelContext*>(ctx));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpKernel);

  const OrtCustomOp& op_;
  void* op_kernel_;
};

common::Status CreateCustomRegistry(gsl::span<OrtCustomOpDomain* const> op_domains,
                                    std::shared_ptr<CustomRegistry>& output) {
  output = std::make_shared<CustomRegistry>();

  for (const auto& domain : op_domains) {
    // Create an OpSchema for each op and register them

    // Container to hold type template parameters
    std::unordered_map<const OrtCustomOp*, std::vector<std::string>> type_constraint_ids;

#if !defined(ORT_MINIMAL_BUILD)
    // Domain is not empty - add it to the DomainToVersion ONNX map
    // If domain is empty, it is assumed to be part of the ONNX domain
    if (!domain->domain_.empty()) {
      // Add it to the DomainToVersion ONNX map if it doesn't already exist
      // For example, two sessions using the same session_options should not add the same custom op domain to the version map twice
      auto& domain_to_version_range_instance = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
      const auto& domain_to_version_map = domain_to_version_range_instance.Map();

      if (domain_to_version_map.find(domain->domain_) == domain_to_version_map.end()) {
        domain_to_version_range_instance.AddDomainToVersion(domain->domain_, 1, 1000);
      }
    }

    constexpr uint32_t min_ort_ver_io_opt = 8;   // Minimum ORT version supporting optional input/output.
    constexpr uint32_t min_ort_ver_io_var = 14;  // Minimum ORT version supporting variadic input/output.

    std::vector<ONNX_NAMESPACE::OpSchema> schemas_list;
    for (const auto* op : domain->custom_ops_) {
      ONNX_NAMESPACE::OpSchema schema(op->GetName(op), "custom op registered at runtime", 0);

      size_t type_id_counter = 0;
      const size_t input_count = op->GetInputTypeCount(op);
      for (size_t i = 0; i < input_count; i++) {
        onnx::OpSchema::FormalParameterOption option = onnx::OpSchema::FormalParameterOption::Single;
        bool is_homogeneous = true;
        int min_arity = 1;

        // The OrtCustomOp interface did not support the methods to query input/output characteristics before
        // ORT API version 8. So, query the relevant methods ONLY from API version 8 onwards.
        if (op->version >= min_ort_ver_io_opt) {
          const auto characteristic = op->GetInputCharacteristic(op, i);

          // Support for optional and variadic inputs/output was added in versions 8 and 14, respectively.
          if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL) {
            option = onnx::OpSchema::FormalParameterOption::Optional;
          } else if ((op->version >= min_ort_ver_io_var) &&
                     (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC)) {
            // Only the last input can be variadic.
            ORT_ENFORCE(i == input_count - 1, "Only the last input to a custom op may be marked variadic.");
            option = onnx::OpSchema::FormalParameterOption::Variadic;
            min_arity = op->GetVariadicInputMinArity(op);
            is_homogeneous = op->GetVariadicInputHomogeneity(op);
          }
        }

        const auto type = op->GetInputType(op, i);
        if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {  // Dynamic typed input
          schema.Input(i, "Input" + std::to_string(i), "", "T" + std::to_string(type_id_counter), option,
                       is_homogeneous, min_arity);
          schema.TypeConstraint("T" + std::to_string(type_id_counter), DataTypeImpl::ToString(DataTypeImpl::AllTensorTypes()), "all types");
          type_constraint_ids[op].push_back("T" + std::to_string(type_id_counter++));
        } else {
          schema.Input(i, "Input" + std::to_string(i), "",
                       DataTypeImpl::ToString(onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(type)), option,
                       is_homogeneous, min_arity);
        }
      }

      const size_t output_count = op->GetOutputTypeCount(op);
      for (size_t i = 0; i < output_count; i++) {
        onnx::OpSchema::FormalParameterOption option = onnx::OpSchema::FormalParameterOption::Single;
        bool is_homogeneous = true;
        int min_arity = 1;

        // The OrtCustomOp interface did not support the methods to query input/output characteristics before
        // ORT API version 8. So, query the relevant methods ONLY from API version 8 onwards.
        if (op->version >= min_ort_ver_io_opt) {
          const auto characteristic = op->GetOutputCharacteristic(op, i);

          // Support for optional and variadic inputs/output was added in versions 8 and 14, respectively.
          if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL) {
            option = onnx::OpSchema::FormalParameterOption::Optional;
          } else if ((op->version >= min_ort_ver_io_var) &&
                     (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC)) {
            // Only the last output can be variadic.
            ORT_ENFORCE(i == output_count - 1, "Only the last output to a custom op may be marked variadic.");
            option = onnx::OpSchema::FormalParameterOption::Variadic;
            min_arity = op->GetVariadicOutputMinArity(op);
            is_homogeneous = op->GetVariadicOutputHomogeneity(op);
          }
        }

        const auto type = op->GetOutputType(op, i);
        if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {  // Dynamic typed output
          ORT_ENFORCE(type_id_counter == 1,
                      "There must be one (and only one) dynamic typed input to the custom op. "
                      "Its type info at runtime will be used to infer the type info of this dynamic typed output "
                      "which is required for the success of the model loading step. "
                      "More than one dynamic typed inputs are currently not supported as differing types at runtime means the output type "
                      "cannot be inferred without which model loading cannot proceed.");

          schema.Output(i, "Output" + std::to_string(i), "", "T0", option, is_homogeneous, min_arity);
        } else {
          schema.Output(i, "Output" + std::to_string(i), "",
                        DataTypeImpl::ToString(onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(type)), option,
                        is_homogeneous, min_arity);
        }
      }

      schema.SetDomain(domain->domain_);
      schema.SinceVersion(1);
      schema.AllowUncheckedAttributes();
      schemas_list.push_back(schema);
    }

    ORT_RETURN_IF_ERROR(output->RegisterOpSet(schemas_list,
                                              domain->domain_,
                                              1 /* baseline opset version */,
                                              1000 /* opset version */));

#else
    // For a minimal build, we may not need any of the ONNX schema stuff but we still need to track
    // the type template parameters to be used during the kernel def building step below
    for (const auto* op : domain->custom_ops_) {
      size_t type_id_counter = 0;
      auto input_count = op->GetInputTypeCount(op);
      for (size_t i = 0; i < input_count; i++) {
        auto type = op->GetInputType(op, i);
        if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {  // Dynamic typed input
          type_constraint_ids[op].push_back("T" + std::to_string(type_id_counter++));
        }
      }
    }
#endif

    // create the KernelDef for each op and register it
    for (const auto* op : domain->custom_ops_) {
      KernelDefBuilder def_builder;
      def_builder.SetName(op->GetName(op))
          .SetDomain(domain->domain_)
          .SinceVersion(1);

      // GetInputMemoryType was introduced in ver 13. This check allows custom ops compiled using older versions
      // to work with newer versions (> 12) of the ORT binary.
      if (op->version > 12) {
        auto input_count = op->GetInputTypeCount(op);
        for (size_t i = 0; i < input_count; i++) {
          def_builder.InputMemoryType(op->GetInputMemoryType(op, i), i);
        }
      }

      for (auto& id : type_constraint_ids[op]) {
        def_builder.TypeConstraint(id, DataTypeImpl::AllTensorTypes());
      }

      if (const char* provider_type = op->GetExecutionProviderType(op)) {
        def_builder.Provider(provider_type);
      } else {
        def_builder.Provider(onnxruntime::kCpuExecutionProvider);
      }

      KernelCreateFn kernel_create_fn = [op](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
        out = std::make_unique<CustomOpKernel>(info, *op);
        return Status::OK();
      };

      KernelCreateInfo create_info(def_builder.Build(), kernel_create_fn);
      ORT_RETURN_IF_ERROR(output->RegisterCustomKernel(create_info));
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
