// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <unordered_map>

#include "core/framework/kernel_registry.h"

using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
// Function that is called for each formal parameter and its associated TypeProto.
// Returns true if traversal should continue, false otherwise.
using TraverseFn = std::function<bool(const ONNX_NAMESPACE::OpSchema::FormalParameter&,
                                      const ONNX_NAMESPACE::TypeProto*)>;

void TraverseFormalParametersWithTypeProto(const onnxruntime::Node& node, const TraverseFn& traverse_fn) {
  const ONNX_NAMESPACE::OpSchema& op_schema = *node.Op();

  // process inputs:
  const size_t len = node.InputArgCount().size();
  ORT_ENFORCE(len <= op_schema.inputs().size());
  int actual_index = 0;
  for (size_t formal_index = 0; formal_index != len; ++formal_index) {
    auto& param = op_schema.inputs()[formal_index];
    // return type of any corresponding actual parameter, if present
    for (int i = 0, end = node.InputArgCount()[formal_index]; i < end; ++i) {
      const onnxruntime::NodeArg* arg = node.InputDefs()[actual_index + i];
      if (!arg->Exists()) continue;  // a missing optional argument
      if (!traverse_fn(param, arg->TypeAsProto())) return;
    }
    actual_index += node.InputArgCount()[formal_index];
  }

  // process outputs:
  auto& actual_outputs = node.OutputDefs();
  const auto num_actual_outputs = actual_outputs.size();
  const auto last_formal = op_schema.outputs().size() - 1;
  for (size_t i = 0; i != num_actual_outputs; ++i) {
    const onnxruntime::NodeArg* arg = actual_outputs[i];
    if (!arg->Exists()) continue;
    auto& formal = op_schema.outputs()[std::min(i, last_formal)];
    if (!traverse_fn(formal, arg->TypeAsProto())) return;
  }
}

class TypeBindingResolver {
 public:
  TypeBindingResolver(const Node& node) : node_{node} {}
  virtual ~TypeBindingResolver() = default;
  // Resolves a name to a TypeProto* for a given node.
  // The name can represent either a type parameter or an input/output parameter.
  // Returns the resolved TypeProto* or nullptr if unable to resolve.
  virtual const ONNX_NAMESPACE::TypeProto* Resolve(const std::string& name_or_type_str) const = 0;

 protected:
  const Node& GetNode() const { return node_; }

 private:
  const Node& node_;
};

class MappingTypeBindingResolver : public TypeBindingResolver {
 public:
  MappingTypeBindingResolver(const Node& node) : TypeBindingResolver{node},
                                                 type_binding_map_{} {
    TraverseFormalParametersWithTypeProto(
        GetNode(),
        [this](const ONNX_NAMESPACE::OpSchema::FormalParameter& param,
               const ONNX_NAMESPACE::TypeProto* type) -> bool {
          type_binding_map_.emplace(param.GetName(), type);
          type_binding_map_.emplace(param.GetTypeStr(), type);
          return true;
        });
  }

  const ONNX_NAMESPACE::TypeProto* Resolve(const std::string& name_or_type_str) const override {
    auto found_it = type_binding_map_.find(name_or_type_str);
    if (found_it == type_binding_map_.end()) return nullptr;
    return found_it->second;
  }

 private:
  // map from input/output name or type string to TypeProto pointer
  using TypeBindingMap = std::unordered_map<std::string, const ONNX_NAMESPACE::TypeProto*>;
  TypeBindingMap type_binding_map_;
};

class SimpleTypeBindingResolver : public TypeBindingResolver {
 public:
  SimpleTypeBindingResolver(const Node& node) : TypeBindingResolver{node} {}

  const ONNX_NAMESPACE::TypeProto* Resolve(const std::string& name_or_type_str) const override {
    const ONNX_NAMESPACE::TypeProto* result{};
    TraverseFormalParametersWithTypeProto(
        GetNode(),
        [&name_or_type_str, &result](const ONNX_NAMESPACE::OpSchema::FormalParameter& param,
                                     const ONNX_NAMESPACE::TypeProto* type) -> bool {
          if (param.GetName() == name_or_type_str || param.GetTypeStr() == name_or_type_str) {
            result = type;
            return false;
          }
          return true;
        });
    return result;
  }
};
};  // namespace

std::vector<std::string> KernelRegistry::GetAllRegisteredOpNames() const {
  std::vector<std::string> ret(kernel_creator_fn_map_.size());
  size_t i = 0;
  for (const auto& kvp : kernel_creator_fn_map_) {
    ret[i++] = kvp.first;
  }
  return ret;
}

// Check whether the types of inputs/outputs of the given node match the extra
// type-constraints of the given kernel. This serves two purposes: first, to
// select the right kernel implementation based on the types of the arguments
// when we have multiple kernels, e.g., Clip<float> and Clip<int>; second, to
// accommodate (and check) mapping of ONNX (specification) type to the onnxruntime
// implementation type (e.g., if we want to implement ONNX's float16 as a regular
// float in onnxruntime). (The second, however, requires a globally uniform mapping.)
//
// Note that this is not intended for type-checking the node against the ONNX
// type specification of the corresponding op, which is done before this check.
bool KernelRegistry::VerifyKernelDef(const onnxruntime::Node& node,
                                     const KernelDef& kernel_def,
                                     std::string& error_str,
                                     onnxruntime::ProviderType exec_provider) {
  // check if domain matches
  if (node.Domain() != kernel_def.Domain()) {
    std::ostringstream ostr;
    ostr << "Op: " << node.OpType()
         << " Domain mismatch: "
         << " Expected: " << kernel_def.Domain()
         << " Actual: " << node.Domain();
    error_str = ostr.str();
    return false;
  }

  // check if execution provider matches
  const auto& node_provider = node.GetExecutionProviderType();
  const auto& expected_provider = (node_provider.empty() ? exec_provider : node_provider);
  if (expected_provider != kernel_def.Provider()) {
    std::ostringstream ostr;
    ostr << "Op: " << node.OpType()
         << " Execution provider mismatch."
         << " Expected: " << expected_provider
         << " Actual: " << kernel_def.Provider();
    error_str = ostr.str();
    return false;
  }

  // check if version matches
  int kernel_start_version, kernel_end_version;
  kernel_def.SinceVersion(&kernel_start_version, &kernel_end_version);

  int node_since_version = node.Op()->since_version();
  // Ideal case is, if schema is Since(5), current opset version is opset 7,
  // kernel_def Since(8)     Invalid
  // kernel_def Since(6)     Valid
  // kernel_def Since(5)     Valid
  // kernel_def Since(4)     Invalid
  // kernel_def Since(4, 6)  Valid

  // Right now there is no "until version" on schema, it is difficult to get opset version here.(require a lot of interface change.)
  // As a trade off, we will temporary require kernel definition to have the same since version as schema definition.
  // so kernel_def Since(6) will become invalid now.
  // After ONNX add "until version" on the schema object, we will update this place
  bool valid_version = kernel_start_version == node_since_version  // the idea case this branch should be kernel_start_version >= node_version && kernel_start_version <= until_version
                       || (kernel_start_version < node_since_version && kernel_end_version != INT_MAX && kernel_end_version >= node_since_version);
  if (!valid_version) {
    std::ostringstream ostr;
    ostr << "Op: " << node.OpType()
         << " Version mismatch."
         << " node_version: " << node_since_version
         << " kernel start version: " << kernel_start_version
         << " kernel_end_version: " << kernel_end_version;
    error_str = ostr.str();
    return false;
  }

  // check if type matches
  auto& kernel_type_constraints = kernel_def.TypeConstraints();

  // Note: The number of input/output nodes is N and the number of type
  // constraints is M. We select between an O(N*M) and an O(N+M) approach.
  // The O(N*M) approach has lower initial overhead.
  // kTypeBindingResolverComplexityThreshold is an arbitrary value of N*M,
  // above which we will use the O(N+M) approach.
  constexpr int kTypeBindingResolverComplexityThreshold = 50 * 50;
  // TODO could use std::variant (C++17) to avoid heap allocation
  const auto type_binding_resolver = [&]() -> std::unique_ptr<TypeBindingResolver> {
    if (kernel_type_constraints.size() * (node.Op()->inputs().size() + node.Op()->outputs().size()) <
        kTypeBindingResolverComplexityThreshold) {
      return std::make_unique<SimpleTypeBindingResolver>(node);
    } else {
      return std::make_unique<MappingTypeBindingResolver>(node);
    }
  }();

  for (auto& constraint : kernel_type_constraints) {
    const std::string& name = constraint.first;
    const std::vector<MLDataType>& allowed_types = constraint.second;
    const ONNX_NAMESPACE::TypeProto* actual_type = type_binding_resolver->Resolve(name);

    // If actual_type is null, this represents a type-constraint on a
    // missing optional parameter, which can be skipped.
    // TODO: We should check that names specified in kernel_type_constraints are
    // valid names (of types or parameters) at the time that kernels are registered.
    if ((nullptr != actual_type) &&
        !std::any_of(allowed_types.begin(), allowed_types.end(),
                     [actual_type, &node, &error_str](const DataTypeImpl* expected_type) {
                       bool rc = expected_type->IsCompatible(*actual_type);  // for easier debugging
                       if (!rc) {
                         // TODO print type information as well
                         error_str = "Op: " + node.OpType() + " Incompatible types.";
                       }
                       return rc;
                     })) {
      return false;
    }
  }
  return true;
}

Status KernelRegistry::Register(KernelDefBuilder& kernel_builder,
                                const KernelCreateFn& kernel_creator) {
  KernelCreateInfo create_info(kernel_builder.Build(), kernel_creator);
  return Register(create_info);
}

Status KernelRegistry::Register(KernelCreateInfo& create_info) {
  auto& op_name = create_info.kernel_def->OpName();

  // Check op version conflicts.
  auto range = kernel_creator_fn_map_.equal_range(op_name);
  for (auto i = range.first; i != range.second; ++i) {
    if (i->second.kernel_def &&
        i->second.status.IsOK() &&
        i->second.kernel_def->IsConflict(*create_info.kernel_def)) {
      create_info.status =
          Status(ONNXRUNTIME, FAIL,
                 "Failed to add kernel for " + op_name +
                     ": Conflicting with a registered kernel with op versions.");
      // For invalid entries, we keep them in the map now. Must check for status
      // when using the entries from the map.
      kernel_creator_fn_map_.emplace(op_name, std::move(create_info));
      return create_info.status;
    }
  }

  // Register the kernel.
  // Ownership of the KernelDef is transferred to the map.
  kernel_creator_fn_map_.emplace(op_name, std::move(create_info));
  return Status::OK();
}

Status KernelRegistry::CreateKernel(const onnxruntime::Node& node,
                                    const IExecutionProvider& execution_provider,
                                    const SessionState& session_state,
                                    /*out*/ std::unique_ptr<OpKernel>& op_kernel) const {
  const KernelCreateInfo* kernel_create_info = TryFindKernel(node, execution_provider.Type());

  if (!kernel_create_info) {
    return Status(ONNXRUNTIME, FAIL, "Failed to find kernel for " + node.OpType());
  }

  OpKernelInfo kernel_info(node, *kernel_create_info->kernel_def, execution_provider, session_state);
  op_kernel.reset(kernel_create_info->kernel_create_func(kernel_info));
  return Status::OK();
}

static std::string ToString(const std::vector<std::string>& error_strs) {
  std::ostringstream ostr;
  std::for_each(std::begin(error_strs), std::end(error_strs),
                [&ostr](const std::string& str) { ostr << str << " "; });
  return ostr.str();
}

const KernelCreateInfo* KernelRegistry::TryFindKernel(const onnxruntime::Node& node,
                                                      onnxruntime::ProviderType exec_provider) const {
  auto range = kernel_creator_fn_map_.equal_range(node.OpType());
  std::vector<std::string> error_strs;
  for (auto i = range.first; i != range.second; ++i) {
    if (!i->second.status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Failed to create kernel for op: " << node.OpType()
                          << " since it was ill-formed during registration";
      continue;
    }
    std::string error_str;
    if (VerifyKernelDef(node, *i->second.kernel_def, error_str, exec_provider)) {
      return &i->second;
    }
    error_strs.push_back(error_str);
  }
  LOGS_DEFAULT(INFO) << node.OpType() << " kernel is not supported in " << exec_provider
                     << " Encountered following errors: " << ToString(error_strs);
  return nullptr;
}

}  // namespace onnxruntime
