// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.



#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class EinsumOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Helper functions, thanks for DML OperatorHelper
enum class RecognizedOperatorType
{
    None,
    Identity,
    Multiply,
    OuterProduct,
    MatMul,
    MatMulTransposeA,
    MatMulTransposeB,
    MatMulNhcw,
    MatMulNhcwTransposeA,
    MatMulNhcwTransposeB,
    ReduceSum,
    Transpose,
    Total,
};

struct RecognizedOperatorInfo
{
  RecognizedOperatorType recognized_operator_type;
  std::initializer_list<uint32_t> component_ranks;
  std::initializer_list<uint32_t> label_indices;
};

struct Component
{
  uint32_t label_idx_begin;
  uint32_t label_idx_end;

  uint32_t GetDimensionCount() const noexcept
  {
    return label_idx_end - label_idx_begin;
  }
  gsl::span<const uint32_t> GetLabels(gsl::span<const uint32_t> labels) const
  {
    return labels.subspan(label_idx_begin, label_idx_end - label_idx_begin);
  };
};

bool ParseEquationComponents(const InitializedTensorSet& initializers,
                     const Node& node, const std::string& equation,
                     std::vector<uint32_t>& m_label_indices,
                     std::vector<Component>& m_components,
                     std::vector<uint32_t>& m_output_dimensions,
                     const logging::Logger& logger) {

  // Parse the equation and mapping each axis into numeric indices
  std::map<char, uint32_t> label_maps;
  std::set<char> repeated_labels;

  uint32_t current_label_idx = 0;
  Component current_component = {};
  bool at_output = false;
  bool end_flag = false;

  // Parsing inputs and output
  for (const char* it = equation.data(); !end_flag; ++it) { // std::string.data() promises the end of the string is '\0'
    char ch = *it;

    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) {
      const auto [i, inserted] = label_maps.insert({ch, current_label_idx});
      if (inserted) {
        if (at_output) {
          LOGS(logger, VERBOSE) << "Found label in equation output not matching any label from inputs.";
          return false;
        }
        ++current_label_idx;
      }
      else if (!at_output) {
        repeated_labels.insert(ch);
      }
      m_label_indices.push_back(i->second);
    }
    else if (ch == ' ') {
      continue;
    }
    else {
      current_component.label_idx_end = static_cast<uint32_t>(m_label_indices.size());
      m_components.push_back(current_component);
      current_component.label_idx_begin = current_component.label_idx_end;

      switch (ch) {
        case ',':
          break;

        case '-':
          ++it;
          if (*it != '>') {
            LOGS(logger, VERBOSE) << "Expected '->' for output.";
            return false;
          }
          if (at_output) {
            LOGS(logger, VERBOSE) << "Only one output arrow '->' is valid.";
            return false;
          }
          at_output = true;
          break;

        case '.':
          // Ellipsis is unsupported
          LOGS(logger, VERBOSE) << "Ellipsis is unsupported.";
          return false;

        case '\0':
          end_flag = true;
          break; // End of string.

        default:
          LOGS(logger, VERBOSE) << "Unsupported character in equation string.";
          return false;
      }
    }
  }

  // No explicit output was given
  if (!at_output) {
    for (auto i : label_maps) {
      if (repeated_labels.count(i.first) == 0) {
        m_label_indices.push_back(i.second);
      }
    }

    current_component.label_idx_end = static_cast<uint32_t>(m_label_indices.size());
    m_components.push_back(current_component);
  }
  return true;
}

RecognizedOperatorType DetermineRecognizedOperatorType(const std::vector<uint32_t>& m_label_indices,
                     const std::vector<Component>& m_components,
                     const std::vector<uint32_t>& m_output_dimensions) {
  if (m_components.empty()) return RecognizedOperatorType::None;

  auto equals = [](gsl::span<const uint32_t> a, gsl::span<const uint32_t> b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  };

  auto as_span = [](std::initializer_list<uint32_t> il) {
    return gsl::make_span(il.begin(), il.size());
  };

  std::array<uint32_t, 3> component_ranks;
  if (m_components.size() > component_ranks.size()) {
    // So far, not support for more than two inputs and one output.
    return RecognizedOperatorType::None;
  }
  else if (m_components.size() == 2) { // one input
    auto input_labels = m_components[0].GetLabels(m_label_indices);
    auto output_labels = m_components[1].GetLabels(m_label_indices);
    if (input_labels.size() == output_labels.size()) {
      if (equals(input_labels, output_labels)) { // identity
        return RecognizedOperatorType::Identity;
      }
      else {
        return RecognizedOperatorType::Transpose;
      }
    }
    else if (output_labels.empty()) { // scalar output, reduce
      return RecognizedOperatorType::ReduceSum;
    }

  }
  else if (m_components.size() == 3) { // two inputs
    auto input_A_labels = m_components[0].GetLabels(m_label_indices);
    auto input_B_labels = m_components[1].GetLabels(m_label_indices);
    auto output_labels = m_components[2].GetLabels(m_label_indices);
    if (equals(input_A_labels, output_labels) && equals(input_B_labels, output_labels)) { // element-wise product
      return RecognizedOperatorType::Multiply;
    }
  }

  const RecognizedOperatorInfo recognized_operators[] = {
    {RecognizedOperatorType::MatMul,                {2,2,2},{0,1, 1,2, 0,2}}, // ik,kj->ij
    {RecognizedOperatorType::MatMul,                {3,3,3},{0,1,2, 0,2,3, 0,1,3}}, // bik,bkj->bij
    {RecognizedOperatorType::MatMul,                {4,4,4},{0,1,2,3, 0,1,3,4, 0,1,2,4}}, // abik,abkj->abij
    {RecognizedOperatorType::OuterProduct,          {1,1,2},{0, 1, 0,1}}, // i,j->ij
    {RecognizedOperatorType::MatMulTransposeA,      {2,2,2},{0,1, 0,2, 1,2}}, // ji,jk->ik
    {RecognizedOperatorType::MatMulTransposeA,      {3,3,3},{0,1,2, 0,1,3, 0,2,3}}, // bji,bjk->bik
    {RecognizedOperatorType::MatMulTransposeA,      {4,4,4},{0,1,2,3, 0,1,2,4, 0,1,3,4}}, // abji,abjk->abik
    {RecognizedOperatorType::MatMulTransposeB,      {2,2,2},{0,1, 2,1, 0,2}}, // ij,kj->ik
    {RecognizedOperatorType::MatMulTransposeB,      {3,3,3},{0,1,2, 0,3,2, 0,1,3}}, // bij,bkj->bik
    {RecognizedOperatorType::MatMulTransposeB,      {4,4,4},{0,1,2,3, 0,1,4,3, 0,1,2,4}}, // abij,abkj->abik
    {RecognizedOperatorType::MatMulNhcw,            {4,4,4},{0,1,2,3, 0,3,2,4, 0,1,2,4}}, // aibj,ajbk->aibk
    {RecognizedOperatorType::MatMulNhcwTransposeA,  {4,4,4},{0,1,2,3, 0,1,2,4, 0,3,2,4}}, // ajbi,ajbk->aibk
    {RecognizedOperatorType::MatMulNhcwTransposeB,  {4,4,4},{0,1,2,3, 0,4,2,3, 0,1,2,4}}, // aibj,akbj->aibk
    {RecognizedOperatorType::ReduceSum,             {2,1  },{0,1, 0}}, //ij->i
    {RecognizedOperatorType::ReduceSum,             {2,1  },{0,1, 1}}, //ij->j
  };


  for (auto& recognized_operator: recognized_operators) {
    if (equals(m_label_indices, as_span(recognized_operator.label_indices))
    && m_components.size() == recognized_operator.component_ranks.size()) {
      for (size_t i = 0; i < m_components.size(); ++i) {
        component_ranks[i] = m_components[i].GetDimensionCount();
      }

      if (equals(gsl::make_span(component_ranks.data(), m_components.size()), as_span(recognized_operator.component_ranks))) {
        return recognized_operator.recognized_operator_type;
      }
    }
  }

  return RecognizedOperatorType::None;
}


// Add operator related.

void EinsumOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

Status EinsumOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  emscripten::val output = emscripten::val::object();


  NodeAttrHelper helper(node);
  const auto equation = helper.Get("equation", std::string(" "));

  std::vector<uint32_t> m_label_indices;
  std::vector<Component> m_components;
  std::vector<uint32_t> m_output_dimensions;
  ORT_RETURN_IF_NOT(ParseEquationComponents(initializers, node, equation, m_label_indices,
      m_components, m_output_dimensions, logger), "Error parsing equation components.");

  RecognizedOperatorType recognized_operator_type = DetermineRecognizedOperatorType(m_label_indices, m_components, m_output_dimensions);

  static_assert(RecognizedOperatorType::Total == static_cast<RecognizedOperatorType>(12), "Update this switch.");
  switch(recognized_operator_type)
  {
  case RecognizedOperatorType::Multiply:
    {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());
      output = model_builder.GetBuilder().call<emscripten::val>("mul", a, b);
    }
    break;

  case RecognizedOperatorType::OuterProduct:
    {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());

      std::vector<int64_t> a_shape, b_shape;
      ORT_RETURN_IF_NOT(GetShape(*input_defs[0], a_shape, logger), "Cannot get shape");
      ORT_RETURN_IF_NOT(GetShape(*input_defs[1], b_shape, logger), "Cannot get shape");


      std::vector<int64_t> new_a_shape = a_shape;
      new_a_shape.push_back(static_cast<uint32_t>(1));
      std::vector<int64_t> new_b_shape = b_shape;
      new_b_shape.insert(new_b_shape.begin(), static_cast<uint32_t>(1));

      emscripten::val new_a = model_builder.GetBuilder().call<emscripten::val>("reshape",
                  a, emscripten::val::array(new_a_shape));
      emscripten::val new_b = model_builder.GetBuilder().call<emscripten::val>("reshape",
                  b, emscripten::val::array(new_b_shape));

      emscripten::val options = emscripten::val::object();

      output = model_builder.GetBuilder().call<emscripten::val>("gemm", a, b);
    }
    break;

  case RecognizedOperatorType::MatMulTransposeA:
  case RecognizedOperatorType::MatMulTransposeB:
  case RecognizedOperatorType::MatMul:
    {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());

      if (recognized_operator_type == RecognizedOperatorType::MatMulTransposeA)
      {
        std::vector<int64_t> input_shape;
        ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
        auto input_dims = static_cast<int64_t>(input_shape.size());

        std::vector<uint32_t> permutation;

        for (uint32_t i = 0; i < input_dims-2; ++i)
          permutation.push_back(i);

        permutation.push_back(static_cast<uint32_t>(input_dims-1));
        permutation.push_back(static_cast<uint32_t>(input_dims-2));

        emscripten::val options = emscripten::val::object();
        options.set("permutation", emscripten::val::array(permutation));
        a = model_builder.GetBuilder().call<emscripten::val>("transpose", a, options);
      }
      else if (recognized_operator_type == RecognizedOperatorType::MatMulTransposeB)
      {
        std::vector<int64_t> input_shape;
        ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_shape, logger), "Cannot get shape");
        auto input_dims = input_shape.size();

        std::vector<int64_t> permutation;

        for (int64_t i = 0; i < input_dims-2; ++i)
          permutation.push_back(i);

        permutation.push_back(input_dims-1);
        permutation.push_back(input_dims-2);

        emscripten::val options = emscripten::val::object();
        options.set("permutation", emscripten::val::array(permutation));
        b = model_builder.GetBuilder().call<emscripten::val>("transpose", b, options);
      }

      output = model_builder.GetBuilder().call<emscripten::val>("matmul", a, b);
    }
    break;

  case RecognizedOperatorType::MatMulNhcw:
  case RecognizedOperatorType::MatMulNhcwTransposeA:
  case RecognizedOperatorType::MatMulNhcwTransposeB:
    {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());

      emscripten::val options = emscripten::val::object();
      std::vector<int64_t> permutation = {0,2,1,3};
      std::vector<int64_t> permutation_a = {0,2,1,3};
      std::vector<int64_t> permutation_b = {0,2,1,3};
      if (recognized_operator_type == RecognizedOperatorType::MatMulNhcwTransposeA)
      {
        permutation_a = {0,2,3,1};
      }
      else if (recognized_operator_type == RecognizedOperatorType::MatMulNhcwTransposeB)
      {
        permutation_b = {0,2,3,1};
      }

      options.set("permutation", emscripten::val::array(permutation_a));
      a = model_builder.GetBuilder().call<emscripten::val>("transpose", a, options);
      options.set("permutation", emscripten::val::array(permutation_b));
      b = model_builder.GetBuilder().call<emscripten::val>("transpose", b, options);
      output = model_builder.GetBuilder().call<emscripten::val>("matmul", a, b);

      options.set("permutation", emscripten::val::array(permutation));
      output = model_builder.GetBuilder().call<emscripten::val>("transpose", output, options);
    }
    break;

  case RecognizedOperatorType::ReduceSum:
    {
      auto kept_axes = m_components.back().GetLabels(m_label_indices);
      assert(kept_axes.size() <= 1);
      std::vector<uint32_t> reduced_axes;
      // uint32_t kept_axes_mask = (kept_axes.size() << 0) - 1;
      uint32_t kept_axes_mask = 0;
      for (auto axis : kept_axes)
      {
          kept_axes_mask |= (1 << axis);
      }
      std::vector<int64_t> output_shape;
      const auto& output_defs = node.OutputDefs();
      ORT_RETURN_IF_NOT(GetShape(*output_defs[0], output_shape, logger), "Cannot get shape");
      for (uint32_t axis = 0, axis_count = static_cast<uint32_t>(output_shape.size()); axis < axis_count; ++axis)
      {
          if (~kept_axes_mask & (1<<axis))
          {
              reduced_axes.push_back(axis);
          }
      }

      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      emscripten::val options = emscripten::val::object();
      options.set("keepDimensions", false);

      std::vector<int64_t> input_shape;
      ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
      const auto input_rank = input_shape.size();

      std::vector<int32_t> axes_data;
      std::transform(
          reduced_axes.begin(), reduced_axes.end(), std::back_inserter(axes_data),
          [input_rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, input_rank)); });
      options.set("axes", emscripten::val::array(axes_data));

      output = model_builder.GetBuilder().call<emscripten::val>("reduceSum", input, options);
    }
    break;

  case RecognizedOperatorType::Transpose:
    {
      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      // Transpose via input strides. The output tensor is not strided.
      assert(m_components.front().GetDimensionCount() == m_components.back().GetDimensionCount());
      // Remap transposed strides using the component labels from input to output.
      auto label_indices = m_components.back().GetLabels(m_label_indices);

      std::vector<uint32_t> permutation{label_indices.begin(), label_indices.end()};
      emscripten::val options = emscripten::val::object();
      options.set("permutation", emscripten::val::array(permutation));
      output = model_builder.GetBuilder().call<emscripten::val>("transpose", input, options);
    }
    break;
  case RecognizedOperatorType::Identity:
    {
      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      output = model_builder.GetBuilder().call<emscripten::val>("identity", input);
    }
    break;
  default:
    break;
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool EinsumOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                        const Node& node,
                                        const WebnnDeviceType device_type,
                                        const logging::Logger& logger) const {

  if (device_type == WebnnDeviceType::CPU) {
    LOGS(logger, VERBOSE) << "Einsum is not supported for cpu in WebNN EP. Matmul and ReduceSum are not supported in XNNPACK.";
    return false;
  }

  const auto& input_defs = node.InputDefs();

  NodeAttrHelper helper(node);
  const auto equation = helper.Get("equation", std::string(" "));
  std::vector<uint32_t> m_label_indices;
  std::vector<Component> m_components;
  std::vector<uint32_t> m_output_dimensions;

  if (!ParseEquationComponents(initializers, node, equation, m_label_indices,
      m_components, m_output_dimensions, logger))
    return false;

  if (static_cast<uint32_t>(input_defs.size()) + 1 != m_components.size()) {
    LOGS(logger, VERBOSE) << "EinSum input tensor count is inconsistent with the equation component count.";
    return false;
  }

  /* // dml's condition, im not sure if it is necessary
  if (static_cast<uint32_t>(node.OutputDefs()) != 1) {
    LOGS(logger, VERBOSE) << "EinSum expects one output tensor.";
    return false;
  }
  */
  RecognizedOperatorType recognized_operator_type = DetermineRecognizedOperatorType(m_label_indices, m_components, m_output_dimensions);
  if (recognized_operator_type == RecognizedOperatorType::None) {
    LOGS(logger, VERBOSE) << "The equation is not supported in Einsum.";
    return false;
  }

  return true;
}

void CreateEinsumOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<EinsumOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
