// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_normalizer.h"
#include "onnx/defs/schema.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

#include <codecvt>
#include <locale>
#include <functional>
#include <unordered_set>

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    StringNormalizer,
    1,
    string,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    contrib::StringNormalizer);

namespace string_normalizer {
const std::string conv_error("Conversion Error");
const std::wstring wconv_error(L"Conversion Error");
// performs tolower/toupper in-place
void ChangeCase(const std::locale& loc, StringNormalizer::CaseAction caseaction,
                std::wstring& wstr) {
  if (caseaction == StringNormalizer::LOWER) {
    std::transform(wstr.begin(), wstr.end(), wstr.begin(),
                   [&loc](wchar_t ch) { return std::tolower(ch, loc); });
  } else {
    std::transform(wstr.begin(), wstr.end(), wstr.begin(),
                   [&loc](wchar_t ch) { return std::toupper(ch, loc); });
  }
}

template <class ForwardIter>
Status CopyCaseAction(ForwardIter first, ForwardIter end, OpKernelContext* ctx,
                      const std::locale& loc,
                      std::wstring_convert<std::codecvt_utf8<wchar_t>>& converter,
                      size_t N, size_t C,
                      StringNormalizer::CaseAction caseaction) {
  std::vector<int64_t> output_dims;
  if (N == 1) {
    output_dims.push_back(1);
  }

  // Empty output case
  if (C == 0) {
    output_dims.push_back(1);
    TensorShape output_shape(output_dims);
    ctx->Output(0, output_shape);
    return Status::OK();
  }

  output_dims.push_back(C);

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  auto const output_data = output_tensor->template MutableData<std::string>();

  size_t output_idx = 0;
  while (first != end) {
    const std::string& s = *first;
    if (caseaction == StringNormalizer::LOWER || caseaction == StringNormalizer::UPPER) {
      std::wstring wstr = converter.from_bytes(s);
      if (wstr == wconv_error) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                      "Input contains invalid utf8 chars at: " + s);
      }
      // In place transform
      ChangeCase(loc, caseaction, wstr);
      new (output_data + output_idx) std::string(converter.to_bytes(wstr));
    } else {
      // Simple copy
      new (output_data + output_idx) std::string(s);
    }
    ++output_idx;
    ++first;
  }
  return Status::OK();
}
}  // namespace string_normalizer

StringNormalizer::StringNormalizer(const OpKernelInfo& info) : OpKernel(info) {
  std::string casechangeaction;
  auto status = info.GetAttr("casechangeaction", &casechangeaction);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "attribute caseaction is not set");
  if (casechangeaction == "LOWER") {
    casechangeaction_ = LOWER;
  } else if (casechangeaction == "UPPER") {
    casechangeaction_ = UPPER;
  } else if (casechangeaction == "NONE") {
    casechangeaction_ = NONE;
  } else {
    ONNXRUNTIME_ENFORCE(false, "attribute casechangeaction has invalid value");
  }
  int64_t iscasesensitive = 0;
  status = info.GetAttr("iscasesensitive", &iscasesensitive);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "attribute iscasesensitive is not set");
  iscasesensitive_ = iscasesensitive != 0;

  info.GetAttrs<std::string>("stopwords", stopwords_);
  // Default is specified in the schema
  status = info.GetAttr("locale", &locale_);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "Failed to get locale");
}

Status StringNormalizer::Compute(OpKernelContext* ctx) const {
  using namespace string_normalizer;

  auto X = ctx->Input<Tensor>(0);
  auto& input_dims = X->Shape().GetDims();

  size_t N = 0;
  size_t C = 0;
  if (input_dims.size() == 1) {
    if (input_dims[0] < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Single dimension value must be greater than 0");
    }
    C = input_dims[0];
  } else if (input_dims.size() == 2) {
    if (input_dims[0] != 1 || input_dims[1] < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input dimensions are either[C > 0] or [1][C > 0] allowed");
    }
    N = 1;
    C = input_dims[1];
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input dimensions are either[C > 0] or [1][C > 0] allowed");
  }

  Status status;
  std::locale loc(locale_);
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter(conv_error, wconv_error);
  auto const input_data = X->template Data<std::string>();

  using StrRef = std::reference_wrapper<const std::string>;
  if (iscasesensitive_) {
    if (!stopwords_.empty()) {
      // Create a filter and create filtered output
      std::unordered_set<StrRef,
                         std::hash<std::string>,
                         std::equal_to<std::string>>
          swords;
      std::transform(stopwords_.begin(), stopwords_.end(), std::inserter(swords, swords.end()),
                     [](const std::string& s) { return std::cref(s); });

      std::vector<StrRef> filtered_strings;
      filtered_strings.reserve(C);
      for (size_t input_idx = 0; input_idx < C; ++input_idx) {
        const std::string& s = *(input_data + input_idx);
        if (0 == swords.count(s)) {
          filtered_strings.push_back(std::cref(s));
        }
      }
      status = CopyCaseAction(filtered_strings.cbegin(), filtered_strings.cend(), ctx, loc, converter,
                              N, filtered_strings.size(), casechangeaction_);
    } else {
      // Nothing to filter. Copy input to output and change case if needed
      status = CopyCaseAction(input_data, input_data + C, ctx, loc, converter, N, C, casechangeaction_);
    }
  } else {
    if (!stopwords_.empty()) {
      // Perform case-insensitive comparison. Convert to lowercase for NONE, LOWER and UPPER otherwise.
      const CaseAction ca = (casechangeaction_ == UPPER) ? UPPER : LOWER;
      std::unordered_set<std::wstring> swords;
      std::transform(stopwords_.begin(), stopwords_.end(), std::inserter(swords, swords.end()),
                     [&loc, &converter, ca](const std::string& s) {
                       std::wstring wstr = converter.from_bytes(s);
                       ChangeCase(loc, ca, wstr);
                       return wstr;
                     });

      // Filter input. We choose to undergo conversion twice (if needed)
      // as oppose to preserve lower/uppercased strings to favor lower memory
      // consumption.
      std::vector<StrRef> filtered_strings;
      filtered_strings.reserve(C);
      for (size_t input_idx = 0; input_idx < C; ++input_idx) {
        const std::string& s = *(input_data + input_idx);
        std::wstring wstr = converter.from_bytes(s);
        ChangeCase(loc, ca, wstr);
        if (0 == swords.count(wstr)) {
          filtered_strings.push_back(std::cref(s));
        }
      }
      status = CopyCaseAction(filtered_strings.cbegin(), filtered_strings.cend(), ctx, loc, converter,
                              N, filtered_strings.size(), casechangeaction_);
    } else {
      // Nothing to filter. Copy input to output and change case if needed
      status = CopyCaseAction(input_data, input_data + C, ctx, loc, converter, N, C, casechangeaction_);
    }
  }
  return status;
}
}  // namespace contrib
}  // namespace onnxruntime
