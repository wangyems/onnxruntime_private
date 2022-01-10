// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/pad_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <tvm/topi/nn.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::PrimExpr _SelectMapOr(const tvm::Array<tvm::PrimExpr>& conds) {
  // tvm::Map(conds, tvm::tir::Or)
  // TVMTODO
  tvm::PrimExpr res;
  return res;
}

// Note tvm::topi::pad does not support modes {edge, reflect}
// Therefore, MTI implements a generic Pad
tvm::te::Tensor Pad(const tvm::te::Tensor& t,
                const tvm::Array<tvm::PrimExpr>& pad_before,
                const tvm::Array<tvm::PrimExpr>& pad_after,
                float pad_value,
                const std::string& mode,
                const std::string& name) {
  MTI_ASSERT(pad_before.size() >= 1);
  MTI_ASSERT(pad_before.size() == pad_after.size());
  MTI_ASSERT(pad_before.size() == t->shape.size());

  tvm::Array<tvm::PrimExpr> output_shape;
  tvm::arith::Analyzer analyzer;
  for (size_t i = 0; i < t->shape.size(); ++i) {
    output_shape.push_back(
        analyzer.Simplify(t->shape[i] + pad_before[i] + pad_after[i]));
  }

  auto l = [&](const tvm::Array<tvm::tir::Var>& ovars) {
    tvm::Array<tvm::PrimExpr> conds;
    tvm::Array<tvm::PrimExpr> indices;
    tvm::Array<tvm::PrimExpr> coords;

    for (size_t i = 0; i < t->shape.size(); ++i) {
      tvm::PrimExpr ivar = ovars[i] - pad_before[i];
      tvm::PrimExpr min = 0;
      tvm::PrimExpr extent = t->shape[i];

      conds.push_back(ivar < min);
      conds.push_back(ivar >= min + extent);
      indices.push_back(tvm::max(tvm::min(ivar, min + extent - 1), min));

      if (mode == "reflect") {
        // calculate indices for reflect mode
         tvm::PrimExpr limit = extent - 1;
         tvm::PrimExpr coord = ivar - min;
        // Avoid mod zero when tensor shape has 1,
        // e.g. input shape is [1, 3, 3] instead of [3, 3]
        auto* p_limit = tvm::tir::as_const_int(limit);
        if (p_limit != nullptr && *p_limit != 0)
          coord = tvm::floormod((coord + 2 * limit), (2 * limit));  // avoid negative value
        coord = coord - limit;
        coord = tvm::abs(coord);
        coord = limit - coord;
        coord = coord + min;
        coords.push_back(coord);
      }
    }

    /*
    if (mode == "reflect") {
      return tvm::tir::Select(_SelectMapOr(conds),
                              t(coords), t(indices));
    } else if (mode == "constant") {
      return tvm::tir::Select(_SelectMapOr(conds),
                              tvm::tir::make_const(t->dtype, pad_value), t(indices));
    }
    */

    // default mode is edge
    // TVMTODO
    tvm::te::Tensor res;
    return res; //t(indices);
  };

  // TVMTODO
  tvm::te::Tensor res;
  return res;
  //return tvm::te::compute(output_shape, l, name);
}

tvm::te::Tensor Pad(const tvm::te::Tensor& t,
                const tvm::Array<tvm::PrimExpr>& output_shape,
                const  tvm::PrimExpr& pad_value,
                const std::string& name) {
  MTI_ASSERT(t->dtype == pad_value.dtype());

  auto l = [&](const tvm::Array<tvm::tir::Var>& ovars) {
    tvm::Array<tvm::PrimExpr> conds;
    tvm::Array<tvm::PrimExpr> indices;

    for (size_t i = 0; i < t->shape.size(); ++i) {
      tvm::PrimExpr ivar = ovars[i];
      tvm::PrimExpr min = 0;
      tvm::PrimExpr extent = t->shape[i];

      conds.push_back(ivar < min);
      conds.push_back(ivar >= min + extent);
      indices.push_back(tvm::max(tvm::min(ivar, min + extent - 1), min));
    }

    return tvm::tir::Select(_SelectMapOr(conds),
                            pad_value, t(indices));
  };

  return tvm::te::compute(output_shape, l, name);
}

tvm::te::Tensor PadLastDim(const tvm::te::Tensor& t,
                       const int32_t align_size,
                       const tvm::PrimExpr& pad_value,
                       const std::string& name) {
  auto input_shape = t->shape;
  tvm::Array<tvm::PrimExpr> out_shape;
  size_t input_shape_rank = input_shape.size();
  for (size_t i = 0; i < input_shape_rank - 1; ++i) {
    out_shape.push_back(input_shape[i]);
  }
  out_shape.push_back(
      tvm::floordiv((input_shape[input_shape_rank - 1] + align_size - 1),
                    align_size * align_size));

  return Pad(t, out_shape, pad_value, name + "_pad");
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
