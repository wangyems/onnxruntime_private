// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "gradient_builder_base.h"

namespace onnxruntime {
namespace training {
// TODO: maybe group the gradient builders and split them into different files.
#define DECLARE_GRADIENT_BUILDER(name)                         \
  class name : public GradientBuilderBase {                    \
    using GradientBuilderBase::GradientBuilderBase;            \
    std::vector<NodeDef> GetGradientDefsImpl() const override; \
  };

DECLARE_GRADIENT_BUILDER(GetCastGradient)
DECLARE_GRADIENT_BUILDER(GetSinGradient)
DECLARE_GRADIENT_BUILDER(GetTanhGradient)
DECLARE_GRADIENT_BUILDER(GetSqrtGradient)
DECLARE_GRADIENT_BUILDER(GetErfGradient)
DECLARE_GRADIENT_BUILDER(GetMatMulGradient)
DECLARE_GRADIENT_BUILDER(GetSplitGradient)
DECLARE_GRADIENT_BUILDER(GetReluGradient)
DECLARE_GRADIENT_BUILDER(GetAddSubGradient)
DECLARE_GRADIENT_BUILDER(GetMulGradient)
DECLARE_GRADIENT_BUILDER(GetDivGradient)
DECLARE_GRADIENT_BUILDER(GetReduceMeanGradient)
DECLARE_GRADIENT_BUILDER(GetPowGradient)
DECLARE_GRADIENT_BUILDER(GetConcatGradient)
DECLARE_GRADIENT_BUILDER(GetReshapeGradient)
DECLARE_GRADIENT_BUILDER(GetTransposeGradient)
DECLARE_GRADIENT_BUILDER(GetPoolGradient)
DECLARE_GRADIENT_BUILDER(GetAveragePoolGradient)
DECLARE_GRADIENT_BUILDER(GetMaxPoolGradient)
DECLARE_GRADIENT_BUILDER(GetLRNGradient)
DECLARE_GRADIENT_BUILDER(GetDropoutGradient)
DECLARE_GRADIENT_BUILDER(GetGatherGradient)
DECLARE_GRADIENT_BUILDER(GetConvGradient)
DECLARE_GRADIENT_BUILDER(GetUnsqueezeGradient)
DECLARE_GRADIENT_BUILDER(GetSqueezeGradient)
DECLARE_GRADIENT_BUILDER(GetSoftmaxGradient)
DECLARE_GRADIENT_BUILDER(GetSoftmaxCrossEntropyGradient)
DECLARE_GRADIENT_BUILDER(GetSparseSoftmaxCrossEntropyGradient)
DECLARE_GRADIENT_BUILDER(GetGlobalAveragePoolGradient)
DECLARE_GRADIENT_BUILDER(GetGemmGradient)
DECLARE_GRADIENT_BUILDER(GetTrainableDropoutGradient)
DECLARE_GRADIENT_BUILDER(GetGatherNDGradient)
DECLARE_GRADIENT_BUILDER(GetGeluGradient)
DECLARE_GRADIENT_BUILDER(GetLayerNormalizationGradient)
DECLARE_GRADIENT_BUILDER(GetBatchNormalizationGradient)

}  // namespace training
}  // namespace onnxruntime
