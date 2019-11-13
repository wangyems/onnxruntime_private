//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once
#include "APITests.h"
using namespace WEX::Common;

class LearningModelBindingAPITests : public APITests
{
public:
    TEST_CLASS(LearningModelBindingAPITests);

    TEST_METHOD(CpuSqueezeNet);
    TEST_METHOD(CpuSqueezeNet_EmptyOutputs);
    TEST_METHOD(CpuSqueezeNet_BindInputTensorAsInspectable);
    TEST_METHOD(CpuSqueezeNet_UnboundOutputs);
    TEST_METHOD(CpuFnsCandy16);
    TEST_METHOD(CpuFnsCandy16_UnboundOutputs);
    TEST_METHOD(ImageBindingDimensions);
    TEST_METHOD(VerifyInvalidBindExceptions);
    TEST_METHOD(BindInvalidInputName);

    // Simple one node models with CPU only operators
    // https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
    TEST_METHOD(CastMapInt64);
    TEST_METHOD(ZipMapInt64);
    TEST_METHOD(ZipMapInt64_Unbound);
    TEST_METHOD(ZipMapString);
    TEST_METHOD(DictionaryVectorizerMapInt64);
    TEST_METHOD(DictionaryVectorizerMapString);
    TEST_METHOD(GpuSqueezeNet);
    TEST_METHOD(GpuSqueezeNet_EmptyOutputs);
    TEST_METHOD(GpuSqueezeNet_UnboundOutputs);
    TEST_METHOD(GpuFnsCandy16);
    TEST_METHOD(GpuFnsCandy16_UnboundOutputs);

    TEST_METHOD(VerifyOutputAfterEvaluateAsyncCalledTwice);
    TEST_METHOD(VerifyOutputAfterImageBindCalledTwice);
};
