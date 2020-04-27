// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"

struct RawApiTestsApi
{
  SetupClass RawApiTestsApiTestsClassSetup;
  VoidTest CreateModelFromFilePath;
  VoidTest CreateCpuDevice;
  VoidTest Evaluate;
  VoidTest EvaluateNoInputCopy;
};

const RawApiTestsApi& getapi();

WINML_TEST_CLASS_BEGIN(RawApiTests)
WINML_TEST_CLASS_SETUP_CLASS(RawApiTestsApiTestsClassSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(RawApiTests, CreateModelFromFilePath)
WINML_TEST(RawApiTests, CreateCpuDevice)
WINML_TEST(RawApiTests, Evaluate)
WINML_TEST(RawApiTests, EvaluateNoInputCopy)
WINML_TEST_CLASS_END()