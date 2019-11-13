//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once
#include "APITests.h"
using namespace WEX::Common;

class LearningModelSessionAPITests : public APITests
{
public:
    TEST_CLASS(LearningModelSessionAPITests);

    TEST_METHOD(EvaluateSessionAndCloseModel);

    TEST_METHOD(CloseSession);
    TEST_METHOD(CreateSessionDeviceDefault);
    TEST_METHOD(CreateSessionDeviceCpu);
    TEST_METHOD(CreateSessionWithModelLoadedFromStream);
    TEST_METHOD(CreateSessionDeviceDirectX);
    TEST_METHOD(CreateSessionDeviceDirectXHighPerformance);
    TEST_METHOD(CreateSessionDeviceDirectXMinimumPower);
    TEST_METHOD(AdapterIdAndDevice);
    TEST_METHOD(EvaluateFeatures);
    TEST_METHOD(EvaluateFeaturesAsync);
    TEST_METHOD(EvaluationProperties);
    TEST_METHOD(CreateSessionWithCastToFloat16InModel);

    BEGIN_TEST_METHOD(CreateSessionWithFloat16InitializersInModel)
        // Bug 21624720: Model fails to resolve due to ORT using incorrect IR version within partition
        //https://microsoft.visualstudio.com/DefaultCollection/OS/_workitems/edit/21624720
        TEST_METHOD_PROPERTY(L"Ignore", L"true")
    END_TEST_METHOD()
};
