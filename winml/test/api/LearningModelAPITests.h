//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once
#include "APITests.h"
using namespace WEX::Common;

class LearningModelAPITests : public APITests
{
public:
    TEST_CLASS(LearningModelAPITests);
    TEST_CLASS_SETUP(TestClassSetup);
    TEST_METHOD_SETUP(TestMethodSetup);

    TEST_METHOD(CreateModelFromFilePath);
    TEST_METHOD(CreateModelFromIStorage);
    TEST_METHOD(CreateModelFromIStorageOutsideCwd);
    TEST_METHOD(CreateModelFromIStream);
    TEST_METHOD(GetAuthor);
    TEST_METHOD(GetName);
    TEST_METHOD(GetDomain);
    TEST_METHOD(GetDescription);
    TEST_METHOD(GetVersion);
    BEGIN_TEST_METHOD(GetMetaData)
        TEST_METHOD_PROPERTY(L"DataSource", L"Table:metaDataTestTable.xml#metaDataTable")
    END_TEST_METHOD()
    TEST_METHOD(EnumerateInputs);
    TEST_METHOD(EnumerateOutputs);
    TEST_METHOD(CloseModelCheckMetadata);
    TEST_METHOD(CloseModelCheckEval);
    TEST_METHOD(CloseModelNoNewSessions);

private:
    void MetaDataVerifyHelper(
        WEX::Common::String expectedKeyParam,
        WEX::Common::String expectedValueParam,
        winrt::Windows::Foundation::Collections::IIterator<
        winrt::Windows::Foundation::Collections::IKeyValuePair<winrt::hstring, winrt::hstring>> iter);

};
