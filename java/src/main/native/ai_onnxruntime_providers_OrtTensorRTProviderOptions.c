/*
 * Copyright (c) 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_providers_OrtTensorRTProviderOptions.h"

/*
 * Class:     ai_onnxruntime_providers_OrtTensorRTProviderOptions
 * Method:    create
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_providers_OrtTensorRTProviderOptions_create
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtTensorRTProviderOptionsV2* opts;
    checkOrtStatus(jniEnv,api,api->CreateTensorRTProviderOptions(&opts));
    return (jlong) opts;
}

/*
 * Class:     ai_onnxruntime_providers_OrtTensorRTProviderOptions
 * Method:    add
 * Signature: (JJLjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_providers_OrtTensorRTProviderOptions_add
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jstring key, jstring value) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtTensorRTProviderOptionsV2* opts = (OrtTensorRTProviderOptionsV2*) optionsHandle;
  const char* keyStr = (*jniEnv)->GetStringUTFChars(jniEnv, key, NULL);
  const char* valueStr = (*jniEnv)->GetStringUTFChars(jniEnv, value, NULL);
  checkOrtStatus(jniEnv,api,api->UpdateTensorRTProviderOptions(opts, &keyStr, &valueStr, 1));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,key,keyStr);
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,value,valueStr);
}

/*
 * Class:     ai_onnxruntime_providers_OrtTensorRTProviderOptions
 * Method:    close
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_providers_OrtTensorRTProviderOptions_close
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void)jniEnv; (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  api->ReleaseTensorRTProviderOptions((OrtTensorRTProviderOptionsV2*)handle);
}
