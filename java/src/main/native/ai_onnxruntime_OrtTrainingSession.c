/*
 * Copyright (c) 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include "OrtJniUtil.h"
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime_training_c_api.h"
#include "ai_onnxruntime_OrtTrainingSession.h"

const char * const ORTJNI_StringClassName = "java/lang/String";
const char * const ORTJNI_OnnxValueClassName = "ai/onnxruntime/OnnxValue";

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    createTrainingSession
 * Signature: (JJJJJLjava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtTrainingSession_createTrainingSession
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jlong trainApiHandle,
     jlong envHandle, jlong optionsHandle, jlong checkpointHandle,
     jstring trainPath, jstring evalPath, jstring  optimizerPath) {
  (void) clazz; // Required JNI parameters not needed by functions which don't need to access their host class.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*) trainApiHandle;
  const OrtEnv* env = (const OrtEnv*) envHandle;
  const OrtSessionOptions* options = (const OrtSessionOptions*) optionsHandle;
  OrtCheckpointState* checkpoint = (OrtCheckpointState*) checkpointHandle;

  OrtTrainingSession* session = NULL;

#ifdef _WIN32
  // The output of GetStringChars is not null-terminated, so we copy it and add a terminator
  const jchar* trainJavaStr = (*jniEnv)->GetStringChars(jniEnv, trainPath, NULL);
  size_t trainStrLength = (*jniEnv)->GetStringLength(jniEnv, trainPath);
  wchar_t* trainStr = (wchar_t*)calloc(trainStrLength + 1, sizeof(wchar_t));
  if (trainStr == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, trainPath, trainJavaStr);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return (jlong) session;
  }
  const jchar* evalJavaStr = (*jniEnv)->GetStringChars(jniEnv, evalPath, NULL);
  size_t evalStrLength = (*jniEnv)->GetStringLength(jniEnv, evalPath);
  wchar_t* evalStr = (wchar_t*)calloc(evalStrLength + 1, sizeof(wchar_t));
  if (evalStr == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, trainPath, trainJavaStr);
    (*jniEnv)->ReleaseStringChars(jniEnv, evalPath, evalJavaStr);
    free(trainStr);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return (jlong) session;
  }
  const jchar* optimizerJavaStr = (*jniEnv)->GetStringChars(jniEnv, optimizerPath, NULL);
  size_t optimizerStrLength = (*jniEnv)->GetStringLength(jniEnv, optimizerPath);
  wchar_t* optimizerStr = (wchar_t*)calloc(optimizerStrLength + 1, sizeof(wchar_t));
  if (optimizerStr == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, trainPath, trainJavaStr);
    (*jniEnv)->ReleaseStringChars(jniEnv, evalPath, evalJavaStr);
    (*jniEnv)->ReleaseStringChars(jniEnv, optimizerPath, optimizerJavaStr);
    free(trainStr);
    free(evalStr);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return (jlong) session;
  }
  wcsncpy_s(trainStr, trainStrLength + 1, (const wchar_t*)trainJavaStr, trainStrLength);
  wcsncpy_s(evalStr, evalStrLength + 1, (const wchar_t*)evalJavaStr, evalStrLength);
  wcsncpy_s(optimizerStr, optimizerStrLength + 1, (const wchar_t*)optimizerJavaStr, optimizerStrLength);
  (*jniEnv)->ReleaseStringChars(jniEnv, trainPath, trainJavaStr);
  (*jniEnv)->ReleaseStringChars(jniEnv, evalPath, evalJavaStr);
  (*jniEnv)->ReleaseStringChars(jniEnv, optimizerPath, optimizerJavaStr);
  checkOrtStatus(jniEnv, api, trainApi->CreateTrainingSession(env, options, checkpoint, trainStr, evalStr, optimizerStr, &session));
  free(trainStr);
  free(evalStr);
  free(optimizerStr);
#else
  // GetStringUTFChars is null terminated, so can be used directly
  const char* trainStr = (*jniEnv)->GetStringUTFChars(jniEnv, trainPath, NULL);
  const char* evalStr = (*jniEnv)->GetStringUTFChars(jniEnv, evalPath, NULL);
  const char* optimizerStr = (*jniEnv)->GetStringUTFChars(jniEnv, optimizerPath, NULL);
  checkOrtStatus(jniEnv, api, trainApi->CreateTrainingSession(env, options, checkpoint, trainStr, evalStr, optimizerStr, &session));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, trainPath, trainStr);
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, evalPath, evalStr);
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, optimizerPath, optimizerStr);
#endif

  return (jlong) session;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    closeSession
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_closeSession
    (JNIEnv * jniEnv, jobject jobj, jlong trainHandle, jlong nativeHandle) {
  (void)jniEnv; (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainHandle;
  trainApi->ReleaseTrainingSession((OrtTrainingSession*)nativeHandle);
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    saveCheckpoint
 * Signature: (JJJLjava/lang/String;Z)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_saveCheckpoint
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainingApiHandle, jlong nativeHandle, jstring outputPath, jboolean overwrite) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*) trainingApiHandle;

  const OrtTrainingSession* trainSession = (const OrtTrainingSession*) nativeHandle;

#ifdef _WIN32
  // The output of GetStringChars is not null-terminated, so we copy it and add a terminator
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, outputPath, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, outputPath);
  wchar_t* newString = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (newString == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, outputPath, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
  } else {
    wcsncpy_s(newString, stringLength + 1, (const wchar_t*)cPath, stringLength);
    checkOrtStatus(jniEnv, api,
                   trainApi->SaveCheckpoint(newString, trainSession, overwrite));
    free(newString);
    (*jniEnv)->ReleaseStringChars(jniEnv, outputPath, cPath);
  }
#else
  // GetStringUTFChars is null terminated, so can be used directly
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, outputPath, NULL);
  checkOrtStatus(jniEnv, api, trainApi->SaveCheckpoint(cPath, trainSession, overwrite));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, outputPath, cPath);
#endif
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getTrainOutputNames
 * Signature: (JJJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getTrainOutputNames
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  const OrtTrainingSession* trainSession = (const OrtTrainingSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, ORTJNI_StringClassName);

  // Get the number of outputs
  size_t numOutputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetTrainingModelOutputCount(trainSession, &numOutputs));
  if (code != ORT_OK) {
    return NULL;
  }

  int32_t numOutputsInt = (int32_t) numOutputs;
  if (numOutputs != (size_t) numOutputsInt) {
    throwOrtException(jniEnv, 1, "Too many outputs, expected less than 2^31");
  }

  // Allocate the return array
  jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv, numOutputsInt, stringClazz, NULL);
  for (int32_t i = 0; i < numOutputsInt; i++) {
    // Read out the output name and convert it to a java.lang.String
    char* outputName = NULL;
    code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetTrainingModelOutputName(trainSession, i, allocator, &outputName));
    if (code != ORT_OK) {
      // break out on error, return array and let Java throw the exception.
      break;
    }
    jstring name = (*jniEnv)->NewStringUTF(jniEnv, outputName);
    (*jniEnv)->SetObjectArrayElement(jniEnv, array, i, name);
    code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, outputName));
    if (code != ORT_OK) {
      // break out on error, return array and let Java throw the exception.
      break;
    }
  }

  return array;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getEvalOutputNames
 * Signature: (JJJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getEvalOutputNames
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  const OrtTrainingSession* trainSession = (const OrtTrainingSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, ORTJNI_StringClassName);

  // Get the number of outputs
  size_t numOutputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetEvalModelOutputCount(trainSession, &numOutputs));
  if (code != ORT_OK) {
    return NULL;
  }

  int32_t numOutputsInt = (int32_t) numOutputs;
  if (numOutputs != (size_t) numOutputsInt) {
    throwOrtException(jniEnv, 1, "Too many outputs, expected less than 2^31");
  }

  // Allocate the return array
  jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv, numOutputsInt, stringClazz, NULL);
  for (int32_t i = 0; i < numOutputsInt; i++) {
    // Read out the output name and convert it to a java.lang.String
    char* outputName = NULL;
    code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetEvalModelOutputName(trainSession, i, allocator, &outputName));
    if (code != ORT_OK) {
      // break out on error, return array and let Java throw the exception.
      break;
    }
    jstring name = (*jniEnv)->NewStringUTF(jniEnv, outputName);
    (*jniEnv)->SetObjectArrayElement(jniEnv, array, i, name);
    code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, outputName));
    if (code != ORT_OK) {
      // break out on error, return array and let Java throw the exception.
      break;
    }
  }

  return array;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    resetGrad
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_resetGrad
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  checkOrtStatus(jniEnv, api, trainApi->ResetGrad(trainSession));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    trainStep
 * Signature: (JJJJ[Ljava/lang/String;[JJ[Ljava/lang/String;JJ)[Lai/onnxruntime/OnnxValue;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_trainStep
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle,
     jlong nativeHandle, jlong allocatorHandle, jobjectArray inputNamesArr, jlongArray inputHandles, jlong numInputs,
     jobjectArray outputNamesArr, jlong numOutputs, jlong runOptionsHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  OrtRunOptions* runOptions = (OrtRunOptions*)runOptionsHandle;

  jobjectArray outputArray = NULL;

  // Create the buffers for the Java input & output strings, and the input pointers
  const char** inputNames = malloc(sizeof(char*) * numInputs);
  if (inputNames == NULL) {
    // Nothing to cleanup, return and throw exception
    return outputArray;
  }
  const char** outputNames = malloc(sizeof(char*) * numOutputs);
  if (outputNames == NULL) {
    goto cleanup_input_names;
  }
  jobject* javaInputStrings = malloc(sizeof(jobject) * numInputs);
  if (javaInputStrings == NULL) {
    goto cleanup_output_names;
  }
  jobject* javaOutputStrings = malloc(sizeof(jobject) * numOutputs);
  if (javaOutputStrings == NULL) {
    goto cleanup_java_input_strings;
  }
  const OrtValue** inputValuePtrs = malloc(sizeof(OrtValue*) * numInputs);
  if (inputValuePtrs == NULL) {
    goto cleanup_java_output_strings;
  }
  OrtValue** outputValues = malloc(sizeof(OrtValue*) * numOutputs);
  if (outputValues == NULL) {
    goto cleanup_input_values;
  }

  // Extract a C array of longs which are pointers to the input tensors.
  // The Java-side objects store native pointers as 64-bit longs, and on 32-bit systems
  // we cannot cast the long array to a pointer array as they are different sizes,
  // so we copy the longs applying the appropriate cast.
  jlong* inputValueLongs = (*jniEnv)->GetLongArrayElements(jniEnv, inputHandles, NULL);

  // Extract the names and native pointers of the input values.
  for (int i = 0; i < numInputs; i++) {
    javaInputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, inputNamesArr, i);
    inputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaInputStrings[i], NULL);
    inputValuePtrs[i] = (OrtValue*)inputValueLongs[i];
  }

  // Release the java array copy of pointers to the tensors.
  (*jniEnv)->ReleaseLongArrayElements(jniEnv, inputHandles, inputValueLongs, JNI_ABORT);

  // Extract the names of the output values.
  for (int i = 0; i < numOutputs; i++) {
    javaOutputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, outputNamesArr, i);
    outputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaOutputStrings[i], NULL);
    outputValues[i] = NULL;
  }

  // Actually score the inputs.
  //ORT_API2_STATUS(TrainStep, _Inout_ OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
  //                size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
  //                size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->TrainStep(trainSession, runOptions,
                                                                      numInputs, (const OrtValue* const*)inputValuePtrs,
                                                                      numOutputs, outputValues));
  if (code != ORT_OK) {
    goto cleanup_output_values;
  }

  // Construct the output array of ONNXValues
  jclass onnxValueClass = (*jniEnv)->FindClass(jniEnv, ORTJNI_OnnxValueClassName);
  outputArray = (*jniEnv)->NewObjectArray(jniEnv, safecast_int64_to_jsize(numOutputs), onnxValueClass, NULL);

  // Convert the output tensors into ONNXValues
  for (int i = 0; i < numOutputs; i++) {
    if (outputValues[i] != NULL) {
      jobject onnxValue = convertOrtValueToONNXValue(jniEnv, api, allocator, outputValues[i]);
      if (onnxValue == NULL) {
        break;  // go to cleanup, exception thrown
      }
      (*jniEnv)->SetObjectArrayElement(jniEnv, outputArray, i, onnxValue);
    }
  }

  // Note these gotos are in a specific order so they mirror the allocation pattern above.
  // They must be changed if the allocation code is rearranged.
  cleanup_output_values:
  free(outputValues);

  // Release the Java output strings
  for (int i = 0; i < numOutputs; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaOutputStrings[i], outputNames[i]);
  }

  // Release the Java input strings
  for (int i = 0; i < numInputs; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaInputStrings[i], inputNames[i]);
  }

  // Release the buffers
  cleanup_input_values:
  free((void*)inputValuePtrs);
  cleanup_java_output_strings:
  free(javaOutputStrings);
  cleanup_java_input_strings:
  free(javaInputStrings);
  cleanup_output_names:
  free((void*)outputNames);
  cleanup_input_names:
  free((void*)inputNames);

  return outputArray;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    evalStep
 * Signature: (JJJJ[Ljava/lang/String;[JJ[Ljava/lang/String;JJ)[Lai/onnxruntime/OnnxValue;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_evalStep
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle,
     jlong nativeHandle, jlong allocatorHandle, jobjectArray inputNamesArr, jlongArray inputHandles, jlong numInputs,
     jobjectArray outputNamesArr, jlong numOutputs, jlong runOptionsHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  OrtRunOptions* runOptions = (OrtRunOptions*)runOptionsHandle;

  jobjectArray outputArray = NULL;

  // Create the buffers for the Java input & output strings, and the input pointers
  const char** inputNames = malloc(sizeof(char*) * numInputs);
  if (inputNames == NULL) {
    // Nothing to cleanup, return and throw exception
    return outputArray;
  }
  const char** outputNames = malloc(sizeof(char*) * numOutputs);
  if (outputNames == NULL) {
    goto cleanup_input_names;
  }
  jobject* javaInputStrings = malloc(sizeof(jobject) * numInputs);
  if (javaInputStrings == NULL) {
    goto cleanup_output_names;
  }
  jobject* javaOutputStrings = malloc(sizeof(jobject) * numOutputs);
  if (javaOutputStrings == NULL) {
    goto cleanup_java_input_strings;
  }
  const OrtValue** inputValuePtrs = malloc(sizeof(OrtValue*) * numInputs);
  if (inputValuePtrs == NULL) {
    goto cleanup_java_output_strings;
  }
  OrtValue** outputValues = malloc(sizeof(OrtValue*) * numOutputs);
  if (outputValues == NULL) {
    goto cleanup_input_values;
  }

  // Extract a C array of longs which are pointers to the input tensors.
  // The Java-side objects store native pointers as 64-bit longs, and on 32-bit systems
  // we cannot cast the long array to a pointer array as they are different sizes,
  // so we copy the longs applying the appropriate cast.
  jlong* inputValueLongs = (*jniEnv)->GetLongArrayElements(jniEnv, inputHandles, NULL);

  // Extract the names and native pointers of the input values.
  for (int i = 0; i < numInputs; i++) {
    javaInputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, inputNamesArr, i);
    inputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaInputStrings[i], NULL);
    inputValuePtrs[i] = (OrtValue*)inputValueLongs[i];
  }

  // Release the java array copy of pointers to the tensors.
  (*jniEnv)->ReleaseLongArrayElements(jniEnv, inputHandles, inputValueLongs, JNI_ABORT);

  // Extract the names of the output values.
  for (int i = 0; i < numOutputs; i++) {
    javaOutputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, outputNamesArr, i);
    outputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaOutputStrings[i], NULL);
    outputValues[i] = NULL;
  }

  // Actually score the inputs.
  //ORT_API2_STATUS(EvalStep, _In_ const OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
  //                size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
  //                size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->EvalStep(trainSession, runOptions,
                                                                      numInputs, (const OrtValue* const*)inputValuePtrs,
                                                                      numOutputs, outputValues));
  if (code != ORT_OK) {
    goto cleanup_output_values;
  }

  // Construct the output array of ONNXValues
  jclass onnxValueClass = (*jniEnv)->FindClass(jniEnv, ORTJNI_OnnxValueClassName);
  outputArray = (*jniEnv)->NewObjectArray(jniEnv, safecast_int64_to_jsize(numOutputs), onnxValueClass, NULL);

  // Convert the output tensors into ONNXValues
  for (int i = 0; i < numOutputs; i++) {
    if (outputValues[i] != NULL) {
      jobject onnxValue = convertOrtValueToONNXValue(jniEnv, api, allocator, outputValues[i]);
      if (onnxValue == NULL) {
        break;  // go to cleanup, exception thrown
      }
      (*jniEnv)->SetObjectArrayElement(jniEnv, outputArray, i, onnxValue);
    }
  }

  // Note these gotos are in a specific order so they mirror the allocation pattern above.
  // They must be changed if the allocation code is rearranged.
  cleanup_output_values:
  free(outputValues);

  // Release the Java output strings
  for (int i = 0; i < numOutputs; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaOutputStrings[i], outputNames[i]);
  }

  // Release the Java input strings
  for (int i = 0; i < numInputs; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaInputStrings[i], inputNames[i]);
  }

  // Release the buffers
  cleanup_input_values:
  free((void*)inputValuePtrs);
  cleanup_java_output_strings:
  free(javaOutputStrings);
  cleanup_java_input_strings:
  free(javaInputStrings);
  cleanup_output_names:
  free((void*)outputNames);
  cleanup_input_names:
  free((void*)inputNames);

  return outputArray;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    setLearningRate
 * Signature: (JJJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_setLearningRate
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle, jfloat learningRate) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  checkOrtStatus(jniEnv, api, trainApi->SetLearningRate(trainSession, learningRate));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getLearningRate
 * Signature: (JJJ)F
 */
JNIEXPORT jfloat JNICALL Java_ai_onnxruntime_OrtTrainingSession_getLearningRate
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  jfloat learningRate = 0.0f;
  checkOrtStatus(jniEnv, api, trainApi->GetLearningRate(trainSession, &learningRate));
  return learningRate;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    optimizerStep
 * Signature: (JJJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_optimizerStep
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle, jlong runOptionsHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  const OrtRunOptions* options = (const OrtRunOptions*) runOptionsHandle;
  checkOrtStatus(jniEnv, api, trainApi->OptimizerStep(trainSession, options));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    registerLinearLRScheduler
 * Signature: (JJJJJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_registerLinearLRScheduler
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle, jlong warmupSteps, jlong totalSteps, jfloat initialLearningRate) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  checkOrtStatus(jniEnv, api, trainApi->RegisterLinearLRScheduler(trainSession, warmupSteps, totalSteps, initialLearningRate));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    schedulerStep
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_schedulerStep
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  checkOrtStatus(jniEnv, api, trainApi->SchedulerStep(trainSession));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    exportModelForInference
 * Signature: (JJJJLjava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_exportModelForInference
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle, jstring outputPath, jlong numOutputs, jobjectArray outputNamesArr) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;

  // prep output names array
  const char** outputNames = malloc(sizeof(char*) * numOutputs);
  if (outputNames == NULL) {
    return;
  }
  jobject* javaOutputStrings = malloc(sizeof(jobject) * numOutputs);
  if (javaOutputStrings == NULL) {
    free(outputNames);
  }
  // Extract the names of the output values.
  for (int i = 0; i < numOutputs; i++) {
    javaOutputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, outputNamesArr, i);
    outputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaOutputStrings[i], NULL);
  }

#ifdef _WIN32
  // The output of GetStringChars is not null-terminated, so we copy it and add a terminator
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, outputPath, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, outputPath);
  wchar_t* outputStr = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (outputStr == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, outputPath, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
    goto cleanup_array;
  }
  wcsncpy_s(outputStr, stringLength + 1, (const wchar_t*)cPath, stringLength);
  (*jniEnv)->ReleaseStringChars(jniEnv, outputPath, cPath);
  checkOrtStatus(jniEnv, api, trainApi->ExportModelForInferencing(trainSession, outputStr, numOutputs, outputNames));
  free(outputStr);
#else
  // GetStringUTFChars is null terminated, so can be used directly
  const char* outputStr = (*jniEnv)->GetStringUTFChars(jniEnv, outputPath, NULL);
  checkOrtStatus(jniEnv, api, trainApi->ExportModelForInferencing(trainSession, outputStr, numOutputs, outputNames));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, outputPath, outputStr);
#endif

cleanup_array:
  // Release the Java output strings
  for (int i = 0; i < numOutputs; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaOutputStrings[i], outputNames[i]);
  }
  free(javaOutputStrings);
  free(outputNames);
}
