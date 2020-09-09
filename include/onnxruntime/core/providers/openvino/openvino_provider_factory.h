// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

/**
 * \param device_id openvino device id, starts from zero.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_OpenVINO,
    _In_ OrtSessionOptions* options, const char* device_id, bool enable_vpu_fast_compile);

#ifdef __cplusplus
}
#endif
