REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0\lib
set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\v12.2\bin;%AGENT_TEMPDIRECTORY%\v12.2\extras\CUPTI\lib64
set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY
