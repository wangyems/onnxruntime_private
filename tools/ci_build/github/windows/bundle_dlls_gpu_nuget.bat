REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM for available runtime identifiers, see https://github.com/dotnet/corefx/blob/release/3.1/pkg/Microsoft.NETCore.Platforms/runtime.json
set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion 
FOR /R %%i IN (*.nupkg) do (
   set filename=%%~ni
   IF NOT "!filename:~25,7!"=="Managed" (
       mkdir build\native\include
       copy %BUILD_SOURCESDIRECTORY%\include\onnxruntime\core\providers\tensorrt\tensorrt_provider_factory.h build\native\include\tensorrt_provider_factory.h 
       7z a  %%~ni.nupkg build 
   )
) 
