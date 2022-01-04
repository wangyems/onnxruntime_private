/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include "core/platform/env_time.h"

#include <time.h>
#include <windows.h>
#include <chrono>
#include <numeric>
#include <algorithm>

namespace onnxruntime {

namespace {

class WindowsEnvTime : public EnvTime {
 public:
  uint64_t NowMicros() override {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch())
        .count();
  }

  void SleepForMicroseconds(int64_t micros) { Sleep(static_cast<DWORD>(micros) / 1000); }
};

}  // namespace

EnvTime* EnvTime::Default() {
  static WindowsEnvTime default_time_env;
  return &default_time_env;
}

bool GetMonotonicTimeCounter(TIME_SPEC* value) {
  static_assert(sizeof(LARGE_INTEGER) == sizeof(TIME_SPEC), "type mismatch");
  return QueryPerformanceCounter((LARGE_INTEGER*)value) != 0;
}

static INIT_ONCE g_InitOnce = INIT_ONCE_STATIC_INIT;
static LARGE_INTEGER freq;

BOOL CALLBACK InitHandleFunction(
    PINIT_ONCE,
    PVOID,
    PVOID*) {
  return QueryPerformanceFrequency(&freq);
}

void SetTimeSpecToZero(TIME_SPEC* value) {
  *value = 0;
}

void AccumulateTimeSpec(TIME_SPEC* base, const TIME_SPEC* start, const TIME_SPEC* end) {
  *base += std::max<TIME_SPEC>(0, *end - *start);
}

//Return the interval in seconds.
//If the function fails, the return value is zero
double TimeSpecToSeconds(const TIME_SPEC* value) {
  BOOL initState = InitOnceExecuteOnce(&g_InitOnce, InitHandleFunction, nullptr, nullptr);
  if (!initState) return 0;
  return *value / (double)freq.QuadPart;
}

}  // namespace onnxruntime
