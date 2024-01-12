set(NEURAL_SPEED_URL https://github.com/intel/neural-speed.git)
set(NEURAL_SPEED_TAG 7601eba5849138f8f246d6c7f5483b2efd8ea1f1) # kernel-only release v0.1

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND onnxruntime_target_platform STREQUAL "x86_64")
  set(USE_NEURAL_SPEED TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND onnxruntime_target_platform STREQUAL "x64")
  set(USE_NEURAL_SPEED TRUE)
endif()

if(USE_NEURAL_SPEED)
  FetchContent_Declare(
      neural_speed
      GIT_REPOSITORY ${NEURAL_SPEED_URL}
      GIT_TAG        ${NEURAL_SPEED_TAG}
  )
  FetchContent_MakeAvailable(neural_speed)
  add_compile_definitions(ORT_NEURAL_SPEED)
endif()