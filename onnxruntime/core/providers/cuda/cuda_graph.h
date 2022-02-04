#pragma once
#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

using CaptureId_t = unsigned long long;

struct CUDAGraph {
  CUDAGraph() {}
  CUDAGraph(cudaStream_t stream);
  ~CUDAGraph();

  void CaptureBegin();
  void CaptureEnd();
  Status Replay();
  void Reset();
  void SetStream(cudaStream_t stream);

private:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif

  bool has_graph_ = false;
  bool has_graph_exec_ = false;

  CaptureId_t id_;
  cudaStream_t capture_stream_ = nullptr;
  OrtMutex lock_;
  };
 
} // namespace onnxruntime
