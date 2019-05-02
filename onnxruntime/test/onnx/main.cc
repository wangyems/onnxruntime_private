// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#ifdef _WIN32
#include "getopt.h"
#else
#include <getopt.h>
#include <thread>
#endif
#include "TestResultStat.h"
#include "testenv.h"
#include "runner.h"
#include "sync_api.h"
#include "providers.h"
#include <google/protobuf/stubs/common.h>
#include "core/framework/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"

using namespace onnxruntime;

namespace {
void usage() {
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-j [models]: Specifies the number of models to run simultaneously.\n"
      "\t-A : Disable memory arena\n"
      "\t-c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.\n"
      "\t-r [repeat]: Specifies the number of times to repeat\n"
      "\t-v: verbose\n"
      "\t-n [test_case_name]: Specifies a single test case to run.\n"
      "\t-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda', 'mkldnn', 'tensorrt' or 'ngraph'. Default: 'cpu'.\n"
      "\t-x: Use parallel executor, default (without -x): sequential executor.\n"
      "\t-h: help\n");
}

#ifdef _WIN32
int GetNumCpuCores() {
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
  DWORD returnLength = sizeof(buffer);
  if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
    // try GetSystemInfo
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    if (sysInfo.dwNumberOfProcessors <= 0) {
      ORT_THROW("Fatal error: 0 count processors from GetSystemInfo");
    }
    // This is the number of logical processors in the current group
    return sysInfo.dwNumberOfProcessors;
  }
  int processorCoreCount = 0;
  int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  for (int i = 0; i != count; ++i) {
    if (buffer[i].Relationship == RelationProcessorCore) {
      ++processorCoreCount;
    }
  }
  if (!processorCoreCount) ORT_THROW("Fatal error: 0 count processors from GetLogicalProcessorInformation");
  return processorCoreCount;
}
#else
int GetNumCpuCores() { return std::thread::hardware_concurrency(); }
#endif
}  // namespace

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[], Ort::Env& env) {
#else
int real_main(int argc, char* argv[], Ort::Env& env) {
#endif
  // if this var is not empty, only run the tests with name in this list
  std::vector<std::basic_string<PATH_CHAR_TYPE> > whitelisted_test_cases;
  int concurrent_session_runs = GetNumCpuCores();
  bool enable_cpu_mem_arena = true;
  bool enable_sequential_execution = true;
  int repeat_count = 1;
  int p_models = GetNumCpuCores();
  bool enable_cuda = false;
  bool enable_mkl = false;
  bool enable_ngraph = false;
  bool enable_nuphar = false;
  bool enable_tensorrt = false;
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  {
    int ch;
    while ((ch = getopt(argc, argv, ORT_TSTR("Ac:hj:m:n:r:e:xv"))) != -1) {
      switch (ch) {
        case 'A':
          enable_cpu_mem_arena = false;
          break;
        case 'v':
          logging_level = ORT_LOGGING_LEVEL_INFO;
          break;
        case 'c':
          concurrent_session_runs = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (concurrent_session_runs <= 0) {
            usage();
            return -1;
          }
          break;
        case 'j':
          p_models = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (p_models <= 0) {
            usage();
            return -1;
          }
          break;
        case 'r':
          repeat_count = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (repeat_count <= 0) {
            usage();
            return -1;
          }
          break;
        case 'm':
          // ignore.
          break;
        case 'n':
          // run only some whitelisted tests
          // TODO: parse name str to an array
          whitelisted_test_cases.emplace_back(optarg);
          break;
        case 'e':
          if (!CompareCString(optarg, ORT_TSTR("cpu"))) {
            // do nothing
          } else if (!CompareCString(optarg, ORT_TSTR("cuda"))) {
            enable_cuda = true;
          } else if (!CompareCString(optarg, ORT_TSTR("mkldnn"))) {
            enable_mkl = true;
          } else if (!CompareCString(optarg, ORT_TSTR("ngraph"))) {
            enable_ngraph = true;
          } else if (!CompareCString(optarg, ORT_TSTR("nuphar"))) {
            enable_nuphar = true;
          } else if (!CompareCString(optarg, ORT_TSTR("tensorrt"))) {
            enable_tensorrt = true;
          } else {
            usage();
            return -1;
          }
          break;
        case 'x':
          enable_sequential_execution = false;
          break;
        case '?':
        case 'h':
        default:
          usage();
          return -1;
      }
    }
  }
  if (concurrent_session_runs > 1 && repeat_count > 1) {
    fprintf(stderr, "when you use '-r [repeat]', please set '-c' to 1\n");
    usage();
    return -1;
  }
  argc -= optind;
  argv += optind;
  if (argc < 1) {
    fprintf(stderr, "please specify a test data dir\n");
    usage();
    return -1;
  }

  try {
    env = Ort::Env{logging_level, "Default"};
  } catch (std::exception& ex) {
    fprintf(stderr, "Error creating environment: %s \n", ex.what());
    return -1;
  }

  std::vector<std::basic_string<PATH_CHAR_TYPE> > data_dirs;
  TestResultStat stat;

  for (int i = 0; i != argc; ++i) {
    data_dirs.emplace_back(argv[i]);
  }
  {
    double per_sample_tolerance = 1e-3;
    // when cuda is enabled, set it to a larger value for resolving random MNIST test failure
    double relative_per_sample_tolerance = enable_cuda ? 0.017 : 1e-3;
    Ort::SessionOptions sf;
    if (enable_cpu_mem_arena)
      sf.EnableCpuMemArena();
    else
      sf.DisableCpuMemArena();
    if (enable_sequential_execution)
      sf.EnableSequentialExecution();
    else
      sf.DisableSequentialExecution();
    if (enable_tensorrt) {
#ifdef USE_TENSORRT
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf));
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
#else
      fprintf(stderr, "TensorRT is not supported in this build");
      return -1;
#endif
    }
    if (enable_cuda) {
#ifdef USE_CUDA
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
#else
      fprintf(stderr, "CUDA is not supported in this build");
      return -1;
#endif
    }
    if (enable_nuphar) {
#ifdef USE_NUPHAR
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nuphar(sf, 0, ""));
#else
      fprintf(stderr, "Nuphar is not supported in this build");
      return -1;
#endif
    }
    if (enable_mkl) {
#ifdef USE_MKLDNN
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(sf, enable_cpu_mem_arena ? 1 : 0));
#else
      fprintf(stderr, "MKL-DNN is not supported in this build");
      return -1;
#endif
    }
    if (enable_ngraph) {  //TODO: Re-order the priority?
#ifdef USE_NGRAPH
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_NGraph(sf, "CPU"));
#else
      fprintf(stderr, "nGraph is not supported in this build");
      return -1;
#endif
    }

    std::unordered_set<std::string> cuda_flaky_tests = {
        "fp16_inception_v1", "fp16_shufflenet", "fp16_tiny_yolov2"};

#if (defined(_WIN32) && !defined(_WIN64)) || (defined(__GNUG__) && !defined(__LP64__))
    //Minimize mem consumption
    LoadTests(data_dirs, whitelisted_test_cases, per_sample_tolerance, relative_per_sample_tolerance, [&stat, &sf, enable_cuda, &cuda_flaky_tests](ITestCase* l) {
      std::unique_ptr<ITestCase> test_case_ptr(l);
      if (enable_cuda && cuda_flaky_tests.find(l->GetTestCaseName()) != cuda_flaky_tests.end()) {
        return;
      }
      TestResultStat per_case_stat;
      std::vector<ITestCase*> per_case_tests = {l};
      TestEnv per_case_args(per_case_tests, per_case_stat, sf);
      RunTests(per_case_args, 1, 1, 1, GetDefaultThreadPool(Env::Default()));
      stat += per_case_stat;
    });
#else
    std::vector<ITestCase*> tests;
    LoadTests(data_dirs, whitelisted_test_cases, per_sample_tolerance, relative_per_sample_tolerance, [&tests](ITestCase* l) { tests.push_back(l); });
    if (enable_cuda) {
      for (auto it = tests.begin(); it != tests.end();) {
        auto iter = cuda_flaky_tests.find((*it)->GetTestCaseName());
        if (iter != cuda_flaky_tests.end()) {
          delete *it;
          it = tests.erase(it);
        } else {
          ++it;
        }
      }
    }

    TestEnv args(tests, stat, env, sf);
    Status st = RunTests(args, p_models, concurrent_session_runs, static_cast<size_t>(repeat_count),
                         GetDefaultThreadPool(Env::Default()));
    if (!st.IsOK()) {
      fprintf(stderr, "%s\n", st.ErrorMessage().c_str());
      return -1;
    }
    for (ITestCase* l : tests) {
      delete l;
    }
#endif
    std::string res = stat.ToString();
    fwrite(res.c_str(), 1, res.size(), stdout);
  }
  // clang-format off
  std::map<std::string, std::string> broken_tests{
      {"AvgPool1d", "disable reason"},
      {"AvgPool1d_stride", "disable reason"},
      {"AvgPool2d", "disable reason"},
      {"AvgPool2d_stride", "disable reason"},
      {"AvgPool3d", "disable reason"},
      {"AvgPool3d_stride", "disable reason"},
      {"AvgPool3d_stride1_pad0_gpu_input", "disable reason"},
      {"BatchNorm1d_3d_input_eval", "disable reason"},
      {"BatchNorm2d_eval", "disable reason"},
      {"BatchNorm2d_momentum_eval", "disable reason"},
      {"BatchNorm3d_eval", "disable reason"},
      {"BatchNorm3d_momentum_eval", "disable reason"},
      {"constantofshape_float_ones", "test data bug"},
      {"constantofshape_int_zeros", "test data bug"},
      {"GLU", "disable reason"},
      {"GLU_dim", "disable reason"},
      {"Linear", "disable reason"},
      {"PReLU_1d", "disable reason"},
      {"PReLU_1d_multiparam", "disable reason"},
      {"PReLU_2d", "disable reason"},
      {"PReLU_2d_multiparam", "disable reason"},
      {"PReLU_3d", "disable reason"},
      {"PReLU_3d_multiparam", "disable reason"},
      {"PoissonNLLLLoss_no_reduce", "disable reason"},
      {"Softsign", "disable reason"},
      {"convtranspose_1d", "disable reason"},
      {"convtranspose_3d", "disable reason"},
      {"flatten_axis0", "disable reason"},
      {"flatten_axis1", "disable reason"},
      {"flatten_axis2", "disable reason"},
      {"flatten_axis3", "disable reason"},
      {"flatten_default_axis", "disable reason"},
      {"gemm_broadcast", "disable reason"},
      {"gemm_nobroadcast", "disable reason"},
      {"greater", "disable reason"},
      {"greater_bcast", "disable reason"},
      {"less", "disable reason"},
      {"less_bcast", "disable reason"},
      {"matmul_2d", "disable reason"},
      {"matmul_3d", "disable reason"},
      {"matmul_4d", "disable reason"},
      {"mvn", "disable reason"},
      {"operator_add_broadcast", "disable reason"},
      {"operator_add_size1_broadcast", "disable reason"},
      {"operator_add_size1_right_broadcast", "disable reason"},
      {"operator_add_size1_singleton_broadcast", "disable reason"},
      {"operator_addconstant", "disable reason"},
      {"operator_addmm", "disable reason"},
      {"operator_basic", "disable reason"},
      {"operator_lstm", "disable reason"},
      {"operator_mm", "disable reason"},
      {"operator_non_float_params", "disable reason"},
      {"operator_params", "disable reason"},
      {"operator_pow", "disable reason"},
      {"operator_rnn", "disable reason"},
      {"operator_rnn_single_layer", "disable reason"},
      {"prelu_broadcast", "disable reason"},
      {"prelu_example", "disable reason"},
      {"cast_STRING_to_FLOAT", "Cast opset 9 not supported yet"},
      {"cast_FLOAT_to_STRING", "Cast opset 9 not supported yet"},
      {"tf_inception_resnet_v2", "Cast opset 9 not supported yet"},
      {"tf_inception_v4", "Cast opset 9 not supported yet"},
      {"tf_nasnet_large", "disable temporarily"},
      {"tf_nasnet_mobile", "disable temporarily"},
      {"tf_pnasnet_large", "disable temporarily"},
      {"shrink", "test case is wrong"},
      {"maxpool_2d_precomputed_strides", "ShapeInferenceError"},
      {"averagepool_2d_precomputed_strides", "ShapeInferenceError"},
      {"maxpool_with_argmax_2d_precomputed_strides", "ShapeInferenceError"},
      {"tf_inception_v2", "result mismatch"},
      {"tf_mobilenet_v2_1.0_224", "result mismatch"},
      {"tf_mobilenet_v2_1.4_224", "result mismatch"},
      {"tf_mobilenet_v1_1.0_224", "result mismatch"},
      {"mobilenetv2-1.0", "result mismatch"},
      {"mxnet_arcface", "result mismatch"},
      {"mod_float_mixed_sign_example", "faulty test"}
  };

#ifdef USE_NGRAPH
  broken_tests["dequantizelinear"] = "ambiguity in scalar dimensions [] vs [1]";
  broken_tests["qlinearconv"] = "ambiguity in scalar dimensions [] vs [1]";
  broken_tests["quantizelinear"] = "ambiguity in scalar dimensions [] vs [1]";
#endif

#ifdef USE_CUDA
  broken_tests["mxnet_arcface"] = "result mismatch";
  broken_tests["tf_inception_v1"] = "flaky test"; //TODO: Investigate cause for flakiness
#endif
  // clang-format on

#if defined(_WIN32) && !defined(_WIN64)
  broken_tests["vgg19"] = "failed: bad allocation";
#endif

#if defined(__GNUG__) && !defined(__LP64__)
  broken_tests["nonzero_example"] = "failed: type mismatch";
#endif

#ifdef DISABLE_CONTRIB_OPS
  broken_tests["coreml_SqueezeNet_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_Permute_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_ReLU_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_Padding-Upsampling-Normalizer_ImageNet"] = "This model uses contrib ops.";
  broken_tests["tiny_yolov2"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_Pooling_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_Padding_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_Normalizer_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_linear_sklearn_load_breast_cancer"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_linear_ImageNet_small"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_linear_ImageNet_large"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_linear_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_leakyrelu_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_hard_sigmoid_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_elu_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_Dense_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_Conv2D_ImageNet"] = "This model uses contrib ops.";
  broken_tests["coreml_VGG16_ImageNet"] = "This model uses contrib ops.";
  broken_tests["coreml_Resnet50_ImageNet"] = "This model uses contrib ops.";
  broken_tests["coreml_Inceptionv3_ImageNet"] = "This model uses contrib ops.";
  broken_tests["coreml_FNS-Candy_ImageNet"] = "This model uses contrib ops.";
  broken_tests["coreml_AgeNet_ImageNet"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_thresholdedrelu_ImageNet_large"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_thresholdedrelu_ImageNet_small"] = "This model uses contrib ops.";
  broken_tests["keras2coreml_thresholdedrelu_sklearn_load_breast_cancer"] = "This model uses contrib ops.";
  broken_tests["thresholdedrelu"] = "This model uses contrib ops.";
  broken_tests["thresholdedrelu_default"] = "This model uses contrib ops.";
  broken_tests["dynamic_slice_default_axes"] = "This model uses contrib ops.";
  broken_tests["thresholdedrelu_example"] = "This model uses contrib ops.";
  broken_tests["dynamic_slice_neg failed"] = "This model uses contrib ops.";
  broken_tests["dynamic_slice_start_out_of_bounds"] = "This model uses contrib ops.";
  broken_tests["dynamic_slice"] = "This model uses contrib ops.";
  broken_tests["dynamic_slice_end_out_of_bounds"] = "This model uses contrib ops.";
  broken_tests["dynamic_slice_neg"] = "This model uses contrib ops.";
#endif

  int result = 0;
  for (const std::string& s : stat.GetFailedTest()) {
    if (broken_tests.find(s) == broken_tests.end()) {
      fprintf(stderr, "test %s failed, please fix it\n", s.c_str());
      result = -1;
    }
  }

  return result;
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  Ort::Env env{nullptr};
  int retval = -1;
  try {
    retval = real_main(argc, argv, env);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    retval = -1;
  }
  // Release the protobuf library if we failed to create an env (the env will release it automatically on destruction)
  if (!env) {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
  return retval;
}
