// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"
#include "core/training/training_optimizer.h"
#include "core/training/training_session.h"
#include "core/training/weight_updater.h"
#include "test/training/runner/training_runner.h"
#include "test/training/runner/training_util.h"
#include "test/training/runner/data_loader.h"

#ifdef USE_HOROVOD
#include "core/graph/training/horovod_adapters.h"
#include <mpi.h>
#endif

#include <tuple>

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace std;

const static float LEARNING_RATE = 1e-4f;
const PATH_STRING_TYPE TRAINING_DATA_PATH = ORT_TSTR("bert_data/train");
const PATH_STRING_TYPE TEST_DATA_PATH = ORT_TSTR("bert_data/test");

struct TrainingConfig {
  TrainingConfig() : model_name(""), num_of_epoch(0), num_of_training_samples(0), num_of_testing_samples(0), batch_size(1) {}
  std::string model_name;
  int num_of_epoch;
  int num_of_training_samples;
  int num_of_testing_samples;
  int batch_size;
  bool is_perf_test;
};

int validate_params(int argc, char* args[], TrainingConfig& config) {
  if (argc < 7) {
    printf("Incorrect command line for %s\n", args[0]);
    printf("usage: exe_name model_name num_of_epoch num_of_training_samples num_of_testing_samples batch_size [train|perf] [optional:world_rank]\n");
    return -1;
  }
  config.model_name = args[1];
  config.num_of_epoch = stoi(args[2]);
  config.num_of_training_samples = stoi(args[3]);
  config.num_of_testing_samples = stoi(args[4]);
  config.batch_size = stoi(args[5]);

  auto mode = string(args[6]);
  if (mode == "perf" || mode == "train") {
    config.is_perf_test = mode == "perf";
  } else {
    printf("Incorrect command line for mode: it must be one of [perf|train]\n");
    return -1;
  }

  return 0;
}

#ifdef USE_HOROVOD
std::pair<int, int> setup_horovod() {
  using namespace horovod::common;
  // setup MPI amd horovod
  MPI_Init(0, 0);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int* ranks = (int*)malloc(sizeof(int) * world_size);

  MPI_Allgather(&world_rank, 1, MPI_INT, ranks, 1, MPI_INT, MPI_COMM_WORLD);

  horovod_init(ranks, world_size);

  return {world_rank, world_size};
}

void shutdown_horovod() {
  horovod::common::horovod_shutdown();
  MPI_Finalize();
}

#endif

// NOTE: these variables need to be alive when the error_function is called.
int true_count = 0;
float total_loss = 0.0f;

void setup_training_params(const TrainingConfig& config, TrainingRunner::Parameters& params) {
  params.model_path_ = config.model_name + ".onnx";
  params.model_with_loss_func_path_ = config.model_name + "_with_cost.onnx";
  params.model_with_training_graph_path_ = config.model_name + "_bw.onnx";
  params.model_actual_running_graph_path_ = config.model_name + "_bw_running.onnx";
  params.model_trained_path_ = config.model_name + "_trained.onnx";
  params.model_trained_with_loss_func_path_ = config.model_name + "_with_cost_trained.onnx";
  params.batch_size_ = config.batch_size;
  params.num_of_samples_for_evaluation_ = config.num_of_testing_samples;
  params.num_of_epoch_ = config.num_of_epoch;
  params.loss_func_info_ = LossFunctionInfo(OpDef("BertLoss", kOnnxDomain),
                                            "total_loss",
                                            {/*prediction_masked_lm*/ "output1",
                                             /*prediction_next_sentence*/ "output2",
                                             /*masked_lm_positions*/ "masked_lm_positions",
                                             /*masked_lm_ids*/ "masked_lm_ids",
                                             /*masked_lm_weights*/ "masked_lm_weights",
                                             /*next_sentence_labels*/ "next_sentence_labels",
                                             /*batch_size*/ std::to_string(params.batch_size_),
                                             /*max_sequence_len*/ std::to_string(512),
                                             /*max_predictions_per_sequence*/ std::to_string(80)});
  params.model_prediction_name_ = "output1";  //"output2";
  params.weights_not_to_train_ = {
      "position_01",            // Slice's dat input
      "op_min_ends_expand_10",  //op_min_ends_expand_10
  };

  params.immutable_weigths_ = {
      {"Div", {{1, 8.0f}, {1, 1.4142135381698608f}}},
      {"Add", {{1, 1.0f}, {1, 9.999999960041972e-13f}}},
      {"Mul", {{1, 0.5f}, {1, -10000.0f}}},
      {"Sub", {{0, 1.0f}}}};

  params.in_graph_optimizer_name_ = "AdamOptimizer";
  params.adam_opt_params_.alpha_ = 0.9f;
  params.adam_opt_params_.beta_ = 0.999f;
  params.adam_opt_params_.lambda_ = 0;
  params.adam_opt_params_.epsilon_ = 1e-6f;

  params.shuffle_data_ = false;

  // name_in_data_file -> name_in_model
  params.input_name_map_ = {
      {"input_ids", "input1"},
      {"segment_ids", "input2"},
      {"input_mask", "input3"},
      {"masked_lm_positions", "masked_lm_positions"},
      {"masked_lm_ids", "masked_lm_ids"},
      {"masked_lm_weights", "masked_lm_weights"},
      {"next_sentence_label", "next_sentence_labels"}};

  params.use_cuda_ = true;

  params.skip_evaluation_ = config.is_perf_test;

  params.error_function_ = [](const MLValue& /*predict*/, const MLValue& /*label*/, const MLValue& loss) {
    // const Tensor& predict_t = predict.Get<Tensor>();
    // const Tensor& label_t = label.Get<Tensor>();
    const Tensor& loss_t = loss.Get<Tensor>();

    // const float* prediction_data = predict_t.template Data<float>();
    // const int64_t* label_data = label_t.template Data<int64_t>();
    const float* loss_data = loss_t.template Data<float>();

    //const TensorShape predict_shape = predict_t.Shape();
    //const TensorShape label_shape = label_t.Shape();
    //const TensorShape loss_shape = loss_t.Shape();
    //ORT_ENFORCE(predict_shape.NumDimensions() == label_shape.NumDimensions() + 1);

    //int64_t batch_size = predict_shape[0];
    //for (int n = 0; n < batch_size; ++n) {
    //  auto max_class_index = std::distance(prediction_data,
    //                                       std::max_element(prediction_data, prediction_data + NUM_CLASS));

    //  if (static_cast<int>(label_data[max_class_index]) == 1) {
    //    true_count++;
    //  }

    //  prediction_data += predict_shape.SizeFromDimension(1);
    //  label_data += label_shape.SizeFromDimension(1);
    //}
    total_loss += *loss_data;
  };

  params.post_evaluation_callback_ = [](size_t num_samples) {
    float precision = float(true_count) / num_samples;
    float average_loss = total_loss / float(num_samples);
    printf("#examples: %d, #correct: %d, precision: %0.04f, loss: %0.04f \n\n",
           static_cast<int>(num_samples),
           true_count,
           precision,
           average_loss);
    true_count = 0;
    total_loss = 0.0f;
  };
}

int main(int argc, char* args[]) {
#ifndef USE_CUDA
  printf("BERT training is not supported in non-CUDA build. ");
#endif

  TrainingConfig config;
  if (validate_params(argc, args, config) == -1) return -1;

  // setup logger
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  // setup onnxruntime env
  unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(env).IsOK());

  // setup training params
  TrainingRunner::Parameters params;
  setup_training_params(config, params);

  int device_id = 0, device_count = 1;

// setup horovod
#ifdef USE_HOROVOD
  std::tie(device_id, device_count) = setup_horovod();
#endif

  // TODO: This should be done in SGD optimizer. Will refactor when optimizing the kernel.
  // Adding another cuda kernel call for this division seems wasteful currently.
  // params.learning_rate_ = LEARNING_RATE / params.batch_size_;
  params.learning_rate_ = LEARNING_RATE / device_count;
  params.world_rank_ = device_id;
  params.world_size_ = device_count;
  if (params.use_cuda_) {
    printf("Using cuda device #%d \n", params.world_rank_);
  }

  const size_t max_num_files_preload = 2;
  DataLoader training_data_loader(params.input_name_map_,
                                  TRAINING_DATA_PATH,
                                  max_num_files_preload,
                                  device_id,
                                  device_count);
  DataLoader test_data_loader(params.input_name_map_,
                              TEST_DATA_PATH,
                              max_num_files_preload);
  RETURN_IF_FAIL(training_data_loader.Load());
  // Evaluation is only done in device #0
  if (device_id == 0) {
    RETURN_IF_FAIL(test_data_loader.Load());
  }

  // setup fake data
  int batch_size = static_cast<int>(params.batch_size_);
  int max_seq_len_in_batch = 512;
  std::vector<std::string> tensor_names = {"input1",
                                           "input2",
                                           "input3",
                                           "masked_lm_positions",
                                           "masked_lm_ids",
                                           "masked_lm_weights",
                                           "next_sentence_labels"};
  std::vector<TensorShape> tensor_shapes = {{batch_size, max_seq_len_in_batch},
                                            {batch_size, max_seq_len_in_batch},
                                            {batch_size, max_seq_len_in_batch},
                                            {batch_size, 80},
                                            {batch_size, 80},
                                            {batch_size, 80},
                                            {batch_size}};
  std::vector<onnx::TensorProto_DataType> tensor_types = {onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_FLOAT,
                                                          onnx::TensorProto_DataType_INT64};
  RandomDataSet trainingData(config.num_of_training_samples, tensor_names, tensor_shapes, tensor_types);
  RandomDataSet testData(config.num_of_testing_samples, tensor_names, tensor_shapes, tensor_types);

  // start training session
  std::unique_ptr<TrainingRunner> runner;
  if (config.is_perf_test) {
    runner = std::make_unique<TrainingRunner>(&trainingData, &testData, params);
  } else {
    runner = std::make_unique<TrainingRunner>(&training_data_loader, &test_data_loader, params);
  }
  RETURN_IF_FAIL(runner->Initialize());
  RETURN_IF_FAIL(runner->Run());

#ifdef USE_HOROVOD
  shutdown_horovod();
#endif
}
