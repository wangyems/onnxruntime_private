// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <math.h>
#include "constant.h"
#include "core/framework/callback.h"
#include "core/framework/ml_value.h"
#include "core/framework/framework_common.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#define RETURN_IF_FAIL(expr)                                \
  do {                                                      \
    auto status = (expr);                                   \
    if ((!status.IsOK())) {                                 \
      printf("Fail: %s \n", status.ErrorMessage().c_str()); \
      return -1;                                            \
    }                                                       \
  } while (0);

namespace onnxruntime {
namespace training {

// A class to hold a dataset.
// which contains:
// 1. tensor names, we make a simple assumption that the last one is the label !!!
// 2. data samples: each sample contains ort_values for all the tensor names above.
class DataSet {
 public:
  typedef std::unique_ptr<std::vector<OrtValue>> SampleType;

  typedef typename std::vector<SampleType>::const_iterator IteratorType;

  DataSet(const std::vector<std::string>& tensor_names);

  virtual ~DataSet();

  // Get all tensor names
  const std::vector<std::string> TensorNames() const;

  size_t NumInputs() const { return tensor_names_.size(); }

  std::string GetInputName(size_t input_index) const { return tensor_names_[input_index]; }

  common::Status AddData(SampleType&& single_sample);

  common::Status AddData(const std::vector<ONNX_NAMESPACE::TensorProto>& features);

  virtual size_t NumSamples() const { return data_.size(); }

  // Given a batch_size, get the total num of batches.
  size_t TotalBatch(size_t batch_size) const;

  virtual std::vector<OrtValue> GetKthBatch(size_t batch_size, size_t k_th, AllocatorPtr allocator = nullptr) const;

  void RandomShuffle();

  /**
     The method is for getting model training params that are part of training data
     first load .onnx model in Netron to get the mapping between input data and the graph
     for example, a bert model (see input_name_map in bert/main.cc) requires 7 inputs
     each input may have different tensor shape, like so
        intput1 : int64[batch,sequence]
        masked_lm_ids:  int64[batch,dynamic_prediction_count]
     So to get sequence length, we can look for input name "input1" and get its value in shape vector's position 1 element 
     based on metrics_map (see metrics_map example in bert/main.cc) to map the name with the vector position, like so
        {"input1", {"SegLen", 1}}  => sequence->SeqLen , where SeqLen will be populated as key in perf_properties
     @param batch_size batch size of a tensor. Normally, batch_size(i.e., batch) will be at postition 0 of tensor inputs 
                       e.g., input1 above. Although batch size typically is not used, we are not skipping it in the process 
                       to make it looked consistent with the graph loaded by, say, Netron. Thus we are expecting training 
                       params (other than batch size) would start from position 1. For example, "sequence" in input1. 
     @param metrics_map tensor shape dimension mapping from training data, example above {"input1", {"SegLen", 1}} to map 
                        input1's "sequence" at position 1 into "SeqLen" as perf_properties key
     @param perf_properties populated as json for perf monitoring
   */ 
  common::Status GetTensorDimensionsFromInputs(size_t batch_size, const std::map<std::string, std::pair<std::string, size_t>>& metrics_map,
                                               MapStringToString& perf_properties) const;

 private:
  // The names of the tensors.
  std::vector<std::string> tensor_names_;

  // The data of multiple training samples.
  // data_[i] points to a vector of ORTValues, whose order matches the above names_.
  std::vector<SampleType> data_;

  std::vector<std::unique_ptr<char[]>> ortvalue_buffers_;

  std::vector<OrtCallback> ortvalue_deleters_;
};

class RandomDataSet : public DataSet {
 public:
  explicit RandomDataSet(size_t num_samples,
                         const std::vector<std::string>& tensor_names,
                         const std::vector<TensorShape> tensor_shapes,
                         const std::vector<onnx::TensorProto_DataType> tensor_types)
      : DataSet(tensor_names),
        num_samples_(num_samples),
        tensor_shapes_(tensor_shapes),
        tensor_types_(tensor_types){};

  virtual ~RandomDataSet() {}

  virtual size_t NumSamples() const override { return num_samples_; }

  virtual std::vector<OrtValue> GetKthBatch(size_t batch_size, size_t k_th, AllocatorPtr allocator = nullptr) const override;

 private:
  size_t num_samples_;
  const std::vector<TensorShape> tensor_shapes_;
  const std::vector<onnx::TensorProto_DataType> tensor_types_;
};

class TrainingUtil {
 public:
  template <typename T>
  static void CreateCpuMLValue(const std::vector<int64_t>& dims,
                               const std::vector<T>& value,
                               MLValue* p_mlvalue,
                               AllocatorPtr alloc = nullptr) {
    TensorShape shape(dims);
    assert(shape.Size() == static_cast<int64_t>(value.size()));
    auto element_type = DataTypeImpl::GetType<T>();
    auto allocator = alloc ? alloc : GetCpuAllocator();
    auto p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, allocator);

    if (value.size() > 0) {
      memcpy(p_tensor->MutableDataRaw(), value.data(), p_tensor->SizeInBytes());
    }

    p_mlvalue->Init(p_tensor.release(),
                    DataTypeImpl::GetType<Tensor>(),
                    DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  }

  template <typename T>
  static void CreateCpuMLScalar(const T value,
                                MLValue* p_mlvalue,
                                AllocatorPtr alloc = nullptr) {
    // Scalar has empty shape.
    TensorShape shape;
    auto element_type = DataTypeImpl::GetType<T>();
    auto allocator = alloc ? alloc : GetCpuAllocator();
    auto p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, allocator);

    // Copy one element, the scalar, to a tensor object.
    memcpy(p_tensor->MutableDataRaw(), &value, p_tensor->SizeInBytes());

    // Convert tensor to MLValue.
    p_mlvalue->Init(p_tensor.release(),
                    DataTypeImpl::GetType<Tensor>(),
                    DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  }

  static AllocatorPtr GetCpuAllocator() {
    static CPUExecutionProviderInfo info;
    static CPUExecutionProvider cpu_provider(info);
    return cpu_provider.GetAllocator(0, OrtMemTypeDefault);
  }

  static void PrintNameMLValMap(const NameMLValMap& mlvalue_map);

  static void PrintTensor(const std::string& name, const Tensor& tensor, std::ostream& os = std::cout);
};

struct LearningRateParameters {
  float initial_lr;
  float warmup_ratio;

  // Learning rate schedule to perform warmup : None, Cosine, Constant, Linear, Poly.
  std::string warmup_mode = LRSchedule_NoWarmup;
  std::string feed_name = "Learning_Rate";
};

class LossScaler {
 public:
  LossScaler(const std::string loss_scale_input_name,
             bool is_dynamic_scale,
             float loss_scale = static_cast<float>(1 << 16),
             size_t up_scale_window = 2000,
             float min_loss_scale = 1.0f,
             float max_loss_scale = static_cast<float>(1 << 24))
      : loss_scale_input_name_(loss_scale_input_name),
        is_dynamic_scale_(is_dynamic_scale),
        initial_loss_scale_(loss_scale),
        up_scale_window_(up_scale_window),
        min_loss_scale_(min_loss_scale),
        max_loss_scale_(max_loss_scale),
        loss_scale_(loss_scale),
        stable_steps_(0){};

  std::string GetLossScaleInputName() const { return loss_scale_input_name_; }

  float GetLossScale() const { return loss_scale_; }
  void UpdateLossScale(bool is_all_finite) {
    if (!is_dynamic_scale_) {
      return;
    }

    if (is_all_finite) {
      stable_steps_++;

      if (stable_steps_ >= up_scale_window_) {
        loss_scale_ = std::fmin(max_loss_scale_, loss_scale_ * 2);
        stable_steps_ = 0;
      }
    } else {
      loss_scale_ = std::fmax(min_loss_scale_, loss_scale_ / 2);
      stable_steps_ = 0;
    }
  }

  void Reset() {
    loss_scale_ = initial_loss_scale_;
    stable_steps_ = 0;
  }

  // for checkpointing
  std::string SaveToString() const;
  Status LoadFromString(const std::string& input);

 private:
  const std::string loss_scale_input_name_;
  const bool is_dynamic_scale_;
  const float initial_loss_scale_;
  const size_t up_scale_window_;
  const float min_loss_scale_;
  const float max_loss_scale_;
  float loss_scale_;
  size_t stable_steps_;
};

class LearningRateScheduler {
 public:
  LearningRateScheduler(LearningRateParameters& lr_params, size_t training_step_count)
      : lr_params_(lr_params),
        total_step_count_(static_cast<float>(training_step_count)) {}

  float GetLearningRate(const size_t current_step) const {
    const float cur_ratio = static_cast<float>(current_step) / total_step_count_;
    float schedule_factor = this->GetLearningRateFactor(cur_ratio, lr_params_.warmup_ratio);
    return lr_params_.initial_lr * schedule_factor;
  }

  virtual ~LearningRateScheduler() = default;

  virtual float GetLearningRateFactor(float cur_ratio, float warmp_ratio) const = 0;

  static std::unique_ptr<LearningRateScheduler> Create(LearningRateParameters& lr_params, size_t training_step_count);

 private:
  const LearningRateParameters lr_params_;
  const float total_step_count_;
};

class NoWarmpScheduler : public LearningRateScheduler {
 public:
  NoWarmpScheduler(LearningRateParameters& lr_params, size_t training_step_count)
      : LearningRateScheduler(lr_params, training_step_count) {}

  float GetLearningRateFactor(float /*cur_ratio*/, float /*warmp_ratio*/) const {
    return 1.f;
  }
};

class CosineScheduler : public LearningRateScheduler {
 public:
  CosineScheduler(LearningRateParameters& lr_params, size_t training_step_count)
      : LearningRateScheduler(lr_params, training_step_count) {}

  float GetLearningRateFactor(float cur_ratio, float warmp_ratio) const {
    if (cur_ratio < warmp_ratio) {
      return cur_ratio / warmp_ratio;
    }

    return 0.5f * (1.f + std::cos(static_cast<float>(M_PI) * cur_ratio));
  }
};

class ConstantScheduler : public LearningRateScheduler {
 public:
  ConstantScheduler(LearningRateParameters& lr_params, size_t training_step_count)
      : LearningRateScheduler(lr_params, training_step_count) {}

  float GetLearningRateFactor(float cur_ratio, float warmp_ratio) const {
    if (cur_ratio < warmp_ratio) {
      return cur_ratio / warmp_ratio;
    }

    return 1.f;
  }
};

class LinearScheduler : public LearningRateScheduler {
 public:
  LinearScheduler(LearningRateParameters& lr_params, size_t training_step_count)
      : LearningRateScheduler(lr_params, training_step_count) {}

  float GetLearningRateFactor(float cur_ratio, float warmp_ratio) const {
    if (cur_ratio < warmp_ratio) {
      return cur_ratio / warmp_ratio;
    }

    return std::max((cur_ratio - 1.f) / (warmp_ratio - 1.f), 0.f);
  }
};

class PolyScheduler : public LearningRateScheduler {
 public:
  PolyScheduler(LearningRateParameters& lr_params, size_t training_step_count)
      : LearningRateScheduler(lr_params, training_step_count) {}

  float GetLearningRateFactor(float cur_ratio, float warmp_ratio) const {
    if (cur_ratio < warmp_ratio) {
      return cur_ratio / warmp_ratio;
    }

    const float degree = 0.5f;
    return std::pow(1.f - cur_ratio, degree);
  }
};

}  // namespace training
}  // namespace onnxruntime
