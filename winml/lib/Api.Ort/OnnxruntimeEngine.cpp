#include "pch.h"

#include "OnnxruntimeEngine.h"

#include "PheonixSingleton.h"
#include "OnnxruntimeEnvironment.h"
#include "OnnxruntimeEngineBuilder.h"
#include "OnnxruntimeModel.h"
#include "OnnxruntimeSessionBuilder.h"
#include "OnnxruntimeErrors.h"

#include "core/providers/winml/winml_provider_factory.h"

using namespace WinML;

static const OrtApi* GetVersionedOrtApi() {
  static const uint32_t ort_version = 1;
  const auto ort_api_base = OrtGetApiBase();
  return ort_api_base->GetApi(ort_version);
}

static const WinmlAdapterApi* GetVersionedWinmlAdapterApi() {
  return OrtGetWinMLAdapter(GetVersionedOrtApi());
}

static ONNXTensorElementDataType
ONNXTensorElementDataTypeFromTensorKind(winml::TensorKind kind) {
  switch (kind) {
    case winml::TensorKind::Boolean: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    case winml::TensorKind::String: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }
    case winml::TensorKind::Float16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    case winml::TensorKind::Float: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    case winml::TensorKind::Double: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
    case winml::TensorKind::Int8: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    }
    case winml::TensorKind::Int16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    }
    case winml::TensorKind::Int32: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
    case winml::TensorKind::Int64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    case winml::TensorKind::UInt8: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
    case winml::TensorKind::UInt16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    }
    case winml::TensorKind::UInt32: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    }
    case winml::TensorKind::UInt64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    }
    case winml::TensorKind::Complex64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    }
    case winml::TensorKind::Complex128: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
    }
    default: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
}

OnnxruntimeValue::OnnxruntimeValue() : value_(nullptr, nullptr), allocator_(nullptr, nullptr) {}

OnnxruntimeValue::~OnnxruntimeValue() {
  value_.reset(nullptr);
  allocator_.reset(nullptr);
}

HRESULT OnnxruntimeValue::RuntimeClassInitialize(OnnxruntimeEngine* engine, UniqueOrtValue&& ort_value, UniqueOrtAllocator&& allocator) {
  engine_ = engine;
  value_ = std::move(ort_value);
  allocator_ = std::move(allocator);

  return S_OK;
}

HRESULT OnnxruntimeValue::IsEmpty(bool* out) {
  *out = UseOrtValue() == nullptr;
  return S_OK;
}

HRESULT OnnxruntimeValue::IsCpu(bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();
  auto winml_adapter_api = engine_->GetEngineFactory()->UseWinmlAdapterApi();

  OrtMemoryInfo* ort_memory_info;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->GetValueMemoryInfo(value_.get(), &ort_memory_info),
                                   ort_api);
  auto memory_info = UniqueOrtMemoryInfo(ort_memory_info, ort_api->ReleaseMemoryInfo);

  const char* name;
  ort_api->MemoryInfoGetName(memory_info.get(), &name);

  OrtMemType type;
  ort_api->MemoryInfoGetMemType(memory_info.get(), &type);

  *out = !strcmp(name, "Cpu") ||
         type == OrtMemType::OrtMemTypeCPUOutput ||
         type == OrtMemType::OrtMemTypeCPUInput;
  return S_OK;
}

static auto GetStrings(const OrtApi* ort_api, const OrtValue* ort_value,
                       OrtTensorTypeAndShapeInfo* type_and_shape_info) {
  std::vector<std::string> out;

  size_t size;
  ort_api->GetDimensionsCount(type_and_shape_info, &size);

  std::vector<int64_t> shape(size);
  ort_api->GetDimensions(type_and_shape_info, &shape[0], size);

  // there needs to be only one dimension
  if (shape.size() != 1) {
    throw;
  }

  auto length = shape[0];

  // make a big buffer to hold all the string data
  size_t buffer_length;
  ort_api->GetStringTensorDataLength(ort_value, &buffer_length);

  std::vector<std::pair<const char*, size_t>> strings;
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[buffer_length]);
  std::vector<size_t> offsets(length);

  ort_api->GetStringTensorContent(ort_value, buffer.get(), buffer_length, offsets.data(), offsets.size());

  // now go build all the strings
  for (auto i = 0; i < length; ++i) {
    size_t str_len = 0;
    // are we on the last one?
    if (i == (length - 1)) {
      str_len = buffer_length - offsets[i];
    } else {
      str_len = offsets[i + 1] - offsets[i];
    }
    strings.push_back(std::make_pair(reinterpret_cast<const char*>(buffer.get() + offsets[i]), str_len));
  }

  return std::make_shared<std::pair<decltype(strings), decltype(buffer)>>(std::move(strings), std::move(buffer));
}

WinML::unique_void OnnxruntimeValue::GetResource() {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();
  auto winml_adapter_api = engine_->GetEngineFactory()->UseWinmlAdapterApi();

  void* mutable_data = nullptr;
  ort_api->GetTensorMutableData(value_.get(), &mutable_data);

  OrtExecutionProvider* ort_provider;
  THROW_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvider(engine_->UseOrtSession(), 0, &ort_provider),
                               engine_factory_->UseOrtApi());

  bool is_cpu = false;
  if (SUCCEEDED(IsCpu(&is_cpu)) && !is_cpu) {
    void* resource;
    THROW_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->DmlGetD3D12ResourceFromAllocation(ort_provider, mutable_data,
                                                         reinterpret_cast<ID3D12Resource**>(&resource),
                                 engine_factory_->UseOrtApi());
    return WinML::unique_void(resource, [](void*) { /*do nothing, as this pointer is actually a com pointer! */ });
  } else {
    int is_tensor;
    ort_api->IsTensor(value_.get(), &is_tensor);
    if (is_tensor == 0) {
      return WinML::unique_void(mutable_data, [](void*) { /*do nothing, as this pointer is actually owned elsewhere in ORT! */ });
    }

    OrtTensorTypeAndShapeInfo* info = nullptr;
    ort_api->GetTensorTypeAndShape(value_.get(), &info);
    auto type_and_shape_info = UniqueOrtTensorTypeAndShapeInfo(info, ort_api->ReleaseTensorTypeAndShapeInfo);

    ONNXTensorElementDataType data_type;
    ort_api->GetTensorElementType(type_and_shape_info.get(), &data_type);

    if (data_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      auto strings = GetStrings(ort_api, value_.get(), info);
      auto string_data = strings->first.data();
      return WinML::unique_void(string_data, [capture_strings = strings](void*) { /*This deleter does nothing but capture the strings, which extends the lifetime of the returned strings.*/ });
    } else {
      return WinML::unique_void(mutable_data, [](void*) { /*do nothing, as this pointer is actually owned elsewhere in ORT! */ });
    }
  }
}

HRESULT OnnxruntimeValue::IsTensor(bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();

  ONNXType type = ONNXType::ONNX_TYPE_UNKNOWN;
  ort_api->GetValueType(value_.get(), &type);
  *out = type == ONNXType::ONNX_TYPE_TENSOR;
  return S_OK;
}

HRESULT OnnxruntimeValue::IsOfTensorType(winml::TensorKind kind, bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();
  OrtTensorTypeAndShapeInfo* info = nullptr;
  ort_api->GetTensorTypeAndShape(value_.get(), &info);
  auto type_and_shape_info = UniqueOrtTensorTypeAndShapeInfo(info, ort_api->ReleaseTensorTypeAndShapeInfo);

  ONNXTensorElementDataType data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ort_api->GetTensorElementType(type_and_shape_info.get(), &data_type);

  *out = data_type == ONNXTensorElementDataTypeFromTensorKind(kind);
  return S_OK;
}

HRESULT OnnxruntimeValue::GetTensorShape(std::vector<int64_t>& shape_vector) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();
  OrtTensorTypeAndShapeInfo* info = nullptr;
  ort_api->GetTensorTypeAndShape(value_.get(), &info);
  auto type_and_shape_info = UniqueOrtTensorTypeAndShapeInfo(info, ort_api->ReleaseTensorTypeAndShapeInfo);

  size_t size;
  ort_api->GetDimensionsCount(type_and_shape_info.get(), &size);

  std::vector<int64_t> shape(size);
  ort_api->GetDimensions(type_and_shape_info.get(), &shape[0], size);

  shape_vector = std::move(shape);
  return S_OK;
}

HRESULT OnnxruntimeValue::IsOfMapType(winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) {
  /*
    
  bool LearningModelBinding::IsOfMapType(const Ort::Value& ort_value, TensorKind key_kind, TensorKind value_kind) {
    if (ort_value.GetTypeInfo().GetONNXType() != ONNX_TYPE_MAP)
      return false;

    ONNXTensorElementDataType onnx_key_type;
    ONNXTensorElementDataType onnx_value_type;

    WINML_THROW_IF_FAILED(adapter_->GetMapType(ort_value, &onnx_key_type, &onnx_value_type));

    if (onnx_key_type != GetONNXTensorElementDataType(key_kind))
      return false;

    if (onnx_value_type != GetONNXTensorElementDataType(value_kind))
      return false;

    return true;
  };
    */

  return E_NOTIMPL;
}

HRESULT OnnxruntimeValue::IsOfVectorMapType(winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) {
  /*
  bool LearningModelBinding::IsOfVectorMapType(const Ort::Value& ort_value, TensorKind key_kind, TensorKind value_kind) {
    if (ort_value.GetTypeInfo().GetONNXType() != ONNX_TYPE_SEQUENCE)
      return false;

    ONNXTensorElementDataType onnx_key_type;
    ONNXTensorElementDataType onnx_value_type;

    WINML_THROW_IF_FAILED(adapter_->GetVectorMapType(ort_value, &onnx_key_type, &onnx_value_type));

    if (onnx_key_type != GetONNXTensorElementDataType(key_kind))
      return false;

    if (onnx_value_type != GetONNXTensorElementDataType(value_kind))
      return false;

    return true;
  };
  */
  return E_NOTIMPL;
}

HRESULT OnnxruntimeValue::SetParameter(IUnknown* param) {
  param_ = param;
  return S_OK;
}

OrtValue* OnnxruntimeValue::UseOrtValue() {
  return value_.get();
}

HRESULT OnnxruntimeValue::AssignOrtValue(OrtValue* in) {
  value_.reset(in);
  return S_OK;
}

OnnxruntimeEngine::OnnxruntimeEngine() : session_(nullptr, nullptr) {
}

HRESULT OnnxruntimeEngine::RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory,
                                                  UniqueOrtSession&& session,
                                                  IOrtSessionBuilder* session_builder) {
  engine_factory_ = engine_factory;
  session_ = std::move(session);
  session_builder_ = session_builder;
  return S_OK;
}

HRESULT OnnxruntimeEngine::LoadModel(_In_ IModel* model) {
  Microsoft::WRL::ComPtr<IOnnxruntimeModel> onnxruntime_model;
  RETURN_IF_FAILED(model->QueryInterface(IID_PPV_ARGS(&onnxruntime_model)));

  OrtModel* ort_model;
  RETURN_IF_FAILED(onnxruntime_model->DetachOrtModel(&ort_model));

  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionLoadAndPurloinModel(session_.get(), ort_model),
                                   engine_factory_->UseOrtApi());
  return S_OK;
}

HRESULT OnnxruntimeEngine::Initialize() {
  RETURN_IF_FAILED(session_builder_->Initialize(session_.get()));
  return S_OK;
}

HRESULT OnnxruntimeEngine::RegisterGraphTransformers() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionRegisterGraphTransformers(session_.get()),
                                   engine_factory_->UseOrtApi());
  return S_OK;
}

HRESULT OnnxruntimeEngine::RegisterCustomRegistry(IMLOperatorRegistry* registry) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionRegisterCustomRegistry(session_.get(), registry),
                                   engine_factory_->UseOrtApi());
  return S_OK;
}

HRESULT OnnxruntimeEngine::EndProfiling() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionEndProfiling(session_.get()),
                                   engine_factory_->UseOrtApi());
  return S_OK;
}

HRESULT OnnxruntimeEngine::StartProfiling() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtEnv* ort_env;
  engine_factory_->GetOrtEnvironment(&ort_env);

  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionStartProfiling(ort_env, session_.get()),
                                   engine_factory_->UseOrtApi());
  return S_OK;
}

HRESULT OnnxruntimeEngine::FlushContext() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider),
                                   engine_factory_->UseOrtApi());

  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->DmlExecutionProviderFlushContext(ort_provider),
                                   engine_factory_->UseOrtApi());
  return S_OK;
}

HRESULT OnnxruntimeEngine::TrimUploadHeap() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider),
                                   engine_factory_->UseOrtApi());

  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->DmlExecutionProviderTrimUploadHeap(ort_provider),
                                   engine_factory_->UseOrtApi());

  return S_OK;
}

HRESULT OnnxruntimeEngine::ReleaseCompletedReferences() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider),
                                   engine_factory_->UseOrtApi());

  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->DmlExecutionProviderReleaseCompletedReferences(ort_provider),
                                   engine_factory_->UseOrtApi());

  return S_OK;
}

HRESULT OnnxruntimeEngine::CopyValueAcrossDevices(IValue* src, IValue* dest) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider);

  auto src_value = static_cast<OnnxruntimeValue*>(src);
  auto dest_value = static_cast<OnnxruntimeValue*>(dest);

  bool is_empty;
  auto has_null_source = (SUCCEEDED(src_value->IsEmpty(&is_empty)) && is_empty);
  RETURN_HR_IF(E_FAIL, has_null_source);

  auto has_null_dest = (SUCCEEDED(dest_value->IsEmpty(&is_empty)) && is_empty);
  RETURN_HR_IF(E_FAIL, has_null_dest);

  winml_adapter_api->DmlCopyTensor(ort_provider, src_value->UseOrtValue(), dest_value->UseOrtValue());
  return S_OK;
}

HRESULT OnnxruntimeEngine::Sync() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider),
                                   engine_factory_->UseOrtApi());

  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->ExecutionProviderSync(ort_provider),
                                   engine_factory_->UseOrtApi());

  return S_OK;
}

OrtSession* OnnxruntimeEngine::UseOrtSession() {
  return session_.get();
}

const OrtApi* OnnxruntimeEngine::UseOrtApi() {
  return engine_factory_->UseOrtApi();
}

OnnxruntimeEngineFactory* OnnxruntimeEngine::GetEngineFactory() {
  return engine_factory_.Get();
}

HRESULT OnnxruntimeEngine::CreateTensorValue(const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider),
                                   engine_factory_->UseOrtApi());

  OrtAllocator* ort_allocator;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->GetProviderAllocator(ort_provider, &ort_allocator),
                                   engine_factory_->UseOrtApi());

  auto unique_allocator = UniqueOrtAllocator(ort_allocator, winml_adapter_api->FreeProviderAllocator);  // the release here should probably not return anything

  OrtValue* ort_value;
  ort_api->CreateTensorAsOrtValue(unique_allocator.get(), shape, count, ONNXTensorElementDataTypeFromTensorKind(kind), &ort_value);
  auto unique_value = UniqueOrtValue(ort_value, ort_api->ReleaseValue);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(out, this, std::move(unique_value), std::move(unique_allocator)));
  return S_OK;
}

using DmlAllocatorResource = std::unique_ptr<void, void (*)(void*)>;
class DmlAllocatorWrapper : public Microsoft::WRL::RuntimeClass<
                                Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                                IUnknown> {
 public:
  DmlAllocatorWrapper() : dml_resource_(nullptr, nullptr) {}

  HRESULT RuntimeClassInitialize(DmlAllocatorResource&& dml_resource) {
    dml_resource_ = std::move(dml_resource);
    return S_OK;
  }

 private:
  DmlAllocatorResource dml_resource_;
};

HRESULT OnnxruntimeEngine::CreateTensorValueFromExternalD3DResource(ID3D12Resource* d3d_resource, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider),
                                   engine_factory_->UseOrtApi());

  OrtMemoryInfo* dml_memory = nullptr;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->GetProviderMemoryInfo(ort_provider, &dml_memory),
                                   engine_factory_->UseOrtApi());

  void* dml_allocator_resource;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->DmlCreateGPUAllocationFromD3DResource(d3d_resource, &dml_allocator_resource),
                                   engine_factory_->UseOrtApi());

  auto unique_dml_allocator_resource =
      DmlAllocatorResource(dml_allocator_resource,
                           [](void* ptr) {
                             GetVersionedWinmlAdapterApi()->DmlFreeGPUAllocation(ptr);
                           });

  // create the OrtValue as a tensor letting ort know that we own the data buffer
  OrtValue* ort_value;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(ort_api->CreateTensorWithDataAsOrtValue(
                                       dml_memory,
                                       unique_dml_allocator_resource.get(),
                                       d3d_resource->GetDesc().Width,
                                       shape,
                                       count,
                                       ONNXTensorElementDataTypeFromTensorKind(kind),
                                       &ort_value),
                                   engine_factory_->UseOrtApi());
  auto unique_value = UniqueOrtValue(ort_value, ort_api->ReleaseValue);

  Microsoft::WRL::ComPtr<OnnxruntimeValue> out_value;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(&out_value, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)));

  // Cache the allocator on the value so it destructs appropriately when the value is dropped
  Microsoft::WRL::ComPtr<DmlAllocatorWrapper> dml_allocator_resource_wrapper;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<DmlAllocatorWrapper>(&dml_allocator_resource_wrapper, std::move(unique_dml_allocator_resource)));

  RETURN_IF_FAILED(out_value->SetParameter(dml_allocator_resource_wrapper.Get()));

  *out = out_value.Detach();

  return S_OK;
}

static int64_t ShapeSize(const int64_t* shape, size_t count) {
  // for each dim
  int64_t size = 1;
  for (int i = 0; i < count; i++) {
    // find out it's total size
    size *= shape[i];
    // make sure there are no invalid dimensions (-1 or any invalid shape)
    THROW_HR_IF(E_INVALIDARG, shape[i] <= 0);
  }
  return size;
}

HRESULT OnnxruntimeEngine::CreateStringTensorValueFromDataWithCopy(const char* const* data, size_t num_elements, const int64_t* shape, size_t count, _Out_ IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();
  RETURN_IF_FAILED(CreateTensorValue(shape, count, winml::TensorKind::String, out));

  auto ort_value = reinterpret_cast<WinML::OnnxruntimeValue*>(*out)->UseOrtValue();
  ort_api->FillStringTensor(ort_value, reinterpret_cast<const char* const*>(data), num_elements);
  return S_OK;
}

HRESULT OnnxruntimeEngine::CreateTensorValueFromExternalBuffer(void* data, size_t size_in_bytes, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();

  if (kind == winml::TensorKind::String) {
    // String buffers cannot be passed into the ort api directly because ort c-api tensor strings cannot be backed by external memory
    return E_NOTIMPL;
  }

  // TODO:  what is the difference between the device allocator and the arena allocator?
  OrtMemoryInfo* cpu_memory;
  ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory);

  OrtValue* ort_value;
  ort_api->CreateTensorWithDataAsOrtValue(
      cpu_memory,
      data,
      size_in_bytes,
      shape,
      count,
      ONNXTensorElementDataTypeFromTensorKind(kind),
      &ort_value);
  auto unique_value = UniqueOrtValue(ort_value, ort_api->ReleaseValue);

  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(out, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)));
  return S_OK;
}

HRESULT OnnxruntimeEngine::CreateNullValue(_Out_ IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto unique_value = UniqueOrtValue(nullptr, ort_api->ReleaseValue);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(out, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)));
  return S_OK;
}
#include "winrt/windows.foundation.collections.h"

template <typename TAbiType>
struct AbiTypeInfo {
  using CppWinRTType = TAbiType;
  using OrtType = TAbiType;
};

template <>
struct AbiTypeInfo<HSTRING> {
  using CppWinRTType = winrt::hstring;
  using OrtType = const char*;
};

template <typename TRawType>
typename auto CppwinrtTypeToOrtType(TRawType raw) {
  return raw;
}

template <>
typename auto CppwinrtTypeToOrtType<winrt::hstring>(winrt::hstring raw) {
  return WinML::Strings::UTF8FromHString(raw);
}

template <typename TAbiKey, typename TAbiValue>
auto CastToWinrtMap(IInspectable* map_insp) {
  using cppwinrt_key_type = typename AbiTypeInfo<TAbiKey>::CppWinRTType;
  using cppwinrt_value_type = typename AbiTypeInfo<TAbiValue>::CppWinRTType;

  ::winrt::Windows::Foundation::Collections::IMap<cppwinrt_key_type, cppwinrt_value_type> map;
  winrt::copy_from_abi(map, map_insp);
  return map;
}

template <typename TAbiKey, typename TAbiValue> struct FillMapTensors {
  static HRESULT Run(const OrtApi* ort_api, IInspectable* map_insp, OrtValue* keys_ort_value, OrtValue* values_ort_value) {
	  AbiTypeInfo<TAbiKey>::OrtType* keys_mutable_data;
	  ort_api->GetTensorMutableData(keys_ort_value, reinterpret_cast<void**>(&keys_mutable_data));

	  AbiTypeInfo<TAbiValue>::OrtType* values_mutable_data;
	  ort_api->GetTensorMutableData(values_ort_value, reinterpret_cast<void**>(&values_mutable_data));

	  auto map = CastToWinrtMap<TAbiKey, TAbiValue>(map_insp);
	  size_t index = 0;
	  for (const auto& pair : map) {
		keys_mutable_data[index] = CppwinrtTypeToOrtType(pair.Key());
		values_mutable_data[index] = CppwinrtTypeToOrtType(pair.Value());
		index++;
	  }
	  return S_OK;
	}
};

template <typename TAbiValue> struct FillMapTensors<HSTRING, TAbiValue> {
  static HRESULT Run(const OrtApi* ort_api, IInspectable* map_insp, OrtValue* keys_ort_value, OrtValue* values_ort_value) {
	  AbiTypeInfo<TAbiValue>::OrtType* values_mutable_data;
	  ort_api->GetTensorMutableData(values_ort_value, reinterpret_cast<void**>(&values_mutable_data));

	  auto map = CastToWinrtMap<HSTRING, TAbiValue>(map_insp);
	  size_t index = 0;
	  std::vector<std::string> keys;
	  for (const auto& pair : map) {
		keys.push_back(CppwinrtTypeToOrtType(pair.Key()));
        values_mutable_data[index] = CppwinrtTypeToOrtType(pair.Value());
		index++;
	  }

	  std::vector<const char*> raw_values;
	  std::transform(
		  keys.begin(),
		  keys.end(),
		  std::back_inserter(raw_values),
		  [&](auto& str) { return str.c_str(); });
  
	  ort_api->FillStringTensor(keys_ort_value, raw_values.data(), raw_values.size());

	  return S_OK;
	}
};

template <typename TAbiKey> struct FillMapTensors<TAbiKey, HSTRING> {
  static HRESULT Run(const OrtApi* ort_api, IInspectable* map_insp, OrtValue* keys_ort_value, OrtValue* values_ort_value) {
    AbiTypeInfo<TAbiKey>::OrtType* keys_mutable_data;
    ort_api->GetTensorMutableData(keys_ort_value, reinterpret_cast<void**>(& keys_mutable_data));

    auto map = CastToWinrtMap<TAbiKey, HSTRING>(map_insp);
    size_t index = 0;
    std::vector<std::string> values;
    for (const auto& pair : map) {
      keys_mutable_data[index] = CppwinrtTypeToOrtType(pair.Key());
      values.push_back(CppwinrtTypeToOrtType(pair.Value()));
      index++;
    }

    std::vector<const char*> raw_values;
    std::transform(
        values.begin(),
        values.end(),
        std::back_inserter(raw_values),
        [&](auto& str) { return str.c_str(); });

    ort_api->FillStringTensor(keys_ort_value, raw_values.data(), raw_values.size());
	return S_OK;
  }
};

template <>
struct FillMapTensors<HSTRING, HSTRING> {
  static HRESULT Run(const OrtApi* ort_api, IInspectable* map_insp, OrtValue* keys_ort_value, OrtValue* values_ort_value) {
    auto map = CastToWinrtMap<HSTRING, HSTRING>(map_insp);
    size_t index = 0;
    std::vector<std::string> keys;
    std::vector<std::string> values;
    for (const auto& pair : map) {
      keys.push_back(CppwinrtTypeToOrtType(pair.Key()));
      values.push_back(CppwinrtTypeToOrtType(pair.Value()));
      index++;
    }

    std::vector<const char*> raw_keys;
    std::transform(
        keys.begin(),
        keys.end(),
        std::back_inserter(raw_keys),
        [&](auto& str) { return str.c_str(); });

    std::vector<const char*> raw_values;
    std::transform(
        values.begin(),
        values.end(),
        std::back_inserter(raw_values),
        [&](auto& str) { return str.c_str(); });

    ort_api->FillStringTensor(keys_ort_value, raw_keys.data(), raw_keys.size());
    ort_api->FillStringTensor(values_ort_value, raw_values.data(), raw_values.size());
    return S_OK;
  }
};

template <typename TAbiKey, typename TAbiValue>
HRESULT CreateMapValue(OnnxruntimeEngine* engine, IInspectable* map_insp, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue** out) {
  auto ort_api = engine->UseOrtApi();
  auto map = CastToWinrtMap<TAbiKey, TAbiValue>(map_insp);
  std::vector<int64_t> shape = {static_cast<int64_t>(map.Size())};

  winrt::com_ptr<WinML::IValue> key_value;
  RETURN_IF_FAILED(engine->CreateTensorValue(shape.data(), shape.size(), key_kind, key_value.put()));
  auto keys_ort_value = static_cast<OnnxruntimeValue*>(key_value.get())->UseOrtValue();

  winrt::com_ptr<WinML::IValue> value_value;
  RETURN_IF_FAILED(engine->CreateTensorValue(shape.data(), shape.size(), value_kind, value_value.put()));
  auto values_ort_value = static_cast<OnnxruntimeValue*>(value_value.get())->UseOrtValue();

  auto hr = FillMapTensors<TAbiKey, TAbiValue>::Run(ort_api, map_insp, keys_ort_value, values_ort_value);
  RETURN_IF_FAILED(hr);

  OrtValue* inputs[2] = {keys_ort_value, values_ort_value};

  OrtValue* map_value;
  ort_api->CreateValue(inputs, 2, ONNXType::ONNX_TYPE_MAP, &map_value);
  auto unique_map_ort_value = UniqueOrtValue(map_value, ort_api->ReleaseValue);

  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(out, engine, std::move(unique_map_ort_value), UniqueOrtAllocator(nullptr, nullptr)));
  return S_OK;
}

static auto GetMapValueCreator(OnnxruntimeEngine* engine, winml::TensorKind key_kind, winml::TensorKind value_kind) {
  using namespace std::placeholders;
  if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Int64) {
    return std::bind(&CreateMapValue<int64_t, int64_t>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::Int64, _2);
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Float) {
    return std::bind(&CreateMapValue<int64_t, float>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::Float, _2);
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Double) {
    return std::bind(&CreateMapValue<int64_t, double>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::Double, _2);
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::String) {
    return std::bind(&CreateMapValue<int64_t, HSTRING>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::String, _2);
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Int64) {
    return std::bind(&CreateMapValue<HSTRING, int64_t>, engine, _1, winml::TensorKind::String, winml::TensorKind::Int64, _2);
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Float) {
    return std::bind(&CreateMapValue<HSTRING, float>, engine, _1, winml::TensorKind::String, winml::TensorKind::Float, _2);
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Double) {
    return std::bind(&CreateMapValue<HSTRING, double>, engine, _1, winml::TensorKind::String, winml::TensorKind::Double, _2);
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::String) {
    return std::bind(&CreateMapValue<HSTRING, HSTRING>, engine, _1, winml::TensorKind::String, winml::TensorKind::String, _2);
  }

  THROW_HR(E_NOTIMPL);
}

HRESULT OnnxruntimeEngine::CreateMapValue(IInspectable* map, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue** out) {
  return GetMapValueCreator(this, key_kind, value_kind)(map, out);
}

HRESULT OnnxruntimeEngine::CreateSequenceValue(IInspectable* map, winml::TensorKind element_kind, _Out_ IValue** out) {
  UNREFERENCED_PARAMETER(map);
  UNREFERENCED_PARAMETER(element_kind);
  UNREFERENCED_PARAMETER(out);
  return E_NOTIMPL;
}

HRESULT OnnxruntimeEngine::CreateOneInputAcrossDevices(const char* name, IValue* src, IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  auto src_value = static_cast<OnnxruntimeValue*>(src);

  bool is_set;
  auto is_empty = SUCCEEDED(src_value->IsEmpty(&is_set)) && is_set;
  auto is_tensor = SUCCEEDED(src_value->IsTensor(&is_set)) && is_set;

  if (is_tensor && !is_empty) {
    int16_t source_location;
    int16_t input_required_location;
    winml_adapter_api->ValueGetDeviceId(src_value->UseOrtValue(), &source_location);
    winml_adapter_api->SessionGetInputRequiredDeviceId(session_.get(), name, &input_required_location);

    if (source_location != input_required_location) {
      OrtValue* dest_ort_value = nullptr;
      winml_adapter_api->SessionCopyOneInputAcrossDevices(session_.get(), name, src_value->UseOrtValue(), &dest_ort_value);
      auto unique_dest_ort_value = UniqueOrtValue(dest_ort_value, ort_api->ReleaseValue);

      RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(out, this, std::move(unique_dest_ort_value), UniqueOrtAllocator(nullptr, nullptr)));
      return S_OK;
    }
  }

  *out = src;
  (*out)->AddRef();
  return S_OK;
}

HRESULT OnnxruntimeEngine::Run(const char** input_names, IValue** inputs, size_t num_inputs, const char** output_names, IValue** outputs, size_t num_outputs) {
  auto ort_api = engine_factory_->UseOrtApi();

  OrtRunOptions* run_options;
  ort_api->CreateRunOptions(&run_options);
  auto unique_run_options = UniqueOrtRunOptions(run_options, ort_api->ReleaseRunOptions);

  std::vector<OrtValue*> input_ort_values;
  std::transform(
      inputs,
      inputs + num_inputs,
      std::back_inserter(input_ort_values),
      [&](auto& input) {
        auto input_value = static_cast<OnnxruntimeValue*>(input);
        return input_value->UseOrtValue();
      });

  std::vector<OrtValue*> output_ort_values;
  std::transform(
      outputs,
      outputs + num_outputs,
      std::back_inserter(output_ort_values),
      [&](auto& output) {
        auto output_value = static_cast<OnnxruntimeValue*>(output);
        return output_value->UseOrtValue();
      });

  ort_api->Run(session_.get(),
               unique_run_options.get(),
               input_names,
               input_ort_values.data(),
               num_inputs,
               output_names,
               num_outputs,
               output_ort_values.data());

  for (size_t index = 0; index < num_outputs; index++) {
    auto output_value = static_cast<OnnxruntimeValue*>(outputs[index]);
    if (output_value->UseOrtValue() != output_ort_values[index]) {
      RETURN_IF_FAILED(output_value->AssignOrtValue(output_ort_values[index]));
    }
  }

  return S_OK;
}

// TODO supposedly this doesnt work if it is not static
static std::shared_ptr<OnnxruntimeEnvironment> onnxruntime_environment_;

HRESULT OnnxruntimeEngineFactory::RuntimeClassInitialize() {
  ort_api_ = GetVersionedOrtApi();
  winml_adapter_api_ = GetVersionedWinmlAdapterApi();

  environment_ = onnxruntime_environment_ = PheonixSingleton<OnnxruntimeEnvironment>(ort_api_);
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateModel(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) {
  OrtModel* ort_model = nullptr;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api_->CreateModelFromPath(model_path, len, &ort_model),
                                   ort_api_);

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateModel(_In_ void* data, _In_ size_t size, _Outptr_ IModel** out) {
  OrtModel* ort_model = nullptr;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api_->CreateModelFromData(data, size, &ort_model),
                                   ort_api_);

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateEngineBuilder(_Outptr_ Windows::AI::MachineLearning::IEngineBuilder** out) {
  Microsoft::WRL::ComPtr<OnnxruntimeEngineBuilder> onnxruntime_engine_builder;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeEngineBuilder>(&onnxruntime_engine_builder, this));
  RETURN_IF_FAILED(onnxruntime_engine_builder.CopyTo(out));
  return S_OK;
}

const OrtApi* OnnxruntimeEngineFactory::UseOrtApi() {
  return ort_api_;
}

const WinmlAdapterApi* OnnxruntimeEngineFactory::UseWinmlAdapterApi() {
  return winml_adapter_api_;
}

HRESULT OnnxruntimeEngineFactory::GetOrtEnvironment(OrtEnv** ort_env) {
  RETURN_IF_FAILED(environment_->GetOrtEnvironment(ort_env));
  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::EnableDebugOutput(bool is_enabled) {
  RETURN_IF_FAILED(environment_->EnableDebugOutput(is_enabled));
  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::CreateCustomRegistry(IMLOperatorRegistry** registry) {
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api_->CreateCustomRegistry(registry),
                                   ort_api_);
  return S_OK;
}

STDAPI CreateOnnxruntimeEngineFactory(_Out_ Windows::AI::MachineLearning::IEngineFactory** engine_factory) {
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> onnxruntime_engine_factory;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeEngineFactory>(&onnxruntime_engine_factory));
  RETURN_IF_FAILED(onnxruntime_engine_factory.CopyTo(engine_factory));
  return S_OK;
}