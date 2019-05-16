#pragma once
#include "core/framework/op_kernel.h"
#include "core/framework/func_api.h"
#include "core/graph/function.h"
namespace onnxruntime {

void* allocate_helper_func(void* allocator, size_t alignment, size_t size);

void release_helper_func(void* allocator, void* p);

DType ORT_type_to_c_type(MLDataType type);

//A kernel that wrapper the ComputeFunction call generated by execution provider when fuse the sub-graph
class FunctionKernel : public OpKernel {
 public:
  //The original design is we load the dll, find the entry point and wrapper it.
  //Here for quick prototype, we keep the entry pointer in the node.
  explicit FunctionKernel(const OpKernelInfo& info) : OpKernel(info) {
    num_inputs_ = info.node().InputDefs().size();
    num_outputs_ = info.node().OutputDefs().size();
    CreateFunctionStateFunc create_func;
    auto status = info.GetFusedFuncs(&func_, &create_func, &release_func_);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    if (create_func) {
      //TODO: we are only provide host allocate method in compute context.
      //Do we need to hold the ref-counting here?
      host_allocator_ = info.GetAllocator(0, OrtMemType::OrtMemTypeDefault);
      ComputeContext context = {allocate_helper_func, release_helper_func, host_allocator_.get(), info.node().Name().c_str()};
      ORT_ENFORCE(create_func(&context, &func_state_) == 0);
    }
  }

  ~FunctionKernel() override {
    if (release_func_ && func_state_) {
      release_func_(func_state_);
    }
  }

  Status Compute(OpKernelContext* context) const override {
    std::vector<ONNXRunTimeTensor> input_tensors;
    for (int i = 0; static_cast<size_t>(i) < num_inputs_; i++) {
      const Tensor* input = context->Input<Tensor>(i);
      auto& shape = input->Shape();
      auto& dims = shape.GetDims();
      ONNXRunTimeTensor input_tensor = {
          const_cast<void*>(input->DataRaw()),
          shape.NumDimensions(),
          //hard code to double now
          ORT_type_to_c_type(input->DataType()),
          dims.empty() ? nullptr : const_cast<int64_t*>(&dims[0])};
      input_tensors.push_back(input_tensor);
    }

    std::vector<ONNXRunTimeTensor> output_tensors(num_outputs_);
    int ret = func_(func_state_, input_tensors.empty() ? nullptr : &input_tensors[0], input_tensors.size(), &output_tensors[0], output_tensors.size());
    if (ret != 0)
      return Status(common::ONNXRUNTIME, common::FAIL, "FuncKernel call failed with error code: " + std::to_string(ret));

    for (int i = 0; static_cast<size_t>(i) < num_outputs_; i++) {
      TensorShape output_shape(std::vector<int64_t>(output_tensors[i].shape, output_tensors[i].shape + output_tensors[i].ndim));
      Tensor* output = context->Output(i, output_shape);
      auto data = output->MutableDataRaw();
      //TODO: for string tensors, this copy is not correct.
      ORT_ENFORCE(output->DataType() != DataTypeImpl::GetType<std::string>());
      memcpy(data, output_tensors[i].data, output->DataType()->Size() * output_shape.Size());
      //Release output tensors (buffer, shape).
      host_allocator_->Free(output_tensors[i].data);
      // for shape, becauset the TempSpaceAllocator we use could be a device allocator, if the kernel is assigned to a device like gpu.
      // so we prefer to directly allocate shape on heap. otherwise we need pass in multile allocator function for host and device.
      delete[] output_tensors[i].shape;
    }

    return Status::OK();
  }

 private:
  ComputeFunc func_;
  DestroyFunctionStateFunc release_func_;
  FunctionState func_state_;
  size_t num_inputs_;
  size_t num_outputs_;
  AllocatorPtr host_allocator_;
};
}  // namespace onnxruntime
