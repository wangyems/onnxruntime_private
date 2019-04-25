#include <iostream>
#include <cstdlib>
#include <map>
#include <string>
#include <memory>
#include <cstdlib>

#include <inference_engine.hpp>
#include <ie_builders.hpp>

#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

#include "openvino_graph.h"
#include "openvino_node.h"

namespace openvino_ep {

OpenVINOGraph::OpenVINOGraph(onnxruntime::Node* fused_node, std::string /*device_info*/) {
	//TODO: parse device info to obtain the following values


  device_id_ = "CPU";
  precision_ = InferenceEngine::Precision::FP32;
  std::string precision_str = "FP32";

#ifdef OPENVINO_CONFIG_CPU_FP32
	device_id_ = "CPU";
	precision_ = InferenceEngine::Precision::FP32;
	precision_str = "FP32";
#endif
#ifdef OPENVINO_CONFIG_GPU_FP32
	device_id_ = "GPU";
	precision_ = InferenceEngine::Precision::FP32;
	precision_str = "FP32";
#endif
#ifdef OPENVINO_CONFIG_GPU_FP16
	device_id_ = "GPU";
	precision_ = InferenceEngine::Precision::FP16;
	precision_str = "FP16";
#endif
#ifdef OPENVINO_CONFIG_MYRIAD
	device_id_ = "MYRIAD";
	precision_ = InferenceEngine::Precision::FP16;
	precision_str = "FP16";
#endif
#ifdef OPENVINO_CONFIG_VAD_R
	device_id_ = "HDDL";
	precision_ = InferenceEngine::Precision::FP16;
	precision_str = "FP16";
#endif

  std::cout<< "OpenVINO EP device:" << device_id_ << std::endl;
  std::cout<< "OpenVINO EP precision:" << precision_str << std::endl;


	num_inf_reqs_ = (device_id_ == "HDDL") ? 8 : 1;

	fused_node_ = fused_node;


	// Make a local copy of the initializers
	const auto& attributes = fused_node->GetAttributes();
	const auto& izers_list = attributes.at("initializers").tensors();
	for(const auto& izer : izers_list) {
	  izers_map_.insert({izer.name(), &izer});
	}

	/*
	for(auto iter : izers_map_) {
	  auto name = iter.first;
	  float* ptr = (float*) iter.second->raw_data().c_str();
	  std::cout << name << " , " << ptr << std::endl;
	}
	*/



  builder_ = std::make_shared<InferenceEngine::Builder::Network>(fused_node_->Name());
	onnx_graph_ = &(fused_node_->GetFunctionBody()->Body());

	/*
	auto inputs =  onnx_graph_->GetInputs();
	auto inputaninits = onnx_graph_->GetInputsIncludingInitializers();
	auto izer_list = onnx_graph_->GetAllInitializedTensors();
	*/

  cnn_network_ = BuildCNNNetwork();

  // TODO: make this a debug option
  // Uncomment the below code to enable dumping IE CNN graphs.
  //std::string file_name = "./conv_" + fused_node->Name();
  //cnn_network_->serialize( file_name+".xml", file_name+".bin");


  infer_requests_ = GetExecutableHandle(cnn_network_, device_id_, precision_);
}

std::vector<std::string> OpenVINOGraph::GetEnvLdLibraryPath() {
    std::string plugin_path = std::getenv("LD_LIBRARY_PATH");
    std::vector<std::string> paths;
    std::string token;
    std::istringstream tokenStream(plugin_path);
    char delimiter = ':';

    while (std::getline(tokenStream , token, delimiter)) {
      paths.push_back(token);
    }
    return paths;
}

std::shared_ptr<InferenceEngine::CNNNetwork> OpenVINOGraph::BuildCNNNetwork() {

  // Create Input nodes
  for(auto* input_arg : onnx_graph_->GetInputs()) {
    if(IsInitializer(input_arg->Name())) {
      //workaround for the input args & initializers bug
      continue;
    }
    auto input_node = std::make_shared<OpenVINONode>(input_arg, this);
    input_node->is_input_node_ = true;
    openvino_nodes_.push_back(input_node);
    tensor_producers_.insert({input_arg->Name(), input_node});
  }

  // Create Output nodes
  for(auto* output_arg : onnx_graph_->GetOutputs()) {
    auto output_node = std::make_shared<OpenVINONode>(output_arg, this);
    output_node->is_output_node_ = true;
    openvino_nodes_.push_back(output_node);
  }


  // Create graph nodes
  for (int i = 0; i < onnx_graph_->NumberOfNodes(); i++) {
    auto* onnx_node = onnx_graph_->GetNode(i);
    auto openvino_node = std::make_shared<OpenVINONode>(onnx_node, this);
    openvino_nodes_.push_back(openvino_node);
    for(auto* output_arg : onnx_node->OutputDefs()) {
    	tensor_producers_.insert({output_arg->Name(), openvino_node});
    }
  }

  // Create OpenVINO ops for intermediate node (non graph I/O nodes)
  for (auto openvino_node : openvino_nodes_) {
    //   if(openvino_node->onnx_node_->OpType() == "Unsqueeze"){
        //   openvino_node->CreateUnsqueezeLayer(precision_,const_blob_map_);
    //   }
    //   else{
        openvino_node->InitializeOp(const_blob_map_);
    //   }
  }


  // Connect the OpenVINO Graph
  for(auto openvino_node : openvino_nodes_) {
	  openvino_node->ConnectToInputs();
  }

  auto inetworkptr = builder_->build();

  std::cout << " builder built\n";

  return std::make_shared<InferenceEngine::CNNNetwork>(
      InferenceEngine::Builder::convertToICNNNetwork(inetworkptr));
}

std::vector<InferenceEngine::InferRequest::Ptr> OpenVINOGraph::GetExecutableHandle(
    std::shared_ptr<InferenceEngine::CNNNetwork> network,
    const std::string& device, InferenceEngine::Precision precision) {


  // TODO: make this better

  precision = InferenceEngine::Precision::FP32;


  // Load Plugin for inference engine
  std::cout << "[OpenVINO-EP]Loading plugin" << std::endl;

  std::vector<std::string> plugin_path = GetEnvLdLibraryPath();
  plugin_path.push_back("");
  InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher(
      plugin_path).getPluginByDevice(device);
  //InferenceEngine::printPluginVersion(plugin, std::cout);

  // Configure input & output
  // Prepare input blobs
  std::cout << "[OpenVINO-EP]Preparing input blobs" << std::endl;

  auto inputInfo = network->getInputsInfo();
  for(auto iter = inputInfo.begin(); iter != inputInfo.end(); ++iter) {
    iter->second->setPrecision(precision);
    switch (iter->second->getTensorDesc().getDims().size()) {
      case 1:
        iter->second->setLayout(InferenceEngine::Layout::C);
        break;
      case 2:
        iter->second->setLayout(InferenceEngine::Layout::NC);
        break;
      case 3:
        iter->second->setLayout(InferenceEngine::Layout::CHW);
        break;
      case 4:
        iter->second->setLayout(InferenceEngine::Layout::NCHW);
        break;
      case 5:
        iter->second->setLayout(InferenceEngine::Layout::NCDHW);
        break;
      default:
        throw "Invalid Dims type for input data map for: " + iter->first;
    }
  }

  network->setBatchSize(1);

  // Prepare output blobs
  auto outputInfo = network->getOutputsInfo();
  for(auto iter = outputInfo.begin(); iter != outputInfo.end(); ++iter) {
    iter->second->setPrecision(precision);
    switch (iter->second->getTensorDesc().getDims().size()) {
      case 1:
        iter->second->setLayout(InferenceEngine::Layout::C);
        break;
      case 2:
        iter->second->setLayout(InferenceEngine::Layout::NC);
        break;
      case 3:
        iter->second->setLayout(InferenceEngine::Layout::CHW);
        break;
      case 4:
        iter->second->setLayout(InferenceEngine::Layout::NCHW);
        break;
      case 5:
        iter->second->setLayout(InferenceEngine::Layout::NCDHW);
        break;
      default:
        throw "Invalid Dims type for output data map for: " + iter->first;
    }
  }

  // Loading model to the plugin
  std::cout << "[OpenVINO-EP]Loading model to the plugin" << std::endl;
  InferenceEngine::ExecutableNetwork exeNetwork = plugin.LoadNetwork(*network,
      { });

  // Create infer request
  std::cout << "[OpenVINO-EP]Creating Infer requests : " << num_inf_reqs_ << std::endl;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests;
  for(int i = 0; i < num_inf_reqs_; i++) {
      infer_requests.push_back(exeNetwork.CreateInferRequestPtr());
  }

  return infer_requests;
}

std::shared_ptr<InferenceEngine::CNNNetwork> OpenVINOGraph::GetCNNNetwork() {
  return cnn_network_;
}


void OpenVINOGraph::Infer(onnxruntime::ONNXRunTimeTensor* input_tensors,
		size_t num_inputs, onnxruntime::ONNXRunTimeTensor* output_tensors,
		size_t num_outputs, onnxruntime::AllocateFunc& output_allocator_func,
		onnxruntime::AllocatorHandle& output_allocator_handle) {

  std::cout << "INFERENCE STARTED\n";
  std::cout << "Num inputs: " << num_inputs << std::endl;
  for (int i=0; i< num_inputs; i++) {
    std::cout << "intput: " << i << std::endl;
    for (int j=0; j < input_tensors[i].ndim; j++) {
      std::cout << " dim " << j << " = " << input_tensors[i].shape[j];
    }
    std::cout << std::endl;
  }

	// Check I/O sizes
	auto graph_input_info = cnn_network_->getInputsInfo();
	if (num_inputs != graph_input_info.size()) {
		throw "OpenVINO Inference: Inputs count mismatch!";
	}

	auto graph_output_info = cnn_network_->getOutputsInfo();
	if (num_outputs != graph_output_info.size()) {
		throw "OpenVINO Inference: Outputs count mismatch!";
	}

	//
	// Copies the same input to all infer request blobs and
	// starts an async inference on each of them.
	// Output from only the first infer_request is returned.
	//



	// Prepare input
	for(auto infer_request : infer_requests_) {

    size_t i = 0;
    for (auto input_info_iter = graph_input_info.begin();
        input_info_iter != graph_input_info.end(); ++input_info_iter, ++i) {

      // Get OpenVINO's input buffer
      auto graph_input_blob = infer_request->GetBlob(input_info_iter->first);
      auto graph_input_buffer =
          graph_input_blob->buffer().as<
              InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

      // Get data size
      size_t num_input_elements = 1;
      for (auto dim : input_info_iter->second->getTensorDesc().getDims()) {
        num_input_elements *= dim;
      }

      size_t input_data_size = num_input_elements * sizeof(float);

      // Copy input data into OpenVINO's input buffer
      std::memcpy(graph_input_buffer, input_tensors[i].data, input_data_size);
    }
  }


	// Start Async inferences
	for(auto infer_request : infer_requests_) {
	  infer_request->StartAsync();
	}

	// Wait for results
	for(auto infer_request : infer_requests_) {
	  infer_request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
	}

	// Process output

	auto infer_request = infer_requests_[0];

	size_t i = 0;
	for (auto output_info_iter = graph_output_info.begin();
			output_info_iter != graph_output_info.end();
			++output_info_iter, ++i) {

		// Get OpenVINO's output buffer
		auto graph_output_blob = infer_request->GetBlob(
				output_info_iter->first);
		auto graph_output_buffer =
				graph_output_blob->buffer().as<
						InferenceEngine::PrecisionTrait<
								InferenceEngine::Precision::FP32>::value_type*>();

        // auto size = graph_output_blob->size();

        // for(int i=0; i< size; i++){
        //     std::cout << "Output values " << graph_output_buffer[i] << std::endl;
        // }




		// Get data size & initialize output tensor info
		auto graph_output_dims = graph_output_blob->getTensorDesc().getDims();
		auto num_dims = graph_output_dims.size();
		size_t output_data_size = graph_output_blob->byteSize();

    // TODO: Memory Leak!!!!
    // fix before shipping.
		output_tensors[i].shape = new int64_t[num_dims];
		for (int j = 0; j < num_dims; j++) {
			output_tensors[i].shape[j] = (int64_t)graph_output_dims[j];
		}


		output_tensors[i].ndim = num_dims;
		output_tensors[i].dtype = onnxruntime::DType::TFloat32;
		output_tensors[i].data = (*output_allocator_func)(output_allocator_handle, 64, output_data_size);
		std::memcpy(output_tensors[i].data, graph_output_buffer, output_data_size);

	}
}
} // namespace openvino_ep
