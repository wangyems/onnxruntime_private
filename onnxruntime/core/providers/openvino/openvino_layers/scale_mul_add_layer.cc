#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

#include <inference_engine.hpp>
#include <ie_builders.hpp>

#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

#include "core/providers/openvino/openvino_graph.h"


namespace openvino_ep {

void OpenVINONode::CreateScaleMulAddLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    int type,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map,
    std::map<std::string,InferenceEngine::Blob::Ptr>& blob_map) {

        std::cout << "In scale shift layer" << std::endl;

  auto scale_shift_mul_layer =
      std::make_shared<InferenceEngine::Builder::ScaleShiftLayer>(
          onnx_node_->Name());

  //
  // *** Set inputs ***
  //
//   float* scale_tensor = nullptr;
//   float* mean_tensor = nullptr;
//   float* var_tensor = nullptr;
//   float* bias_tensor = nullptr;

  auto formal_params = onnx_node_->Op()->inputs();
  if(blob_map.count("OC2_DUMMY_0")){
      std::cout << "Success" << std::endl;
  }

//   size_t num_channels = 0;
    // auto attributes = onnx_node_->GetAttributes();
    // auto epsilon = attributes["epsilon"].f();

  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();
    std::cout << "Formal name is " << formal_name << std::endl;

    if (formal_name == "A") {

      // Set Input info
      std::shared_ptr<OpenVINONode> in_ov_node = nullptr;

    //   auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*(input_defs_[i]->Shape()));
    //   num_channels = shape_vector[1];
    //   std::cout << "Num of channels is " << num_channels << std::endl;

      if (node_connects_to_graph_inputs_) {
        auto input_name = input_defs_[i]->Name();
        in_ov_node = openvino_io_map[input_name];
      } else {
        in_ov_node = onnx_openvino_map[&(input_edges_[0].GetNode())];
      }
      InferenceEngine::idx_t in_port = 0;
      input_connections_.push_back( { in_ov_node, in_port });

    } else if (formal_name == "B") {

        if(type == 1){
            auto input_name = input_defs_[i]->Name();

            auto it = blob_map.find(input_name);
            if(it != blob_map.end()){

                auto scalePtr = it->second;
                scale_shift_mul_layer->setWeights(scalePtr);
            }
        }
        else if(type == 2){

            auto input_name = input_defs_[i]->Name();

            auto it = blob_map.find(input_name);
            if(it != blob_map.end()){

                auto scalePtr = it->second;
                scale_shift_mul_layer->setBiases(scalePtr);
                auto size = scalePtr->size();
                float* scale = new float[size];
                for(int i =0; i< size; i++){
                    scale[i] = 1;
                }
                auto ptrWeights = InferenceEngine::make_shared_blob(
                    InferenceEngine::TensorDesc(precision, {size},
                        InferenceEngine::Layout::C),scale);
                scale_shift_mul_layer->setWeights(ptrWeights);
            }
        }


    } else {
      std::stringstream msg;
      msg << "Node: " << onnx_node_->Name() << "| Param: "
          << formal_name.c_str() << "not found";
      throw msg.str();

    }


  }

    // float* new_scale = new float[num_channels];
    // float* new_bias = new float[num_channels];

    // for(int i=0; i < num_channels; i++){
    //     float den = var_tensor[i] + epsilon;
    //     float den_sqrt = sqrt(den);
    //     new_scale[i] = scale_tensor[i]/den_sqrt;
    //     // std::cout << "New scale is " << new_scale[i] << std::endl;

    //     float num = scale_tensor[i] * mean_tensor[i] * -1;
    //     new_bias[i] = num/den_sqrt + bias_tensor[i];
        // std::cout << "New bias is " << new_bias[i] << std::endl;
    // }

/*    float scale_broadcast[1][64][112][112] = {{{{0}}}};

    // float* scale_broadcast = new float[1][64][112][112];
    for(int i=0; i < 64; i++){
        for(int j=0; j < 112; j++){
            for(int k=0; k < 112; k++){
                scale_broadcast[0][i][j][k] = new_scale[i];
            }
        }
    }
    float bias_broadcast[1][64][112][112] = {{{{0}}}};
    for(int i=0; i < 64; i++){
        for(int j=0; j < 112; j++){
            for(int k=0; k < 112; k++){
                bias_broadcast[0][i][j][k] = new_bias[i];
            }
        }
    }*/

    // std::cout << "Scale Broadcast " << scale_broadcast[0][1][1][1]  << std::endl;
    // InferenceEngine::SizeVector size;
    // size.push_back(1);
    // size.push_back(64);
    // size.push_back(112);
    // size.push_back(112);
    // auto ptrWeights = InferenceEngine::make_shared_blob(
    //       InferenceEngine::TensorDesc(precision,{num_channels},
    //         InferenceEngine::Layout::C), new_scale);
    // auto ptrBiases = InferenceEngine::make_shared_blob(
    //       InferenceEngine::TensorDesc(precision,{num_channels},
    //         InferenceEngine::Layout::C), new_bias);
    // scale_shift_layer->setWeights(ptrWeights);
    // scale_shift_layer->setBiases(ptrBiases);
  //
  // *** Set Outputs ***
  //
  formal_params = onnx_node_->Op()->outputs();
  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();
    if (formal_name == "C") {
        std::cout << "Output is set " << std::endl;

      std::shared_ptr<OpenVINONode> out_ov_node = nullptr;
      if (node_connects_to_graph_outputs_) {
        auto output_name = output_defs_[i]->Name();
        out_ov_node = openvino_io_map[output_name];
      } else {
        out_ov_node = onnx_openvino_map[&(output_edges_[0].GetNode())];
      }
      InferenceEngine::idx_t out_port = 0;
      output_connections_.push_back( { out_ov_node, out_port });

    } else {
      std::stringstream msg;
      msg << "Node: " << onnx_node_->Name() << "| Param: " << formal_name
          << "not found";
    //   throw msg.str();
    }
  }

//   std::cout << "Layer done" << std::endl;

  layerID_ = builder->addLayer(*scale_shift_mul_layer);
}
} // namespce openvino_ep
