// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "GraphDescBuilder.h"

using namespace winrt::Windows::AI::MachineLearning::implementation;

namespace Dml::GraphDescBuilder
{
    // TODO: This is a hack which strips the suffix added within Lotus transforms that insert mem copies.
    // This shouldn't be necessary if Lotus exposes the inputs/ouputs in the same order between the kernel
    // for a function, and the graph for that function exposed as a kernel property.  When the ordering 
    // mismatch is fixed (WindowsAI: 21114358, Lotus: 1953), this workaround should be removed.
    static std::string GetFusedNodeArgNameMatchingGraph(const std::string& fusedNodeArgeName)
    {
        // The suffix used when inserting mem copies is equal to the below, followed by an incrementing number.
        const char* suffix = strstr(fusedNodeArgeName.c_str(), "_DmlExecutionProvider_");

        if (suffix)
        {
            return std::string(
                fusedNodeArgeName.begin(),
                fusedNodeArgeName.begin() + (suffix - fusedNodeArgeName.c_str())
            );
        }

        return fusedNodeArgeName;
    }

    const std::string& GetUniqueNodeName(const onnxruntime::Node& node)
    {
        // The node's name is optional, and it might be re-created with a different index
        // and pointer after partitioning occurs.  Use the name of the node's first valid 
        // output as the unique identifier for the node itself.
        for (const auto* arg : node.OutputDefs()) 
        {
            if (arg->Exists())
            {
                return arg->Name();
            }
        }

        assert(false);
        THROW_HR(E_UNEXPECTED);
    }

    GraphDesc BuildGraphDesc(
        const onnxruntime::OpKernelInfo& kernelInfo,
        gsl::span<const uint8_t> isConstGpuGraphInput,
        std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap,
        const onnxruntime::Graph& graph,
        const onnxruntime::ConstPointerContainer<std::vector<onnxruntime::NodeArg*>>& fusedNodeInputDefs,
        const onnxruntime::ConstPointerContainer<std::vector<onnxruntime::NodeArg*>>& fusedNodeOutputDefs,
        const std::unordered_map<std::string, GraphNodeProperties>& graphNodePropertyMap,
        IDMLDevice* device,
        const void* executionHandle)
    {
        struct NodeAndIndex
        {
            uint32_t nodeIndex; // The index of the node itself
            uint32_t targetIndex; // The index of the input/output on the node (e.g. 1 for the second input on a node)
        };

        // Map from Lotus node argument names to the new node and index where it will be produced
        std::unordered_map<std::string, NodeAndIndex> nameToNodeAndIndexMap;

        // Map from Lotus node argument names to input indices of the fused kernel node.
        std::unordered_map<std::string, uint32_t> nameToFusedNodeInputIndex;

        for (size_t inputIndex = 0; inputIndex < fusedNodeInputDefs.size(); ++inputIndex)
        {
            const onnxruntime::NodeArg* graphInput = graph.GetNodeArg(
                GetFusedNodeArgNameMatchingGraph(fusedNodeInputDefs[inputIndex]->Name()));

            nameToFusedNodeInputIndex.emplace(graphInput->Name(), gsl::narrow_cast<uint32_t>(inputIndex));
        }

        StackAllocator<1024> allocator; // Used for converting abstract operator descs into DML_OPERATOR_DESC

        std::vector<NodeInfo> graphNodes;
        std::vector<DML_PREVIEW_INPUT_GRAPH_EDGE> graphInputEdges;
        std::vector<DML_PREVIEW_INTERMEDIATE_GRAPH_EDGE> graphIntermediateEdges;
        std::vector<DML_PREVIEW_OUTPUT_GRAPH_EDGE> graphOutputEdges;

        // Get the topological sorting of Lotus nodes
        // paulm: breaking change from LOTUS that removed GetNodesInTopologicalOrder from Graph
        onnxruntime::GraphViewer viewer(graph);
        const std::vector<onnxruntime::NodeIndex>& orderedNodeIndices = viewer.GetNodesInTopologicalOrder();

        // Avoid using separate command lists for small graphs. This value can be reduced by tuning the 
        // flushing behavior of DmlCommandRecorder.  Its current behavior is to assume that graphs contain
        // enough GPU work to be worth flushing immediately.
        const uint32_t minNodeCountToReuseCommandList = 5;
        bool reuseCommandList = false;
        
        if (orderedNodeIndices.size() >= minNodeCountToReuseCommandList)
        {
            reuseCommandList = true;
        }

        auto constantCpuGraphInputGetter = [&fusedNodeInputDefs, &transferredInitializerMap](const std::string& argName)
        {
            ComPtr<OnnxTensorWrapper> tensorWrapper;

            auto iter = transferredInitializerMap.find(argName);
            if (iter != transferredInitializerMap.end())
            {
                tensorWrapper = wil::MakeOrThrow<OnnxTensorWrapper>(&iter->second);
            }

            return tensorWrapper;
        };

        // Iterate through each node and create a corresponding node in the new graph
        for (size_t sortedNodeIndex : orderedNodeIndices) 
        {
            const onnxruntime::Node& node = *graph.GetNode(sortedNodeIndex);

            const GraphNodeProperties& graphNodeProps = graphNodePropertyMap.find(GetUniqueNodeName(node))->second;
            const auto& requiredConstantCpuInputs = graphNodeProps.graphNodeFactoryRegistration->requiredConstantCpuInputs;

            MLOperatorTensorGetter constantCpuNodeInputGetter = [&node, &constantCpuGraphInputGetter, &requiredConstantCpuInputs](uint32_t inputIndex)
            {
                ComPtr<IMLOperatorTensor> tensor = nullptr;

                // Check whether this specific node requested support for constant CPU inputs
                if (std::find(requiredConstantCpuInputs.begin(), requiredConstantCpuInputs.end(), inputIndex) != requiredConstantCpuInputs.end())
                {
                    const onnxruntime::NodeArg* arg = node.InputDefs()[inputIndex];
                    tensor = constantCpuGraphInputGetter(arg->Name());
                }

                return tensor;
            };

            DmlGraphNodeCreateInfo graphNodeInfo;
            graphNodeProps.graphNodeFactoryRegistration->factory(
                node,
                constantCpuNodeInputGetter,
                executionHandle,
                &graphNodeInfo
            );

            // Determine the number of valid inputs and outputs of this node.  The graph currently supports opererators
            // with unused inputs and outputs only at the end of each list.  
            uint32_t validOpInputCount = 0;
            uint32_t validOpOutputCount = 0;

            for (uint32_t i = 0; i < graphNodeInfo.kernelInputIndices.size(); ++i)
            {
                if (graphNodeInfo.kernelInputIndices[i] != std::numeric_limits<uint32_t>::max())
                {
                    assert(i - validOpInputCount == 0);
                    ++validOpInputCount;
                }
            }

            for (uint32_t i = 0; i < graphNodeInfo.kernelOutputIndices.size(); ++i)
            {
                if (graphNodeInfo.kernelOutputIndices[i] != std::numeric_limits<uint32_t>::max())
                {
                    assert(i - validOpOutputCount == 0);
                    ++validOpOutputCount;
                }
            }

            uint32_t nodeIndex = gsl::narrow_cast<uint32_t>(graphNodes.size());
            AbstractOperatorDesc opDesc = *graphNodeInfo.desc; // Make a copy

            // Retrieve lists of input and output tensor descs. These point into the opDesc, which allows us to modify
            // the tensor descs through these pointers.
            std::vector<DmlBufferTensorDesc*> inputTensorDescs = opDesc.GetInputTensors();
            std::vector<DmlBufferTensorDesc*> outputTensorDescs = opDesc.GetOutputTensors();

            // Set connections of the new node
            for (uint32_t inputIndex = 0; inputIndex < validOpInputCount; ++inputIndex)
            {
                uint32_t kernelInputIndex = graphNodeInfo.kernelInputIndices[inputIndex];

                const onnxruntime::NodeArg* arg = node.InputDefs()[kernelInputIndex];

                if (arg->Exists())
                {
                    auto iter = nameToFusedNodeInputIndex.find(arg->Name());
                    if (iter != nameToFusedNodeInputIndex.end())
                    {
                        // This is a graph input

                        const uint32_t fusedNodeInputIndex = iter->second;

                        DML_PREVIEW_INPUT_GRAPH_EDGE edge = {};
                        edge.GraphInputIndex = fusedNodeInputIndex;
                        edge.ToNodeIndex = nodeIndex;
                        edge.ToNodeInputIndex = inputIndex;
                        graphInputEdges.push_back(edge);

                        // If this is a constant input, set the appropriate flags on the desc
                        if (isConstGpuGraphInput[fusedNodeInputIndex])
                        {
                            DmlBufferTensorDesc* tensorDesc = inputTensorDescs[inputIndex];

                            tensorDesc->flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
                        }
                    }
                    else
                    {
                        const auto& inputNodeAndIndex = nameToNodeAndIndexMap.at(arg->Name());

                        DML_PREVIEW_INTERMEDIATE_GRAPH_EDGE edge = {};
                        edge.FromNodeIndex = inputNodeAndIndex.nodeIndex;
                        edge.FromNodeOutputIndex = inputNodeAndIndex.targetIndex;
                        edge.ToNodeIndex = nodeIndex;
                        edge.ToNodeInputIndex = inputIndex;
                        graphIntermediateEdges.push_back(edge);
                    }
                }
            }
            
            // Store the new node for lookup when downstream nodes consume it.

            for (uint32_t outputIndex = 0; outputIndex < validOpOutputCount; ++outputIndex) 
            {
                uint32_t kernelOutputIndex = graphNodeInfo.kernelOutputIndices[outputIndex];
                const onnxruntime::NodeArg* arg = node.OutputDefs()[kernelOutputIndex];
                if (arg->Exists())
                {
                    nameToNodeAndIndexMap[arg->Name()] = NodeAndIndex{ nodeIndex, outputIndex };
                }
            }

            DML_OPERATOR_DESC dmlDesc = SchemaHelpers::ConvertOperatorDesc(opDesc, &allocator);

            ComPtr<IDMLOperator> op;
            THROW_IF_FAILED(device->CreateOperator(&dmlDesc, IID_PPV_ARGS(&op)));
            allocator.Reset();

            NodeInfo nodeInfo = {};
            nodeInfo.op = std::move(op);
            graphNodes.push_back(std::move(nodeInfo));
        }

        assert(graphNodes.size() == orderedNodeIndices.size());

        // Add graph output nodes, which might be in a different order from the encapsulating node
        for (size_t outputIndex = 0; outputIndex < fusedNodeOutputDefs.size(); ++outputIndex)
        {
            const onnxruntime::NodeArg* graphOutput = graph.GetNodeArg(
                GetFusedNodeArgNameMatchingGraph(fusedNodeOutputDefs[outputIndex]->Name()));

            const auto& outputNodeAndIndex = nameToNodeAndIndexMap.at(graphOutput->Name());

            DML_PREVIEW_OUTPUT_GRAPH_EDGE edge = {};
            edge.FromNodeIndex = outputNodeAndIndex.nodeIndex;
            edge.FromNodeOutputIndex = outputNodeAndIndex.targetIndex;
            edge.GraphOutputIndex = gsl::narrow_cast<uint32_t>(outputIndex);
            graphOutputEdges.push_back(edge);
        }
        
        GraphDesc graphDesc{};
        graphDesc.nodes = std::move(graphNodes);
        graphDesc.inputEdges = std::move(graphInputEdges);
        graphDesc.outputEdges = std::move(graphOutputEdges);
        graphDesc.intermediateEdges = std::move(graphIntermediateEdges);
        graphDesc.reuseCommandList = reuseCommandList;
        return graphDesc;
    }
}