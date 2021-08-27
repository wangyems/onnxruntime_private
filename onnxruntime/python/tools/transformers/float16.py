#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# This file is modified from https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py

import itertools
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto

import logging

logger = logging.getLogger(__name__)

def _npfloat16_to_int(np_list):
    '''
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    '''
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]


def convert_np_to_float16(np_array, min_positive_val=1e-7, max_finite_val=1e4):
    '''
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    '''
    def between(a, b, c):
        return np.logical_and(a < b, b < c)

    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float('inf')), max_finite_val, np_array)
    np_array = np.where(between(float('-inf'), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)


def convert_tensor_float_to_float16(tensor, min_positive_val=1e-7, max_finite_val=1e4):
    """Convert tensor float to float16.

    Args:
        tensor (TensorProto): the tensor to converted.
        min_positive_val (float, optional): minimal positive value. Defaults to 1e-7.
        max_finite_val (float, optional): maximal finite value. Defaults to 1e4.

    Raises:
        ValueError: input type is an ONNX TensorProto.

    Returns:
        TensorProto: the converted tensor.
    """

    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(np.array(tensor.float_data), min_positive_val, max_finite_val)
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype='float32')
            # convert float to float16
            float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tostring()
    return tensor


def make_value_info_from_tensor(tensor):
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)


DEFAULT_OP_BLOCK_LIST = [
    'ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer', 'FeatureVectorizer', 'Imputer',
    'LabelEncoder', 'LinearClassifier', 'LinearRegressor', 'Normalizer', 'OneHotEncoder', 'SVMClassifier',
    'SVMRegressor', 'Scaler', 'TreeEnsembleClassifier', 'TreeEnsembleRegressor', 'ZipMap', 'NonMaxSuppression', 'TopK',
    'RoiAlign', 'Resize', 'Range', 'CumSum', 'Min', 'Max', 'Upsample'
]


def convert_float_to_float16(model,
                             min_positive_val=1e-7,
                             max_finite_val=1e4,
                             keep_io_types=False,
                             disable_shape_infer=False,
                             op_block_list=None,
                             node_block_list=None):
    """Convert tensor float type in the ONNX ModelProto input to tensor float16.

    Args:
        model (ModelProto): The ONNX model to convert.
        min_positive_val (float, optional): minimal positive value. Defaults to 1e-7.
        max_finite_val (float, optional): maximal finite value. Defaults to 1e4.
        keep_io_types (Union[bool, List[str]], optional): It could be boolean or a list of float32 input/output names. If True, model inputs/outputs should be left as float32. Defaults to False.
        disable_shape_infer (bool, optional): Skips running onnx shape/type inference. Useful if shape inference has been done. Defaults to False.
        op_block_list (List[str], optional): List of op types to leave as float32. Defaults to None, which will use `float16.DEFAULT_OP_BLOCK_LIST` as default.
        node_block_list (List[str], optional): List of node names to leave as float32. Defaults to None.

    Raises:
        ValueError: input is not ONNX ModelProto.

    Returns:
        ModelProto: converted model.
    """
    assert min_positive_val >= 5.96e-08, "smallest positive float16 value: subnormal 5.96e-08, and normalized 6.104e-05"
    assert max_finite_val <= float(np.finfo(np.float16).max), "largest float16 value: 65504"
    func_infer_shape = None
    if not disable_shape_infer and onnx.__version__ >= '1.2':
        try:
            from onnx.shape_inference import infer_shapes
            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError('Expected model type is an ONNX ModelProto but got %s' % type(model))

    # create blocklists
    if op_block_list is None:
        op_block_list = DEFAULT_OP_BLOCK_LIST
    if node_block_list is None:
        node_block_list = []
    op_block_list = set(op_block_list)
    node_block_list = set(node_block_list)

    # create a queue for BFS
    queue = []
    value_info_list = []
    node_list = []
    # type inference on input model
    if func_infer_shape is not None:
        model = func_infer_shape(model)
    queue.append(model)
    name_mapping = {}
    graph_io_to_skip = set()
    io_casts = set()

    fp32_inputs = [n.name for n in model.graph.input if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT]
    fp32_outputs = [n.name for n in model.graph.output if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT]
    if isinstance(keep_io_types, list):
        fp32_inputs = [n for n in fp32_inputs if n in keep_io_types]
        fp32_outputs = [n for n in fp32_outputs if n in keep_io_types]
    elif not keep_io_types:
        fp32_inputs = []
        fp32_outputs = []

    for i, n in enumerate(model.graph.input):
        if n.name in fp32_inputs:
            output_name = 'graph_input_cast_' + str(i)
            name_mapping[n.name] = output_name
            graph_io_to_skip.add(n.name)

            node_name = 'graph_input_cast' + str(i)
            new_value_info = model.graph.value_info.add()
            new_value_info.CopyFrom(n)
            new_value_info.name = output_name
            new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            # add Cast node (from tensor(float) to tensor(float16) after graph input
            new_node = [helper.make_node('Cast', [n.name], [output_name], to=10, name=node_name)]
            model.graph.node.extend(new_node)
            value_info_list.append(new_value_info)
            io_casts.add(node_name)

    for i, n in enumerate(model.graph.output):
        if n.name in fp32_outputs:
            input_name = 'graph_output_cast_' + str(i)
            name_mapping[n.name] = input_name
            graph_io_to_skip.add(n.name)

            node_name = 'graph_output_cast' + str(i)
            # add Cast node (from tensor(float16) to tensor(float) before graph output
            new_value_info = model.graph.value_info.add()
            new_value_info.CopyFrom(n)
            new_value_info.name = input_name
            new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            new_node = [helper.make_node('Cast', [input_name], [n.name], to=1, name=node_name)]
            model.graph.node.extend(new_node)
            value_info_list.append(new_value_info)
            io_casts.add(node_name)

    fp32_initializer_counters = {}
    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        fp32_initializer_counters[n.name] = [0,
                                                             0]  # two counters: used by fp16 nodes, used by fp32 nodes

                for n in q.node:
                    # if n is in the block list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if n.name in io_casts:
                        continue
                    for i in range(len(n.input)):
                        if n.input[i] in name_mapping:
                            n.input[i] = name_mapping[n.input[i]]
                    for i in range(len(n.output)):
                        if n.output[i] in name_mapping:
                            n.output[i] = name_mapping[n.output[i]]

                    is_node_blocked = n.op_type in op_block_list or n.name in node_block_list
                    for input in n.input:
                        if input in fp32_initializer_counters:
                            fp32_initializer_counters[input][int(is_node_blocked)] += 1

                    if is_node_blocked:
                        node_list.append(n)
                    else:
                        if n.op_type == 'Cast':
                            for attr in n.attribute:
                                if attr.name == 'to' and attr.i == 1:
                                    attr.i = 10
                                    break
                        for attr in n.attribute:
                            next_level.append(attr)
            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, onnx_proto.AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
                q.t.CopyFrom(convert_tensor_float_to_float16(q.t, min_positive_val, max_finite_val))
                for n in q.tensors:
                    n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
            # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        # When intializer is not used by fp32 node in current or parent levels, we will converted it to float16.
                        # Note that it might be used by subgraph, which might cause some precision loss in subgraph.
                        if fp32_initializer_counters[n.name][1] == 0:
                            n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
                            value_info_list.append(make_value_info_from_tensor(n))
                        else:
                            # It is rare that an initializer is used by both fp32 and fp16 nodes.
                            # TODO: Shall we add a Cast, or kept one copy in FP32 and another in FP16 for such case?
                            assert fp32_initializer_counters[n.name][
                                0] == 0, f"Not implemented: initializer {n.name} is used by both fp32 and fp16 nodes."

                # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                for n in itertools.chain(q.input, q.output, q.value_info):
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        if n.name not in graph_io_to_skip:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                            value_info_list.append(n)
        queue = next_level

    # process the nodes in block list that doesn't support tensor(float16)
    for node in node_list:
        # if input's name is in the value_info_list meaning input is tensor(float16) type,
        # insert a float16 to float Cast node before the node,
        # change current node's input name and create new value_info for the new name
        for i in range(len(node.input)):
            input = node.input[i]
            for value_info in value_info_list:
                if input == value_info.name:
                    # create new value_info for current node's new input name
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    output_name = node.name + '_input_cast_' + str(i)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float16) to tensor(float) before current node
                    node_name = node.name + '_input_cast' + str(i)
                    new_node = [helper.make_node('Cast', [input], [output_name], to=1, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.input[i] = output_name
                    break
        # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float to
        # float16 Cast node after the node, change current node's output name and create new value_info for the new name
        for i in range(len(node.output)):
            output = node.output[i]
            for value_info in value_info_list:
                if output == value_info.name:
                    # create new value_info for current node's new output
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    input_name = node.name + '_output_cast_' + str(i)
                    new_value_info.name = input_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float) to tensor(float16) after current node
                    node_name = node.name + '_output_cast' + str(i)
                    new_node = [helper.make_node('Cast', [input_name], [output], to=10, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.output[i] = input_name
                    break
    return model
