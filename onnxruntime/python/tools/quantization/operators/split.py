import itertools

import onnx
from onnx import onnx_pb as onnx_proto

from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase

class QSplit(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_inputs(node, [0])
        if quantized_input_names is None:
            return super().quantize()

        quantized_node_name = ""
        if node.name != "":
            quantized_node_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))

        # Output just derive the scale/zero from input
        quantized_output_names = []
        for output_name in node.output:
            quantized_output_name = output_name + "quantized"
            quantized_output_names.append(quantized_output_name)
            q_output = QuantizedValue(
                output_name,
                quantized_output_name,
                scale_names[0],
                zero_point_names[0],
                QuantizedValueType.Input,
            )
            self.quantizer.quantized_value_map[output_name] = q_output

        if len(node.input) > 1:
            quantized_input_names = quantized_input_names.extend(node.input[1:])
        quantized_node = onnx.helper.make_node(
            node.op_type, quantized_input_names, quantized_output_names, quantized_node_name, **kwargs
        )

        nodes.append(quantized_node)
        self.quantizer.new_nodes += nodes

class QDQSplit(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Split"
        super().quantize()
        # if self.disable_qdq_for_node_output:
        #     tensors_to_quantize = node.input[0]
        # else:
        #     tensors_to_quantize = itertools.chain(node.input[0], node.output)
        # for tensor_name in tensors_to_quantize:
        #     self.quantizer.quantize_tensor(tensor_name)