"""
Run this script to recreate the original onnx model.
Example usage:
python matmulnbits_add_gen.py out_model_path.onnx
"""

from onnx import helper, numpy_helper, TensorProto

import onnx
import numpy as np
import sys
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'matmulnbits_add_gen')

def clear_field(proto, field):
    proto.ClearField(field)
    return proto

def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))

def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == '':
        node.doc_string = ''
    order_repeated_field(node.attribute, 'name', kwargs.keys())
    return node

def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == '':
        graph.doc_string = ''
    return graph

model = helper.make_model(
    opset_imports=[clear_field(helper.make_operatorsetid('', 21), 'domain'), helper.make_operatorsetid('com.microsoft', 1)],
    ir_version=10,
    graph=make_graph(
        name='MatMul_Add',
        inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, shape=['M', 2]), helper.make_tensor_value_info('bias', TensorProto.FLOAT, shape=[3])],
        outputs=[helper.make_tensor_value_info('C', TensorProto.FLOAT, shape=['M', 3])],
        initializer=[
            numpy_helper.from_array(np.load(os.path.join(DATA_DIR, 'const0_B_Q4.npy')).astype('uint8').reshape([3, 1, 16]), name='B_Q4'),
            numpy_helper.from_array(np.array([-0.125, -0.125, -0.125], dtype='float32'), name='B_scales'),
        ],
        nodes=[
            make_node('MatMulNBits', inputs=['A', 'B_Q4', 'B_scales'], outputs=['matmul_out'], name='matmul_Q4', domain='com.microsoft', K=2, N=3, bits=4, block_size=32),
            make_node('Add', inputs=['matmul_out', 'bias'], outputs=['C'], name='add'),
        ],
    ),
)

if __name__ == '__main__' and len(sys.argv) == 2:
    _, out_path = sys.argv
    onnx.save(model, out_path)
