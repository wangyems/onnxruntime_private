from pathlib import Path

import onnx
from onnx import TensorProto, helper

# This model contains a MatMul where:
# - A has shape [M, K] and `M` is a dynamic dimension.
# - B is an initializer with shape [K, N].
#   - This is important for the CoreML EP which only handles the case where B is an initializer.

is_dynamic = False

# M is dynamic
M = "M" if is_dynamic else 3
K = 2
N = 4

graph = helper.make_graph(
    [  # nodes
        helper.make_node("MatMul", ["A", "B"], ["Y"], "MatMul"),
    ],
    "MatMulWith{}InputShape".format("Dynamic" if is_dynamic else "Static"),  # name
    [  # inputs
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N]),
    ],
    [  # initializers
        helper.make_tensor("B", TensorProto.FLOAT, [K, N], [float(i) for i in range(K * N)]),
    ],
)

opset_imports = [helper.make_operatorsetid("", 19)]
model = helper.make_model(graph, opset_imports=opset_imports)
onnx.save(model, str(Path(__file__).parent / "matmul_with_{}_input_shape.onnx".format("dynamic" if is_dynamic else "static")))
