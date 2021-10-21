from onnxruntime.capi import _pybind_state as C

import threading
from functools import wraps


def run_once_aten_op_executor(f):
    """
    Decorator to run a function only once.
    :param f: function to be run only once during execution time despite the number of calls
    :return: The original function with the params passed to it if it hasn't already been run before
    """
    @wraps(f)
    def aten_op_executor_wrapper(*args, **kwargs):
        if not aten_op_executor_wrapper.has_run:
            with aten_op_executor_wrapper.lock:
                if not aten_op_executor_wrapper.has_run:
                    aten_op_executor_wrapper.has_run = True
                    return f(*args, **kwargs)

    aten_op_executor_wrapper.lock = threading.Lock()
    aten_op_executor_wrapper.has_run = False
    return aten_op_executor_wrapper


@run_once_aten_op_executor
def _load_aten_op_executor_cpp_extension():
    from onnxruntime.training.ortmodule.torch_cpp_extensions import aten_op_executor
    C.register_aten_op_executor(str(aten_op_executor.is_tensor_argument_address()),
                                str(aten_op_executor.execute_aten_operator_address()))


def load_aten_op_executor_cpp_extension_if_needed(onnx_model):
    for node in onnx_model.graph.node:
        if (node.op_type == 'ATenOp' and node.domain == 'com.microsoft') or node.op_type == 'ReduceSum':
            _load_aten_op_executor_cpp_extension()
            break
