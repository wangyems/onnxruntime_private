from torch.onnx import register_custom_op_symbolic


_onnx_opset_version = 1


def register_custom_op():
    """
    This function registers symbolic functions for
    custom ops that are implemented as part of ONNX Runtime
    """

    # Symbolic definition
    def inverse(g, self):
        return g.op("com.microsoft::Inverse", self)

    def gelu(g, self):
        return g.op("com.microsoft::Gelu", self)

    def triu(g, self, diagonal):
        return g.op("com.microsoft::Triu", self, diagonal)

    # Op Registration
    register_custom_op_symbolic('::inverse', inverse, _onnx_opset_version)
    register_custom_op_symbolic('::gelu', gelu, _onnx_opset_version)
    register_custom_op_symbolic('::triu', triu, _onnx_opset_version)
