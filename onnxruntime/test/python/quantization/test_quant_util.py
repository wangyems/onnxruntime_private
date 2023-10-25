#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from pathlib import Path

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxruntime.quantization.quant_utils import compute_scale_zp, load_model_with_shape_infer, model_has_infer_metadata


class TestQuantUtil(unittest.TestCase):
    def test_compute_scale_zp(self):
        def _compute_scale_zp(rmin, rmax, qmin, qmax, symmetric=False):
            zp, scale = compute_scale_zp(
                numpy.array(rmin, dtype=numpy.float32),
                numpy.array(rmax, dtype=numpy.float32),
                qmin,
                qmax,
                symmetric=symmetric,
            )
            assert isinstance(zp, numpy.ndarray)
            assert isinstance(scale, numpy.ndarray)
            return [float(zp), float(scale)]

        self.assertEqual(_compute_scale_zp(0.0, 0.0, -127, 127, symmetric=True), [0, 1.0])
        self.assertEqual(_compute_scale_zp(1.0, -1.0, -127, 127, symmetric=True), [0, 1.0])
        self.assertEqual(_compute_scale_zp(0.0, 0.0, 0, 255, symmetric=True), [0, 1.0])
        self.assertEqual(_compute_scale_zp(1.0, -1.0, 0, 255, symmetric=True), [0, 1.0])

        self.assertEqual(_compute_scale_zp(-1.0, 2.0, -127, 127, symmetric=True), [0, numpy.float32(2.0 / 127)])
        self.assertEqual(_compute_scale_zp(-1.0, 2.0, -127, 127, symmetric=False), [-42, numpy.float32(3.0 / 254)])

        self.assertEqual(_compute_scale_zp(-1.0, 2.0, 0, 255, symmetric=True), [128, numpy.float32(4.0 / 255)])
        self.assertEqual(_compute_scale_zp(-1.0, 2.0, 0, 255, symmetric=False), [85, numpy.float32(3.0 / 255)])

        tiny_float = numpy.float32(numpy.finfo(numpy.float32).tiny * 0.1)
        self.assertEqual(_compute_scale_zp(-tiny_float, tiny_float, 0, 255, symmetric=True), [0, 1.0])
        self.assertEqual(_compute_scale_zp(-tiny_float, 0.0, 0, 255, symmetric=False), [0, 1.0])

    def test_load_external_model(self):
        input_name = "input"
        output_name = "output"
        add_shape = [1024, 1024]

        initializers = []
        weight_name = "weight"
        weight_data = numpy.random.normal(0, 0.1, add_shape).astype(numpy.float32)
        initializers.append(numpy_helper.from_array(weight_data, name=weight_name))
        add_node = helper.make_node("Add", [input_name, weight_name], [output_name], name="add_node")

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, add_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, add_shape)
        graph_name = "test_load_external_model"
        graph = helper.make_graph(
            [add_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(model_has_infer_metadata(model))
            model_file_path = temp_dir + "/test_load_external_model.onnx"
            onnx.save(model, model_file_path, save_as_external_data=True)
            model_reloaded = load_model_with_shape_infer(Path(model_file_path))
            self.assertTrue(model_has_infer_metadata(model_reloaded))


if __name__ == "__main__":
    unittest.main()
