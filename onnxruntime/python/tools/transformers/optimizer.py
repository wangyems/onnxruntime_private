#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# Convert Bert ONNX model converted from TensorFlow or exported from PyTorch to use Attention, Gelu,
# SkipLayerNormalization and EmbedLayerNormalization ops to optimize
# performance on NVidia GPU and CPU.
#
# For Bert model exported from PyTorch, OnnxRuntime has bert model optimization support internally.
# You can use the option --use_onnxruntime to check optimizations from OnnxRuntime.
# For Bert model file like name.onnx, optimized model for GPU or CPU from OnnxRuntime will output as
# name_ort_gpu.onnx or name_ort_cpu.onnx in the same directory.
#
# This script is retained for experiment purpose. Useful senarios like the following:
#  (1) Change model from fp32 to fp16 for mixed precision inference in GPU with Tensor Core.
#  (2) Change input data type from int64 to int32.
#  (3) Some model cannot be handled by OnnxRuntime, and you can modify this script to get optimized model.

import logging
import coloredlogs
import onnx
import os
import sys
import argparse
import numpy as np
from typing import Dict
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper, load_model
from onnx_model_bert import BertOnnxModel, BertOptimizationOptions
from onnx_model_bert_tf import BertOnnxModelTF
from onnx_model_bert_keras import BertOnnxModelKeras
from onnx_model_gpt2 import Gpt2OnnxModel
from fusion_options import FusionOptions

logger = logging.getLogger(__name__)

# Map model type to tuple: optimizer class, export tools (pytorch, tf2onnx, keras2onnx), and default opt_level
MODEL_TYPES = {
    "bert": (BertOnnxModel, "pytorch", 1),
    "bert_tf": (BertOnnxModelTF, "tf2onnx", 0),
    "bert_keras": (BertOnnxModelKeras, "keras2onnx", 0),
    "gpt2": (Gpt2OnnxModel, "pytorch", 1),
    "gpt2_tf": (Gpt2OnnxModel, 'tf2onnx', 0)  # might add a class for GPT2OnnxModel for TF later.
}


def optimize_by_onnxruntime(onnx_model_path: str,
                            use_gpu: bool = False,
                            optimized_model_path: str = None,
                            opt_level: int = 99) -> str:
    """
    Use onnxruntime to optimize model.

    Args:
        onnx_model_path (str): the path of input onnx model.
        use_gpu (bool): whether the optimized model is targeted to run in GPU.
        optimized_model_path (str or None): the path of optimized model.
        opt_level (int): graph optimization level.

    Returns:
        optimized_model_path (str): the path of optimized model
    """
    assert opt_level in [1, 2, 99]
    import onnxruntime

    if use_gpu and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        logger.error("There is no gpu for onnxruntime to do optimization.")
        return onnx_model_path

    sess_options = onnxruntime.SessionOptions()
    if opt_level == 1:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == 2:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if optimized_model_path is None:
        path_prefix = onnx_model_path[:-5]  #remove .onnx suffix
        optimized_model_path = "{}_o{}_{}.onnx".format(path_prefix, opt_level, "gpu" if use_gpu else "cpu")

    sess_options.optimized_model_filepath = optimized_model_path

    if not use_gpu:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])
    else:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options)
        assert 'CUDAExecutionProvider' in session.get_providers()  # Make sure there is GPU

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    logger.debug("Save optimized model by onnxruntime to {}".format(optimized_model_path))
    return optimized_model_path


def get_fusion_statistics(optimized_model_path: str) -> Dict[str, int]:
    """
    Get counter of fused operators in optimized model.

    Args:
        optimized_model_path (str): the path of onnx model.

    Returns:
        A dictionary with operator type as key, and count as value
    """
    model = load_model(optimized_model_path, format=None, load_external_data=True)
    optimizer = BertOnnxModel(model, num_heads=12, hidden_size=768)
    return optimizer.get_fused_operator_statistics()


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help="input onnx model path")

    parser.add_argument('--output', required=True, type=str, help="optimized onnx model path")

    parser.add_argument('--model_type',
                        required=False,
                        type=str.lower,
                        default="bert",
                        choices=list(MODEL_TYPES.keys()),
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPES.keys()))

    parser.add_argument(
        '--num_heads',
        required=False,
        type=int,
        default=12,
        help=
        "number of attention heads. 12 for bert-base model and 16 for bert-large. For BERT, set it to 0 to detect automatically."
    )

    parser.add_argument(
        '--hidden_size',
        required=False,
        type=int,
        default=768,
        help=
        "bert model hidden size. 768 for bert-base model and 1024 for bert-large. For BERT, set it to 0 to detect automatically."
    )

    parser.add_argument('--input_int32',
                        required=False,
                        action='store_true',
                        help="Use int32 (instead of int64) tensor as input to avoid unnecessary data cast")
    parser.set_defaults(input_int32=False)

    parser.add_argument(
        '--float16',
        required=False,
        action='store_true',
        help="If your target device is V100 or T4 GPU, use this to convert float32 to float16 for best performance")
    parser.set_defaults(float16=False)

    FusionOptions.add_arguments(parser)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--only_onnxruntime', required=False, action='store_true', help="optimized by onnxruntime only")
    parser.set_defaults(only_onnxruntime=False)

    parser.add_argument('--opt_level',
                        required=False,
                        type=int,
                        choices=[0, 1, 2, 99],
                        default=0,
                        help="onnxruntime optimization level. 0 will disable onnxruntime.")

    parser.add_argument('--use_external_data_format',
                        required=False,
                        action='store_true',
                        help="use external data format")
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()

    return args


def optimize_model(input,
                   model_type='bert',
                   num_heads=0,
                   hidden_size=0,
                   optimization_options=None,
                   opt_level=None,
                   use_gpu=False,
                   only_onnxruntime=False):
    """ Optimize Model by OnnxRuntime and/or python fusion logic.

    In onnxruntime, there are graph optimizations (https://onnxruntime.ai/docs/resources/graph-optimizations.html) avaiable. However, the coverage
    is limited. We also have graph fusions that implemented in python to improve the coverage. These two methods can combined: onnxruntime will run first
    when opt_level > 0, then graph fusions in Python will be applied.

    You can opt to use ONNX Runtime only, and no python fusion logic by specifying only_onnxruntime and a positive opt_level like
        optimize_model(input, opt_level=1, use_gpu=False, only_onnxruntime=True)
    If your model is not exported with constant folding, try opt_level=1 to let onnxruntime do it.

    When opt_level is None, we will choose default optimization level according to model type.

    When opt_level is 0 and only_onnxruntime is False, only python fusion logic is used and onnxruntime is disabled.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need specify these parameters.

    use_gpu shall set properly. When opt_level > 1, optimized graph might contain optimization for GPU only. That makes the optimized model less portable.
    However, if your model is intended for GPU inference only (especially float16 or mixed precision model), you must set use_gpu to be True,
    otherwise the model is not optimized for GPU inference.

    Args:
        input (str): input model path.
        model_type (str): model type - like bert, bert_tf, bert_keras or gpt2.
        num_heads (int): number of attention heads. Default is 0 to allow detect the parameter from graph automatically (for model_type "bert" only).
        hidden_size (int): hidden size. Default is 0 to allow detect the parameter from graph automatically (for model_type "bert" only).
        optimization_options (FusionOptions): optimization options that can use to turn on/off some fusions.
        opt_level (int): onnxruntime graph optimization level (0, 1, 2 or 99) or None. When the level > 0, onnxruntime will be used to optimize model first.
        use_gpu (bool): use gpu or not for onnxruntime.
        only_onnxruntime (bool): only use onnxruntime to optimize model, and no offline fusion logic is used

     Returns:
        object of an optimizer class.
    """
    assert opt_level is None or opt_level in [0, 1, 2, 99]

    (optimizer_class, producer, default_opt_level) = MODEL_TYPES[model_type]

    if opt_level is None:
        opt_level = default_opt_level

    temp_model_path = None
    if opt_level > 1:
        temp_model_path = optimize_by_onnxruntime(input, use_gpu=use_gpu, opt_level=opt_level)
    elif opt_level == 1:
        # basic optimizations (like constant folding and cast elimation) are not specified to exection provider.
        # CPU provider is used here so that there is no extra node for GPU memory copy.
        temp_model_path = optimize_by_onnxruntime(input, use_gpu=False, opt_level=1)

    if only_onnxruntime and not temp_model_path:
        logger.warning("Please specify a positive value for opt_level when only_onnxruntime is True")

    model = load_model(temp_model_path or input, format=None, load_external_data=True)

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f"Model producer not matched: Expect {producer}, Got {model.producer_name} {model.producer_version}. Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = FusionOptions(model_type)

    optimizer = optimizer_class(model, num_heads, hidden_size)

    if not only_onnxruntime:
        optimizer.optimize(optimization_options)

    # Remove the temporary model.
    if temp_model_path:
        os.remove(temp_model_path)
        logger.debug("Remove tempoary model: {}".format(temp_model_path))

    optimizer.model.producer_name = "onnxruntime.transformers"
    from onnxruntime import __version__ as onnxruntime_version
    optimizer.model.producer_version = onnxruntime_version

    return optimizer


def _setup_logger(verbose):
    if verbose:
        coloredlogs.install(level='DEBUG', fmt='[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')
    else:
        coloredlogs.install(fmt='%(funcName)20s: %(message)s')


def main():
    args = _parse_arguments()

    _setup_logger(args.verbose)

    if os.path.realpath(args.input) == os.path.realpath(args.output):
        logger.warning(f"Specified the same input and output path. Note that this may overwrite the original model")

    optimization_options = FusionOptions.parse(args)

    optimizer = optimize_model(args.input,
                               args.model_type,
                               args.num_heads,
                               args.hidden_size,
                               opt_level=args.opt_level,
                               optimization_options=optimization_options,
                               use_gpu=args.use_gpu,
                               only_onnxruntime=args.only_onnxruntime)

    if args.float16:
        optimizer.convert_model_float32_to_float16()

    if args.input_int32:
        optimizer.change_input_to_int32()

    optimizer.save_model_to_file(args.output, args.use_external_data_format)

    if optimizer.is_fully_optimized():
        logger.info("The model has been fully optimized.")
    else:
        logger.info("The model has been optimized.")


if __name__ == "__main__":
    main()
