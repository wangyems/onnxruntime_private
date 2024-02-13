# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import copy
import logging
import os

import torch
from benchmark_helper import Precision, create_onnxruntime_session, prepare_environment, setup_logger
from whisper_chain import chain_model
from whisper_helper import PRETRAINED_WHISPER_MODELS, WhisperHelper

from onnxruntime import quantization

logger = logging.getLogger("")

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    pretrained_models = PRETRAINED_WHISPER_MODELS
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        required=False,
        default=PRETRAINED_WHISPER_MODELS[0],
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(pretrained_models),
    )

    parser.add_argument(
        "--model_impl",
        required=False,
        default="hf",
        choices=["hf", "openai"],
        type=str,
        help="Select implementation for export of encoder and decoder subgraphs",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=os.path.join(".", "onnx_models"),
        help="Output directory",
    )

    parser.add_argument(
        "-o",
        "--optimize_onnx",
        required=False,
        action="store_true",
        help="Use optimizer.py to optimize onnx model",
    )
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument("--use_gpu", required=False, action="store_true", help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=[Precision.FLOAT32, Precision.FLOAT16, Precision.INT8],
        help="Precision of model to run. fp32 for full precision, fp16 for half precision, int8 for quantization",
    )

    parser.add_argument("--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("-e", "--use_external_data_format", required=False, action="store_true")
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument(
        "-s",
        "--use_decoder_start_token",
        required=False,
        action="store_true",
        help="Use config.decoder_start_token_id. Otherwise, add an extra graph input to \
              the encoder-decoder-init subgraph for decoder_input_ids.",
    )
    parser.set_defaults(use_decoder_start_token=False)

    parser.add_argument(
        "-f",
        "--use_forced_decoder_ids",
        required=False,
        action="store_true",
        help="Use decoder_input_ids as an extra graph input to the beam search op",
    )
    parser.set_defaults(use_forced_decoder_ids=False)

    parser.add_argument(
        "-l",
        "--use_logits_processor",
        required=False,
        action="store_true",
        help="Use logits_processor as an extra graph input to enable specific logits processing",
    )
    parser.set_defaults(use_specific_logits_processor=False)

    parser.add_argument(
        "-v",
        "--use_vocab_mask",
        required=False,
        action="store_true",
        help="Use vocab_mask as an extra graph input to enable specific logits processing",
    )
    parser.set_defaults(use_vocab_mask=False)

    parser.add_argument(
        "-u",
        "--use_prefix_vocab_mask",
        required=False,
        action="store_true",
        help="Use prefix_vocab_mask as an extra graph input to enable specific logits processing",
    )
    parser.set_defaults(use_prefix_vocab_mask=False)

    parser.add_argument(
        "-w",
        "--overwrite",
        required=False,
        action="store_true",
        help="overwrite existing ONNX model",
    )
    parser.set_defaults(overwrite=False)

    parser.add_argument(
        "--disable_auto_mixed_precision",
        required=False,
        action="store_true",
        help="use pure fp16 instead of mixed precision",
    )
    parser.set_defaults(disable_auto_mixed_precision=False)

    parser.add_argument(
        "--separate_encoder_and_decoder_init",
        required=False,
        action="store_true",
        help="Do not merge encode and decoder init. Output 3 instead of 2 onnx models.",
    )
    parser.set_defaults(separate_encoder_and_decoder_init=False)

    parser.add_argument(
        "--use_int64_inputs",
        required=False,
        action="store_true",
        help="Use int64 instead of int32 for input_ids, position_ids and attention_mask.",
    )
    parser.set_defaults(use_int64_inputs=False)

    parser.add_argument(
        "--chain_model",
        required=False,
        action="store_true",
        help="Produce beam search model with chained encdecinit and decoder.",
    )
    parser.set_defaults(chain_model=True)

    parser.add_argument(
        "--use_whisper_beamsearch",
        required=False,
        action="store_true",
        help="When chain_model, using WhisperBeamSearch operator rather than BeamSearch operator. \
              It will be set to true when collect_cross_qk, extra_decoding_ids or output_no_speech_probs is set.",
    )
    parser.set_defaults(use_whisper_beamsearch=False)

    parser.add_argument(
        "--extra_decoding_ids",
        required=False,
        action="store_true",
        help="Need extra starting decoding ids for some feature like cross qk. Default if false.",
    )
    parser.set_defaults(extra_decoding_ids=False)

    parser.add_argument(
        "--collect_cross_qk",
        required=False,
        action="store_true",
        help="Beam search model collect stacked cross QK.",
    )
    parser.set_defaults(collect_cross_qk=False)

    parser.add_argument(
        "--output_cross_qk",
        required=False,
        action="store_true",
        help="Beam search model output collected qk as output. Also hint collect_cross_qk",
    )
    parser.set_defaults(output_cross_qk=False)

    parser.add_argument(
        "--no_speech_token_id",
        default=50362,
        type=int,
        help="specify no_speech_token_id. Default is 50362. if >= 0, will be add into beam search attr. \
              Note that default value maybe different between the multilingual and English-only models.",
    )

    parser.add_argument(
        "--output_no_speech_probs",
        required=False,
        action="store_true",
        help="Beam search model output no speech probs which is computed from the encoder/context-decoder graph.",
    )
    parser.set_defaults(output_no_speech_probs=False)

    parser.add_argument(
        "--output_scores",
        required=False,
        action="store_true",
        help="Beam search model output scores over vocab per generated token.",
    )
    parser.set_defaults(output_scores=False)

    parser.add_argument(
        "--output_sequence_scores",
        required=False,
        action="store_true",
        help="Beam search model output scores for each generated sequence.",
    )
    parser.set_defaults(output_sequence_scores=False)

    parser.add_argument(
        "--cross_qk_onnx_model",
        required=False,
        type=str,
        default=None,
        help="the model which consume cross_qk.",
    )

    parser.add_argument(
        "--beam_output_model",
        type=str,
        default="whisper_beamsearch.onnx",
        help="default name is whisper_beamsearch.onnx.",
    )

    parser.add_argument(
        "--quantize_embedding_layer",
        required=False,
        action="store_true",
        help="Quantize MatMul, GEMM, and Gather.",
    )
    parser.set_defaults(quantize_embedding_layer=False)

    parser.add_argument(
        "--quantize_per_channel",
        required=False,
        action="store_true",
        help="Quantize weights per each channel.",
    )
    parser.set_defaults(quantize_per_channel=False)

    parser.add_argument(
        "--quantize_reduce_range",
        required=False,
        action="store_true",
        help="Quantize weights with 7 bits.",
    )
    parser.set_defaults(quantize_reduce_range=False)

    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="default to 0")

    parser.add_argument(
        "--state_dict_path",
        type=str,
        default="",
        help="filepath to load pre-trained model with custom state dictionary (e.g. pytorch_model.bin)",
    )

    parser.add_argument(
        "-r",
        "--provider",
        required=False,
        type=str,
        default="cpu",
        choices=list(PROVIDERS.keys()),
        help="Provider to benchmark. Default is CPUExecutionProvider.",
    )

    args = parser.parse_args(argv)
    args.collect_cross_qk = args.collect_cross_qk or args.output_cross_qk

    return args


def export_onnx_models(
    model_name_or_path,
    model_impl,
    cache_dir,
    output_dir,
    use_gpu,
    use_external_data_format,
    optimize_onnx,
    precision,
    verbose,
    use_decoder_start_token: bool = False,
    merge_encoder_and_decoder_init: bool = True,
    overwrite: bool = False,
    disable_auto_mixed_precision: bool = False,
    use_int32_inputs: bool = True,
    quantize_embedding_layer: bool = False,
    quantize_per_channel: bool = False,
    quantize_reduce_range: bool = False,
    state_dict_path: str = "",
    provider: str = "cpu",
):
    device = torch.device("cuda:0" if use_gpu else "cpu")

    models = WhisperHelper.load_model(
        model_name_or_path, model_impl, cache_dir, device, merge_encoder_and_decoder_init, state_dict_path
    )
    config = models["decoder"].config

    if (not use_external_data_format) and (config.num_hidden_layers > 24):
        logger.info("Try use_external_data_format when model size > 2GB")

    output_paths = []
    for name, model in models.items():
        print(f"========> Handling {name} model......")
        model.to(device)
        filename_suffix = "_" + name

        onnx_path = WhisperHelper.get_onnx_path(
            output_dir,
            model_name_or_path,
            suffix=filename_suffix,
            new_folder=False,
        )

        if overwrite or not os.path.exists(onnx_path):
            logger.info(f"Exporting ONNX model to {onnx_path}")
            # We have to clone model before exporting onnx, otherwise verify_onnx will report large difference.
            device_to_export = torch.device("cpu")
            cloned_model = copy.deepcopy(model).to(device_to_export)
            WhisperHelper.export_onnx(
                cloned_model,
                device_to_export,
                onnx_path,
                verbose,
                use_external_data_format,
                use_decoder_input_ids=not use_decoder_start_token,
                use_int32_inputs=use_int32_inputs,
            )
        else:
            logger.info(f"Skip exporting: existed ONNX model {onnx_path}")

        # Optimize ONNX graph. Note that we have not implemented graph optimization for Whisper yet.
        if optimize_onnx or precision != Precision.FLOAT32:
            output_path = WhisperHelper.get_onnx_path(
                output_dir,
                model_name_or_path,
                suffix=filename_suffix + "_" + str(precision),
                new_folder=False,
            )

            if overwrite or not os.path.exists(output_path):
                if optimize_onnx:
                    logger.info(f"Optimizing model to {output_path}")
                    WhisperHelper.optimize_onnx(
                        onnx_path,
                        output_path,
                        precision == Precision.FLOAT16,
                        config.encoder_attention_heads,
                        config.d_model,
                        use_external_data_format,
                        auto_mixed_precision=not disable_auto_mixed_precision,
                        use_gpu=use_gpu,
                        provider=provider,
                    )
                    onnx_path = output_path

                if precision == Precision.INT8:
                    quantization.quantize_dynamic(
                        onnx_path,
                        output_path,
                        op_types_to_quantize=["MatMul", "Gemm", "Gather"]
                        if quantize_embedding_layer
                        else ["MatMul", "Gemm"],
                        use_external_data_format=use_external_data_format,
                        per_channel=quantize_per_channel,
                        reduce_range=quantize_reduce_range,
                        extra_options={"MatMulConstBOnly": True},
                    )
            else:
                logger.info(f"Skip optimizing: existed ONNX model {onnx_path}")
        else:
            output_path = onnx_path

        ort_session = create_onnxruntime_session(
            output_path,
            use_gpu=use_gpu,
            provider=provider,
        )
        assert ort_session is not None

        output_paths.append(output_path)

    return output_paths


def main(argv=None):
    args = parse_arguments(argv)

    setup_logger(args.verbose)

    logger.info(f"Arguments:{args}")

    cache_dir = args.cache_dir
    output_dir = args.output if not args.output.endswith(".onnx") else os.path.dirname(args.output)
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 requires --use_gpu"

    if args.optimize_onnx:
        logger.warning("Applying graph optimization for Whisper...")

    output_paths = export_onnx_models(
        args.model_name_or_path,
        args.model_impl,
        cache_dir,
        output_dir,
        args.use_gpu,
        args.use_external_data_format,
        args.optimize_onnx,
        args.precision,
        args.verbose,
        args.use_decoder_start_token,
        not args.separate_encoder_and_decoder_init,
        args.overwrite,
        args.disable_auto_mixed_precision,
        not args.use_int64_inputs,
        args.quantize_embedding_layer,
        args.quantize_per_channel,
        args.quantize_reduce_range,
        args.state_dict_path,
        args.provider,
    )

    max_diff = 0
    if args.chain_model:
        logger.info("Chaining model ... :")
        args.beam_model_output_dir = WhisperHelper.get_onnx_path(
            output_dir,
            args.model_name_or_path,
            suffix="_beamsearch",
            new_folder=False,
        )
        for path in output_paths:
            if "encoder_decoder" in path:
                args.encoder_path = path
            elif "decoder" in path:
                args.decoder_path = path
        chain_model(args)
        output_paths.append(args.beam_model_output_dir)

        # Check chained model
        ort_session = create_onnxruntime_session(
            args.beam_model_output_dir,
            use_gpu=args.use_gpu,
            provider=args.provider,
        )
        device = torch.device("cuda:0" if args.use_gpu else "cpu")

        # Wrap parity check in try-except to allow export to continue in case this produces an error
        try:
            with torch.no_grad():
                max_diff = WhisperHelper.verify_onnx(args.model_name_or_path, cache_dir, ort_session, device)
            if max_diff > 1e-4:
                logger.warning("PyTorch and ONNX Runtime results are NOT close")
            else:
                logger.info("PyTorch and ONNX Runtime results are close")
        except Exception as e:
            logger.warning(
                f"An error occurred while trying to verify parity between PyTorch and ONNX Runtime: {e}", exc_info=True
            )

        # Remove extra ONNX models saved in output directory
        for fle in os.listdir(output_dir):
            if "_beamsearch" not in fle:
                os.remove(os.path.join(output_dir, fle))
        output_paths = [args.beam_model_output_dir]

    logger.info(f"Done! Outputs: {output_paths}")
    return max_diff


if __name__ == "__main__":
    main()
