# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy
import onnx
import torch
from io_binding_helper import TypeHelper
from models.t5.past_helper import PastKeyValuesHelper
from onnx_model import OnnxModel
from torch_onnx_export_helper import torch_onnx_export
from transformers import WhisperConfig, file_utils

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


class WhisperDecoderInitOpenai(torch.nn.Module):
    """WhisperDecoderInit for Openai."""
    def __init__(
        self,
        model: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        super().__init__()
        self.whisper_model = model
        self.whisper_decoder = decoder
        self.kv_cache = {}

    @torch.no_grad()
    def forward(
        self,
        tokens,
        audio_features,
        past=None,
    ):

        # Create a kv_cache for past_values
        past_kv_cache = dict()
        if past is not None:
            # Convert past values from 4D to 3D
            past = [torch.transpose(val, 1, 2) for val in past]
            past = [val.reshape(val.shape[:2] + (-1, )) for val in past]
            half_idx = len(past) // 2
            for idx, block in enumerate(self.whisper_decoder.blocks):
                past_kv_cache[block.attn.key] = past[2 * idx]
                past_kv_cache[block.attn.value] = past[2 * idx + 1]
                past_kv_cache[block.cross_attn.key] = past[2 * idx + half_idx]
                past_kv_cache[block.cross_attn.value] = past[2 * idx + half_idx + 1]

        if not self.kv_cache:
            self.kv_cache, _ = self.whisper_model.install_kv_cache_hooks()

        logits = self.whisper_decoder(tokens, audio_features, kv_cache=past_kv_cache)

        # Add concat node for past values
        if past is not None:
            for idx, block in enumerate(self.whisper_decoder.blocks):
                self.kv_cache[block.attn.key] = torch.cat([past_kv_cache[block.attn.key], self.kv_cache[block.attn.key]], dim=1).detach()
                self.kv_cache[block.attn.value] = torch.cat([past_kv_cache[block.attn.value], self.kv_cache[block.attn.value]], dim=1).detach()

        present_self, present_cross = [], []
        # Group self and cross values
        for idx, block in enumerate(self.whisper_decoder.blocks):
            present_self.append(self.kv_cache[block.attn.key])
            present_self.append(self.kv_cache[block.attn.value])
            if past is None:
                present_cross.append(self.kv_cache[block.cross_attn.key])
                present_cross.append(self.kv_cache[block.cross_attn.value])

        present_self = present_self + present_cross
        # Add reshape and transpose ops to convert from 3D to 4D
        present_self = [present_val.reshape(
                present_val.shape[:2] + (-1, 64)
            ).transpose(1, 2) for present_val in present_self]
        return logits, present_self
