# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch


class TensorRTDiTWrapper:
    """Wrapper for TensorRT DiT engine."""

    def __init__(self, engine_path: str, device: int = 0):
        import tensorrt as trt

        self.device = device
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(device)
            logging.info("CUDA initialized via PyTorch: device %s", device)
        else:
            raise RuntimeError("CUDA not available for TensorRT")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        self.context = self.engine.create_execution_context()

    def __call__(self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None):
        sa_embs = sa_embs.to(f"cuda:{self.device}").contiguous()
        vl_embs = vl_embs.to(f"cuda:{self.device}").contiguous()
        timestep = timestep.to(f"cuda:{self.device}").contiguous()
        if image_mask is not None:
            image_mask = image_mask.to(f"cuda:{self.device}").contiguous()
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(f"cuda:{self.device}").contiguous()

        self.context.set_input_shape("sa_embs", sa_embs.shape)
        self.context.set_input_shape("vl_embs", vl_embs.shape)
        self.context.set_input_shape("timestep", timestep.shape)
        if image_mask is not None:
            self.context.set_input_shape("image_mask", image_mask.shape)
        if backbone_attention_mask is not None:
            self.context.set_input_shape("backbone_attention_mask", backbone_attention_mask.shape)

        self.context.set_tensor_address("sa_embs", sa_embs.data_ptr())
        self.context.set_tensor_address("vl_embs", vl_embs.data_ptr())
        self.context.set_tensor_address("timestep", timestep.data_ptr())
        if image_mask is not None:
            self.context.set_tensor_address("image_mask", image_mask.data_ptr())
        if backbone_attention_mask is not None:
            self.context.set_tensor_address("backbone_attention_mask", backbone_attention_mask.data_ptr())

        output = torch.empty(
            tuple(self.context.get_tensor_shape("output")), dtype=torch.bfloat16, device=f"cuda:{self.device}"
        )
        self.context.set_tensor_address("output", output.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        return output


def replace_dit_with_tensorrt(policy: Any, trt_engine_path: str, device: int = 0):
    trt_dit = TensorRTDiTWrapper(trt_engine_path, device=device)

    def trt_forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        if return_all_hidden_states:
            raise RuntimeError("TensorRT only returns the final output. Check inference config")
        return trt_dit(
            sa_embs=hidden_states,
            vl_embs=encoder_hidden_states,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )

    policy.model.action_head.model.forward = trt_forward
    logging.info("[TRT] DiT replaced with TensorRT engine")
