# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import torch


class TensorRTDiTWrapper:
    def __init__(self, engine_path: str, device: int = 0):
        import tensorrt as trt

        self.device = device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for TensorRT")
        torch.cuda.set_device(device)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        self.context = self.engine.create_execution_context()

    def __call__(self, sa_embs, vl_embs, timestep):
        device = f"cuda:{self.device}"
        sa_embs = sa_embs.to(device).contiguous()
        vl_embs = vl_embs.to(device).contiguous()
        timestep = timestep.to(device).contiguous()
        self.context.set_input_shape("sa_embs", sa_embs.shape)
        self.context.set_input_shape("vl_embs", vl_embs.shape)
        self.context.set_input_shape("timestep", timestep.shape)
        self.context.set_tensor_address("sa_embs", sa_embs.data_ptr())
        self.context.set_tensor_address("vl_embs", vl_embs.data_ptr())
        self.context.set_tensor_address("timestep", timestep.data_ptr())
        output = torch.empty(tuple(self.context.get_tensor_shape("output")), dtype=torch.bfloat16, device=device)
        self.context.set_tensor_address("output", output.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        return output


def replace_dit_with_tensorrt(policy: Any, trt_engine_path: str, device: int = 0) -> None:
    trt_dit = TensorRTDiTWrapper(trt_engine_path, device=device)

    def trt_forward(hidden_states, encoder_hidden_states, timestep, **_kwargs):
        return trt_dit(hidden_states, encoder_hidden_states, timestep)

    policy.model.action_head.model.forward = trt_forward
    logging.info("[TRT] DiT replaced with TensorRT engine")
