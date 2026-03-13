# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility helpers for tasks."""

import logging
from typing import Any

import torch


def create_success_hold_wrapper(original_success_term, hold_steps: int, num_envs: int, verbose: bool = False):
    """Wrap a success termination term so success must be held for multiple consecutive steps."""
    from isaaclab.managers import TerminationTermCfg

    state = {
        "success_counters": [0] * num_envs,
        "last_print_step": [0] * num_envs,
        "success_achieved": [False] * num_envs,
        "last_episode_step": [0] * num_envs,
    }

    def success_hold_checker(env, **kwargs):
        for env_idx in range(num_envs):
            current_step = env.episode_length_buf[env_idx].item() if hasattr(env, "episode_length_buf") else 0
            if current_step < state["last_episode_step"][env_idx] or current_step == 0:
                state["success_counters"][env_idx] = 0
                state["last_print_step"][env_idx] = 0
                state["success_achieved"][env_idx] = False
            state["last_episode_step"][env_idx] = current_step

        original_results = original_success_term.func(env, **kwargs)
        held_results = torch.zeros_like(original_results, dtype=torch.bool)

        for env_idx in range(len(original_results)):
            if state["success_achieved"][env_idx]:
                held_results[env_idx] = True
                continue

            if original_results[env_idx]:
                state["success_counters"][env_idx] += 1

                if (
                    verbose
                    and (state["success_counters"][env_idx] % 10 == 0)
                    and state["success_counters"][env_idx] < hold_steps
                ):
                    if state["success_counters"][env_idx] != state["last_print_step"][env_idx]:
                        print(
                            f"[Progress] Env {env_idx}: Success held for "
                            f"{state['success_counters'][env_idx]}/{hold_steps} steps"
                        )
                        state["last_print_step"][env_idx] = state["success_counters"][env_idx]

                if state["success_counters"][env_idx] >= hold_steps:
                    held_results[env_idx] = True
                    if not state["success_achieved"][env_idx]:
                        print(f"\n[SUCCESS] Env {env_idx}: Held success for {hold_steps} steps!")
                        state["success_achieved"][env_idx] = True
            else:
                state["success_counters"][env_idx] = 0
                state["last_print_step"][env_idx] = 0

        return held_results

    return TerminationTermCfg(
        func=success_hold_checker,
        params=original_success_term.params.copy() if original_success_term.params else {},
        time_out=getattr(original_success_term, "time_out", False),
    )


###############################################################################
# TensorRT Utilities for GR00T Policy
###############################################################################
class TensorRTDiTWrapper:
    """Wrapper for TensorRT DiT engine."""

    def __init__(self, engine_path: str, device: int = 0):
        import tensorrt as trt

        self.device = device

        # Ensures CUDA driver is properly loaded
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(device)
            logging.info(f"CUDA initialized via PyTorch: device {device}")
        else:
            raise RuntimeError("CUDA not available for TensorRT")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        logging.info(f"TensorRT engine loaded: {engine_path}")

    def __call__(self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None):
        """Forward pass through TensorRT DiT."""
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

        output_shape = self.context.get_tensor_shape("output")
        output = torch.empty(tuple(output_shape), dtype=torch.bfloat16, device=f"cuda:{self.device}")
        self.context.set_tensor_address("output", output.data_ptr())

        success = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not success:
            raise RuntimeError("TensorRT inference failed")

        return output


def replace_dit_with_tensorrt(policy: Any, trt_engine_path: str, device: int = 0):
    """Replace the DiT forward method with TensorRT inference.

    Args:
        policy: Gr00tPolicy instance
        trt_engine_path: Path to TensorRT engine file
        device: CUDA device index
    """
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
        """TensorRT wrapper matching DiT forward signature."""
        output = trt_dit(
            sa_embs=hidden_states,
            vl_embs=encoder_hidden_states,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )

        if return_all_hidden_states:
            raise RuntimeError("TensorRT only returns the final output. Check inference config")
        return output

    policy.model.action_head.model.forward = trt_forward
    logging.info("[TRT] DiT replaced with TensorRT engine")
