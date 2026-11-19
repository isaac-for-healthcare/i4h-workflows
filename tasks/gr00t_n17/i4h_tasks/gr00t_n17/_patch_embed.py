# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the Qwen3-VL vision patch embedding as the matrix multiply it is.

GR00T N1.7's backbone embeds image patches with an ``nn.Conv3d`` whose kernel
equals its stride, so each output is one dot product of a flattened patch with
a flattened filter: an ``(N, C*T*P*P) @ (C*T*P*P, D)`` matmul. In bf16, the
pinned torch 2.9.0+cu130 gives this convolution no fast kernel. A batch of
8,192 patches takes 26-47 s on A40, RTX PRO 6000 Blackwell and GB300, against
about 1 ms in fp32 and 0.2 ms for the same product as ``F.linear``. The
vision tower is frozen during fine-tuning but still runs forward on every
step, so the catheter fine-tune ran at ~40 s/step (NVBug 6932348).

The weight stays in ``proj`` as a Conv3d parameter and is only viewed as a
matrix, so checkpoints, ``tune_visual`` and gradient flow are unchanged.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("i4h_tasks.gr00t_n17")


def linear_patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """``Qwen3VLVisionPatchEmbed.forward`` computed as ``F.linear``.

    Equal to the convolution because the input is already one flattened patch
    per row in the filter's ``(C, T, P, P)`` order.
    """
    weight = self.proj.weight
    patches = hidden_states.reshape(-1, weight[0].numel()).to(dtype=weight.dtype)
    return F.linear(patches, weight.reshape(weight.shape[0], -1), self.proj.bias)


def install_linear_patch_embed() -> bool:
    """Patch ``Qwen3VLVisionPatchEmbed`` in this process; ``False`` if transformers has none."""
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except ImportError:
        return False
    patch_embed = modeling_qwen3_vl.Qwen3VLVisionPatchEmbed
    if patch_embed.forward is not linear_patch_embed_forward:
        patch_embed.forward = linear_patch_embed_forward
        logger.info("Qwen3-VL vision patch embedding runs as a linear layer")
    return True
