# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3-VL patch embedding runs as a linear layer (NVBug 6932348).

The pinned torch has no fast bf16 kernel for this Conv3d: one fine-tune step
spent ~40 s in it. Its kernel equals its stride, so it is a matmul, and the
patched forward must give the convolution's result from the same weights.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from i4h_tasks.gr00t_n17 import _patch_embed

ORIGINAL_FORWARD = modeling_qwen3_vl.Qwen3VLVisionPatchEmbed.forward


@pytest.fixture
def patch_embed(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
    monkeypatch.setattr(modeling_qwen3_vl.Qwen3VLVisionPatchEmbed, "forward", ORIGINAL_FORWARD)
    torch.manual_seed(0)
    config = Qwen3VLVisionConfig(patch_size=4, temporal_patch_size=2, in_channels=3, hidden_size=32)
    return modeling_qwen3_vl.Qwen3VLVisionPatchEmbed(config)


def _patches(module: torch.nn.Module, count: int = 64) -> torch.Tensor:
    width = module.in_channels * module.temporal_patch_size * module.patch_size * module.patch_size
    return torch.randn(count, width)


def test_the_linear_forward_equals_the_convolution(patch_embed: torch.nn.Module) -> None:
    patches = _patches(patch_embed)
    expected = patch_embed(patches)

    assert _patch_embed.install_linear_patch_embed()
    actual = patch_embed(patches)

    assert actual.shape == expected.shape == (64, 32)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_the_weight_stays_the_convolution_parameter_and_still_trains(patch_embed: torch.nn.Module) -> None:
    """Checkpoints and --tune-visual see the same Conv3d ``proj`` they always did."""
    _patch_embed.install_linear_patch_embed()

    patch_embed(_patches(patch_embed)).sum().backward()

    assert isinstance(patch_embed.proj, torch.nn.Conv3d)
    assert set(dict(patch_embed.named_parameters())) == {"proj.weight", "proj.bias"}
    assert patch_embed.proj.weight.grad is not None
    assert patch_embed.proj.weight.grad.shape == patch_embed.proj.weight.shape


def test_the_output_keeps_the_weight_dtype(patch_embed: torch.nn.Module) -> None:
    _patch_embed.install_linear_patch_embed()
    patch_embed.to(torch.bfloat16)

    assert patch_embed(_patches(patch_embed)).dtype == torch.bfloat16


def test_installing_twice_is_harmless(patch_embed: torch.nn.Module) -> None:
    assert _patch_embed.install_linear_patch_embed()
    assert _patch_embed.install_linear_patch_embed()
    assert modeling_qwen3_vl.Qwen3VLVisionPatchEmbed.forward is _patch_embed.linear_patch_embed_forward
