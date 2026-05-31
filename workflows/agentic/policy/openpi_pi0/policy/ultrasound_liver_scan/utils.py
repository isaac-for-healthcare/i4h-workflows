# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""openpi DataConfig adapter for the LeRobot ultrasound dataset.

Lifted from ``workflows/robotic_ultrasound/scripts/policy/pi0/utils.py``.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

import einops
import numpy as np
import openpi.transforms as _transforms
from openpi import transforms
from openpi.models import model as _model
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class _RemoveStrings(transforms.DataTransformFn):
    """Drop string-typed fields before stats accumulation. Must be module-level
    (not a local class) so the multiprocessing DataLoader workers can pickle it."""

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def compute_normalization_stats(config, max_frames: int | None = None, batch_size: int = 1) -> None:
    """Compute & persist openpi normalization statistics for the configured dataset.

    Mirrors ``workflows/robotic_ultrasound/scripts/policy/pi0/utils.py:compute_normalization_stats``.
    """
    import jax
    import tqdm
    from openpi.shared import normalize
    from openpi.training import data_loader as _data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _RemoveStrings(),
        ],
    )

    num_frames = len(dataset)
    shuffle = False
    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    # Force single-device sharding: TorchDataLoader's default data-parallel
    # sharding across all jax.devices() requires the batch to divide by the
    # device count, which fails for the batch_size=1 norm-stats sweep on a
    # multi-GPU host.
    single_device_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=single_device_sharding,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            if key in batch:
                values = np.asarray(batch[key][0])
                stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats[key].get_statistics() for key in keys}
    output_path = config.assets_dirs / data_config.repo_id
    os.makedirs(output_path, exist_ok=True)
    normalize.save(output_path, norm_stats)
    print(f"[ultrasound.train] wrote norm stats to {output_path}")


@dataclasses.dataclass(frozen=True)
class Inputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}


@dataclasses.dataclass(frozen=True)
class LeRobotDataConfig(DataConfigFactory):
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Agentic LeRobot converter writes v2-style keys: observation.images.<cam>,
        # observation.state, action. RepackTransform maps NEW (left) <- OLD (right).
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.room",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[Outputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
