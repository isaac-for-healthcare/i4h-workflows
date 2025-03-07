import dataclasses
import os
import pathlib

import einops
import numpy as np
import tqdm

import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
from openpi import transforms
from openpi.compute_norm_stats import create_dataset
from openpi.models import model as _model
from openpi.shared import normalize
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory


def _parse_image(image) -> np.ndarray:
    """
    Parse the image to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W).

    Args:
        image: The image to parse.

    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def compute_normalization_stats(config, max_frames=None):
    """Compute normalization statistics for the dataset."""
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    # Reduce num_workers to 0 to avoid multiprocessing issues
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=8,  # Changed from 8 to 0
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            if key in batch:
                values = np.asarray(batch[key][0])
                stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats[key].get_statistics() for key in keys}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    normalize.save(output_path, norm_stats)


@dataclasses.dataclass(frozen=True)
class Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
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

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 6 dims.
        return {"actions": np.asarray(data["actions"][:, :6])}


@dataclasses.dataclass(frozen=True)
class LeRobotDataConfig(DataConfigFactory):
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """
        Repack transforms are used to repack the data into the format expected by the model.

        Args:
            assets_dirs: The directory containing the assets.
            model_config: The model configuration.

        """
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[Outputs()],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
