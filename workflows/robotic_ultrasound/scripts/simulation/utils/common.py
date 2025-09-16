# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
from huggingface_hub import snapshot_download
from simulation.configs.config import Config


def colorize_depth(depth_data: np.ndarray, near=1.0, far=50.0) -> np.ndarray:
    """Colorize depth data for visualization.

    Args:
        depth_data: The depth data to colorize, shape: (H, W, 1).
        near: The near clipping distance.
        far: The far clipping distance.

    Refer to
    https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/programmatic_visualization.html
    #helper-visualization-functions
    """

    depth_data = np.clip(depth_data, near, far)
    depth_data = (np.log(depth_data) - np.log(near)) / (np.log(far) - np.log(near))
    depth_data = 1.0 - depth_data
    depth_data_uint8 = (depth_data * 255).astype(np.uint8)

    return depth_data_uint8


def list_exp_configs(configs_path: str | None = None):
    """List all experiment configurations in the given path.

    Args:
        configs_path: The path to the configurations directory.
    """
    if configs_path is None:
        configs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs"))
    names = [c[:-3] for c in os.listdir(configs_path) if c.endswith(".py")]
    return [n for n in names if n not in ["__init__", "config"]]


def get_exp_config(name: str, configs_path: str | None = None) -> Config:
    """Get the experiment configuration for the given name.

    Args:
        name: The name of the experiment configuration.
        configs_path: The path to the configurations directory.
    """
    if configs_path is None:
        configs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs"))

    module_path = os.path.join(configs_path, f"{name}.py")
    spec = spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise Exception(f"Could not load spec for {name} from {configs_path}")

    module = module_from_spec(spec)
    if module is None:
        raise Exception(f"Could not load module for {name} from {configs_path}")

    spec.loader.exec_module(module)
    return module.config


def resolve_checkpoint_path(ckpt_path: str) -> str:
    """Resolve checkpoint path - handle both local paths and HuggingFace repo IDs.

    Args:
        ckpt_path: Either a local file/directory path or a HuggingFace repository ID

    Returns:
        str: Local path to the checkpoint (either the original local path or
             the downloaded HF repo path)

    Examples:
        >>> resolve_checkpoint_path("/path/to/local/checkpoint")
        "/path/to/local/checkpoint"

        >>> resolve_checkpoint_path("nvidia/Liver_Scan_Pi0_Cosmos_Rel")
        "/home/user/.cache/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel/..."
    """
    # Check if it's a local path that exists
    if os.path.exists(ckpt_path):
        return ckpt_path

    # Check if it's an absolute path that might not exist yet (but we should respect it)
    if os.path.isabs(ckpt_path):
        return ckpt_path

    # Check if it looks like a HuggingFace repo ID (contains forward slash)
    if "/" in ckpt_path and not ckpt_path.startswith("./") and not ckpt_path.startswith("../"):
        try:
            print(f"Downloading checkpoint from HuggingFace Hub: {ckpt_path}")
            return snapshot_download(repo_id=ckpt_path)
        except Exception as e:
            # If HF download fails, treat it as a local path anyway
            print(f"Warning: Failed to download from HuggingFace Hub: {e}")
            print(f"Treating '{ckpt_path}' as local path")
            return ckpt_path

    # Default: treat as local path
    return ckpt_path
