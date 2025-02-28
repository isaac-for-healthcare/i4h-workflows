import os
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
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
