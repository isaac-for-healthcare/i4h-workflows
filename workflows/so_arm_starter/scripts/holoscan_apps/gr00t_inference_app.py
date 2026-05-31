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

import argparse
import importlib.util
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from holoscan.core import Application
from holoscan_apps.operators import GR00TInferenceOp, RobotStatusOp
from lerobot.common.cameras.opencv import OpenCVCameraConfig
from lerobot.common.robots.so101_follower import SO101FollowerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _N17PolicyAdapter:
    """Wraps N1.7 ``Gr00tPolicy`` so that ``GR00TInferenceOp`` (which builds
    flat-dict observations) can drive it transparently.

    * Accepts flat obs  (``video.*``, ``state.*``, ``annotation.*``)
    * Returns flat action (``action.*``)
    """

    def __init__(self, policy: Gr00tPolicy):
        self._policy = policy
        self._language_key = policy.language_key

    def get_action(self, flat_obs: dict) -> dict:
        nested: dict = {"video": {}, "state": {}, "language": {}}
        for k, v in flat_obs.items():
            if k.startswith("video."):
                arr = np.asarray(v)
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                nested["video"][k[len("video.") :]] = arr
            elif k.startswith("state."):
                nested["state"][k[len("state.") :]] = np.asarray(v, dtype=np.float32)
            elif k.startswith("annotation.") or k == "task_description":
                if isinstance(v, list) and v and isinstance(v[0], str):
                    nested["language"][self._language_key] = [[s] for s in v]
                elif isinstance(v, str):
                    nested["language"][self._language_key] = [[v]]
                else:
                    nested["language"][self._language_key] = v

        action_dict, _info = self._policy.get_action(nested)
        return {f"action.{key}": action_dict[key] for key in action_dict}


def setup_tensorrt_engines(policy, trt_engine_path: str, mode: str = "n17_full_pipeline") -> None:
    """Load TensorRT engines and monkey-patch the policy's forward passes."""
    # holoscan_apps -> scripts -> so_arm_starter -> workflows -> i4h-workflows-internal
    groot_root = Path(__file__).resolve().parents[4] / "third_party" / "Isaac-GR00T"
    trt_fwd = groot_root / "scripts" / "deployment" / "trt_model_forward.py"

    spec = importlib.util.spec_from_file_location("trt_model_forward", trt_fwd)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find {trt_fwd}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    logger.info(f"Setting up TensorRT engines from {trt_engine_path} (mode={mode}) ...")
    mod.setup_tensorrt_engines(policy, trt_engine_path, mode=mode)
    logger.info("TensorRT engines loaded successfully.")


class GR00TCyclicApplication(Application):
    """Cyclic Holoscan application for GR00T-controlled SOAR robot"""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from YAML file"""
        if self.config_path:
            if not Path(self.config_path).exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("No configuration file specified")

    def compose(self):
        """Compose the cyclic application graph"""
        robot_config = self.create_robot_config()
        gr00t_policy = self.create_gr00t_policy()

        robot_status_op = RobotStatusOp(self, robot_config=robot_config, name="robot_status")

        gr00t_inference_op = GR00TInferenceOp(
            self,
            policy=gr00t_policy,
            language_instruction=self.config["gr00t"]["language_instruction"],
            action_horizon=self.config["gr00t"]["action_horizon"],
            robot_status_op=robot_status_op,
            name="gr00t_inference",
        )

        self.add_flow(robot_status_op, gr00t_inference_op, {("robot_status", "robot_status")})

        logger.info("Application graph composed successfully")

    def create_robot_config(self):
        """Create robot configuration object from config dict"""
        camera_configs = {}
        for name, cam_config in self.config["robot"]["cameras"].items():
            cam_type = cam_config.pop("type", "opencv")
            if cam_type == "opencv":
                camera_configs[name] = OpenCVCameraConfig(**cam_config)
            else:
                logger.warning(f"Unsupported camera type: {cam_type}, using OpenCV")
                camera_configs[name] = OpenCVCameraConfig(**cam_config)

        logger.info(f"Creating SO101FollowerConfig with port: {self.config['robot']['port']}")
        logger.info(f"Camera configs: {list(camera_configs.keys())}")

        return SO101FollowerConfig(
            port=self.config["robot"]["port"], id=self.config["robot"]["id"], cameras=camera_configs
        )

    def create_gr00t_policy(self):
        """Create GR00T N1.7 policy object.

        Supports optional TensorRT acceleration when ``trt_engine_path`` is
        specified in the config.
        """
        model_path = self.config["gr00t"]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model path not found: {model_path}. " f"Please verify your configuration file: {self.config_path}."
            )

        embodiment_tag = self.config["gr00t"].get("embodiment_tag", "new_embodiment")
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(embodiment_tag),
            model_path=model_path,
            device="cuda",
        )
        logger.info("GR00T N1.7 model loaded")

        trt_engine_path = self.config["gr00t"].get("trt_engine_path")
        if trt_engine_path is not None:
            trt_mode = self.config["gr00t"].get("trt_mode", "n17_full_pipeline")
            setup_tensorrt_engines(policy, trt_engine_path, trt_mode)

        return _N17PolicyAdapter(policy)


def main():
    parser = argparse.ArgumentParser(description="GR00T N1.7 Holoscan Application")
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="Path to configuration YAML file",
        default=f"{os.path.dirname(os.path.abspath(__file__))}/soarm_robot_config.yaml",
    )

    args = parser.parse_args()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Creating cyclic application instance...")
    app = GR00TCyclicApplication(config_path=args.config)

    logger.info(f"Final configuration: {app.config}")

    try:
        logger.info("Starting cyclic application...")
        app.run()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
