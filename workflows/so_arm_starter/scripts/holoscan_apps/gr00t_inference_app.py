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
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import yaml
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from holoscan.core import Application
from holoscan_apps.operators import GR00TInferenceOp, RobotStatusOp
from lerobot.common.cameras.opencv import OpenCVCameraConfig
from lerobot.common.robots.so101_follower import SO101FollowerConfig
from policy_runner.gr00tn1_5.trt.trt_model_forward import setup_tensorrt_engines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # Create robot status operator - this will connect to robot
        robot_status_op = RobotStatusOp(self, robot_config=robot_config, name="robot_status")

        # Create GR00T inference operator
        gr00t_inference_op = GR00TInferenceOp(
            self,
            policy=gr00t_policy,
            language_instruction=self.config["gr00t"]["language_instruction"],
            action_horizon=self.config["gr00t"]["action_horizon"],
            robot_status_op=robot_status_op,
            name="gr00t_inference",
        )

        self.add_flow(robot_status_op, gr00t_inference_op, {("robot_status", "robot_status")})

        logger.info("✅ Application graph composed successfully")

    def create_robot_config(self):
        """Create robot configuration object from config dict"""
        # Convert camera configs to CameraConfig objects
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

        robot_config = SO101FollowerConfig(
            port=self.config["robot"]["port"], id=self.config["robot"]["id"], cameras=camera_configs
        )

        return robot_config

    def create_gr00t_policy(self):
        """Create GR00T policy object"""
        data_config = DATA_CONFIG_MAP[self.config["gr00t"]["data_config"]]
        data_config.video_keys = self.config["gr00t"]["video_keys"]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        if not os.path.exists(self.config["gr00t"]["model_path"]):
            logger.error(f"Model path not found: {self.config['gr00t']['model_path']}")
            model_path = self.config["gr00t"]["model_path"]
            raise FileNotFoundError(f"Model path not found: {model_path}" f"please configure right path in yaml file")

        if self.config["gr00t"]["trt"] and not os.path.exists(self.config["gr00t"]["trt_engine_path"]):
            logger.error(f"TensorRT engine path not found: {self.config['gr00t']['trt_engine_path']}")
            engine_path = self.config["gr00t"]["trt_engine_path"]
            raise FileNotFoundError(
                f"TensorRT engine path not found: {engine_path}" f"please configure right path in yaml file"
            )

        policy = Gr00tPolicy(
            model_path=self.config["gr00t"]["model_path"],
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag="new_embodiment",
            denoising_steps=4,
        )

        # Set up TensorRT engines
        if self.config["gr00t"]["trt"]:
            setup_tensorrt_engines(policy, self.config["gr00t"]["trt_engine_path"])
            logger.info("✅ TensorRT engines set up successfully")
        else:
            logger.info("✅ PyTorch model is used")

        return policy


def main():
    parser = argparse.ArgumentParser(description="GR00T Holoscan Application - Cyclic Data Flow Version")
    parser.add_argument("--config", required=True, type=str, help="Path to configuration YAML file")

    args = parser.parse_args()

    # Set up global signal handler
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Creating cyclic application instance...")
    app = GR00TCyclicApplication(config_path=args.config)

    logger.info(f"📋 Final configuration: {app.config}")

    try:
        logger.info("🎬 Starting cyclic application...")
        app.run()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
