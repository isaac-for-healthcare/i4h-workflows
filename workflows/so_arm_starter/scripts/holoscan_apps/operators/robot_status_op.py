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

import logging
import time

from holoscan.conditions import PeriodicCondition
from holoscan.core import Operator, OperatorSpec
from lerobot.common.robots import make_robot_from_config

logger = logging.getLogger(__name__)


class RobotStatusOp(Operator):
    """Robot status operator - gets observations and sends to GR00T"""

    def __init__(self, fragment, robot_config=None, **kwargs):
        # Set up periodic condition
        periodic_condition = PeriodicCondition(fragment, recess_period=100, name="robot_status")
        super().__init__(fragment, periodic_condition, **kwargs)

        self.robot_config = robot_config
        self.robot = None
        self.camera_keys = []
        self.robot_state_keys = []
        self.cycle_count = 0
        self.running = True
        self.action_in_progress = False  # Track if action is being executed
        self.last_action_time = 0

    def setup(self, spec: OperatorSpec):
        """Setup operator ports"""
        spec.output("robot_status")  # Send status to GR00T

    def start(self):
        """Initialize robot connection"""
        logger.info("=== Initializing Robot Status Operator ===")
        try:
            if self.robot_config:
                logger.info(f"Creating robot from config: {self.robot_config.type}")
                self.robot = make_robot_from_config(self.robot_config)

                logger.info("Connecting to robot...")
                self.robot.connect()

                self.camera_keys = list(self.robot_config.cameras.keys())
                self.robot_state_keys = list(self.robot._motors_ft.keys())

                logger.info(f"✅ Robot connected successfully: {self.robot_config.type}")
            else:
                raise ValueError("No robot config provided")
        except Exception as e:
            logger.error(f"Failed to initialize robot: {e}")
            raise

    def compute(self, op_input, op_output, context):
        """Get robot status and send to GR00T, execute received actions"""
        if not self.running or not self.robot:
            return

        try:
            self.cycle_count += 1
            current_time = time.time()

            # Check if action is still in progress
            if self.action_in_progress:
                if current_time - self.last_action_time > 2.0:  # 2 second timeout
                    logger.warning("Action execution timeout, continuing...")
                    self.action_in_progress = False
                else:
                    # Skip this cycle, action still in progress
                    return

            # Get robot observation
            observation_dict = self.robot.get_observation()

            status_data = {
                "observation": observation_dict,
                "camera_keys": self.camera_keys,
                "robot_state_keys": self.robot_state_keys,
                "cycle_id": self.cycle_count,
                "timestamp": current_time,
            }

            op_output.emit(status_data, "robot_status")

        except Exception as e:
            logger.error(f"Error in robot status cycle {self.cycle_count}: {e}")

    def set_action_in_progress(self, in_progress=True):
        """Set action execution status"""
        self.action_in_progress = in_progress
        if in_progress:
            self.last_action_time = time.time()

    def _execute_actions(self, actions):
        """Execute actions received from GR00T"""
        if not actions:
            return

        for action_dict in actions:
            try:
                self.robot.send_action(action_dict)
                time.sleep(0.02)
            except Exception as e:
                logger.error(f"Error executing action: {e}")
