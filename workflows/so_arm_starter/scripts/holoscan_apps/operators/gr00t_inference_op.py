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

import numpy as np
from holoscan.core import Operator, OperatorSpec

logger = logging.getLogger(__name__)


class GR00TInferenceOp(Operator):
    """GR00T inference operator - receives status and executes actions"""

    def __init__(
        self, fragment, policy=None, language_instruction="", action_horizon=8, robot_status_op=None, **kwargs
    ):
        super().__init__(fragment, **kwargs)

        self.policy = policy
        self.language_instruction = language_instruction
        self.action_horizon = action_horizon
        self.modality_keys = ["single_arm", "gripper"]
        self.inference_count = 0
        self.robot_status_op = robot_status_op

    def setup(self, spec: OperatorSpec):
        """Setup operator ports"""
        spec.input("robot_status")

    def compute(self, op_input, op_output, context):
        """Run GR00T inference on robot status"""

        status_msg = op_input.receive("robot_status")
        if status_msg is None or not self.policy:
            return

        try:
            self.inference_count += 1

            observation_dict = status_msg.get("observation")
            camera_keys = status_msg.get("camera_keys", [])
            robot_state_keys = status_msg.get("robot_state_keys", [])
            cycle_id = status_msg.get("cycle_id")

            # Preprocess observation for GR00T
            obs_dict = self._preprocess_observation(observation_dict, camera_keys, robot_state_keys)

            action_chunk = self._run_inference(obs_dict)
            lerobot_actions = self._convert_actions(action_chunk, robot_state_keys)

            actions_to_execute = lerobot_actions[: self.action_horizon]
            self._execute_actions(actions_to_execute)

            # Log every 50 inferences
            if self.inference_count % 50 == 0:
                logger.info(f"GR00T inference {self.inference_count} completed for cycle {cycle_id}")

        except Exception as e:
            logger.error(f"Error in GR00T inference {self.inference_count}: {e}")

    def _preprocess_observation(self, observation_dict, camera_keys, robot_state_keys):
        """Preprocess observation for GR00T model"""
        obs_dict = {f"video.{key}": observation_dict[key] for key in camera_keys}

        state = np.array([observation_dict[k] for k in robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float64)
        obs_dict["state.gripper"] = state[5:6].astype(np.float64)
        obs_dict["annotation.human.task_description"] = self.language_instruction

        # Add batch dimension
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        return obs_dict

    def _run_inference(self, obs_dict):
        """Run GR00T model inference"""
        return self.policy.get_action(obs_dict)

    def _convert_actions(self, action_chunk, robot_state_keys):
        """Convert GR00T actions to LeRobot format"""
        lerobot_actions = []
        for i in range(16):  # GR00T outputs 16 steps
            concat_action = np.concatenate(
                [np.atleast_1d(action_chunk[f"action.{key}"][i]) for key in self.modality_keys], axis=0
            )

            action_dict = {key: concat_action[i] for i, key in enumerate(robot_state_keys)}
            lerobot_actions.append(action_dict)

        return lerobot_actions

    def _execute_actions(self, actions):
        """Execute actions on robot"""
        if not actions or not self.robot_status_op or not self.robot_status_op.robot:
            return

        self.robot_status_op.set_action_in_progress(True)

        try:
            for action_dict in actions:
                try:
                    self.robot_status_op.robot.send_action(action_dict)
                    time.sleep(0.02)
                except Exception as e:
                    logger.error(f"Error executing action: {e}")
        finally:
            self.robot_status_op.set_action_in_progress(False)
