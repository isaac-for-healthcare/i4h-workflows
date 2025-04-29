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
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from simulation.environments.state_machine.utils import UltrasoundState, get_joint_states

from .data_collector import RobomimicDataCollector


@dataclass
class DataCollectionManager:
    """Manages data collection operations and state.

    This class encapsulates all data collection related functionality, including:
    - HDF5 data storage
    - Episode tracking and management
    - Data formatting and writing

    Args:
        task_name (str): Name of the task being recorded
        num_episodes (int): Number of episodes to collect
        num_envs (int): Number of parallel environments
        device (str): Device to use for computations
    """

    task_name: str
    num_episodes: int
    num_envs: int
    device: str
    is_testing: bool = False

    def __post_init__(self):
        """Initialize the data collection manager after instance creation.

        Sets up:
        - Date-stamped directory paths to prevent overwriting
        - HDF5 data collection infrastructure
        - Episode tracking counters
        """
        # Setup data collection paths, include the hour and minute to avoid overwriting
        date = datetime.now().strftime("%Y-%m-%d-%H-%M")

        # Setup HDF5 data collection
        self._setup_hdf5(date)

        # Initialize episode storage
        self.completed_episodes = 0

        # Create dummy data for testing if needed
        if self.is_testing:
            # Dummy torso data: position (3) + quaternion (4)
            self.dummy_torso_obs = np.zeros((self.num_envs, 7))
            # Dummy joint positions
            self.dummy_joint_states = np.zeros((self.num_envs, 7))

    def _setup_hdf5(self, date):
        """Set up HDF5 data collection infrastructure.

        Creates the necessary directory structure and initializes the Robomimic
        data collector with appropriate configuration.

        Args:
            date (str): Formatted date string to use in directory naming
        """
        self.hdf5_log_dir = os.path.join("./data/hdf5", f"{date}-{self.task_name}")
        os.makedirs(self.hdf5_log_dir, exist_ok=True)
        self.collector_interface = RobomimicDataCollector(
            env_name=self.task_name,
            directory_path=self.hdf5_log_dir,
            filename="data",
            num_demos=self.num_episodes,
            flush_freq=self.num_envs,
            env_config={"collection_methods": os.path.basename(__file__)},
        )

    def record_step(
        self,
        env: gym.Env,
        obs: Dict,
        rel_action: torch.Tensor,
        abs_action: torch.Tensor,
        robot_obs: torch.Tensor,
        state: str,
    ) -> None:
        """Record a single step of data to the HDF5 collector.

        Processes and stores various observation and action data from the current
        simulation step, including robot state, actions, and environment observations.

        Args:
            env (gym.Env): The gymnasium environment instance
            obs (Dict): Dictionary of observations from the environment
            rel_action (torch.Tensor): Relative action tensor
            abs_action (torch.Tensor): Absolute action tensor
            robot_obs (torch.Tensor): Robot observation tensor
            state (str): Current state machine state as a string
        """
        # Convert tensors to numpy
        rel_action_np = rel_action.cpu().numpy()
        abs_action_np = abs_action.cpu().numpy()
        robot_obs_np = robot_obs.cpu().numpy()

        # Get real or dummy data based on testing flag
        if self.is_testing:
            torso_obs = self.dummy_torso_obs
            abs_joint_pos = self.dummy_joint_states
        else:
            torso_obs = self.get_torso_obs(env)
            abs_joint_pos = get_joint_states(env)
        state_np = self.state_to_np(state)

        # Store in HDF5
        self.collector_interface.add("observations/torso_obs", torso_obs)
        self.collector_interface.add("observations/robot_obs", robot_obs_np)
        self.collector_interface.add("observations/room_camera_pos", env.unwrapped.scene['room_camera'].data.pos_w.cpu().numpy())
        self.collector_interface.add("observations/room_camera_quat_w_world", env.unwrapped.scene['room_camera'].data.quat_w_world.cpu().numpy())
        self.collector_interface.add("observations/room_camera_intrinsic_matrices", env.unwrapped.scene['room_camera'].data.intrinsic_matrices.cpu().numpy())

        self.collector_interface.add("observations/wrist_camera_pos", env.unwrapped.scene['wrist_camera'].data.pos_w.cpu().numpy())
        self.collector_interface.add("observations/wrist_camera_quat_w_world", env.unwrapped.scene['wrist_camera'].data.quat_w_world.cpu().numpy())
        self.collector_interface.add("observations/wrist_camera_intrinsic_matrices", env.unwrapped.scene['wrist_camera'].data.intrinsic_matrices.cpu().numpy())

        self.collector_interface.add("action", rel_action_np)
        self.collector_interface.add("abs_action", abs_action_np)
        self.collector_interface.add("state", state_np)
        self.collector_interface.add("abs_joint_pos", abs_joint_pos)

        # Add remaining observations
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                self.collector_interface.add(f"observations/{key}", value.cpu().numpy())
            else:
                # Skip nested dictionaries containing joint velocities
                pass

    def on_episode_complete(self) -> None:
        """Handle logic when an episode is successfully completed.

        Flushes collected data to disk, increments the completed episode counter,
        and logs completion information.
        """
        # Flush HDF5 data
        self.collector_interface.flush(list(range(self.num_envs)))

        # Clear episode data and increment completed count
        self.completed_episodes += 1
        print(f"[INFO]: Episode {self.completed_episodes} complete.")

    def on_episode_reset(self) -> None:
        """Handle logic when an episode is reset before completion.

        Resets the HDF5 collector interface to prepare for a new episode and
        logs information about the incomplete episode.
        """
        # Reset HDF5 collector interface
        self.collector_interface.reset()

        print(f"[INFO]: Episode {self.completed_episodes} incomplete.")

    def get_torso_obs(self, env):
        """Extract torso observation data from the environment.

        Retrieves position and orientation data for the organ torso from the
        simulation environment and formats it for storage.

        Args:
            env: The simulation environment containing the scene data

        Returns:
            numpy.ndarray: Formatted torso observation data as a numpy array
        """
        torso_data = env.unwrapped.scene["organs"].data
        torso_pos = torso_data.root_pos_w - env.unwrapped.scene.env_origins
        torso_quat = torso_data.root_quat_w
        torso_obs = torch.cat([torso_pos, torso_quat], dim=-1)
        return torso_obs.cpu().numpy()

    @staticmethod
    def state_to_np(state: str) -> np.ndarray:
        """Convert a state string to a numeric representation for storage.

        Maps the string representation of an UltrasoundState enum value to its
        corresponding integer index and returns it as a numpy array for HDF5 storage.

        Args:
            state (str): String representation of an UltrasoundState enum value

        Returns:
            numpy.ndarray: Single-element array containing the state's integer index

        Raises:
            ValueError: If the provided state string doesn't match any UltrasoundState enum value
        """
        for i, state_enum in enumerate(UltrasoundState):
            if str(state) == state_enum.value:
                return np.array([i])
        raise ValueError(f"State {state} not found in UltrasoundState enum.")
