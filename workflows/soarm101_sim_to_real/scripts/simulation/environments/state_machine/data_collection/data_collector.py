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

# Modified from Isaac Lab to allow nested keys in the dataset and skipping incomplete episodes
# Original File here: https://github.com/isaac-sim/IsaacLab/blob/main/source/standalone/workflows/robomimic/collect_demonstrations.py
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interface to collect and store data from the environment using format from `robomimic`."""

# needed to import for allowing type-hinting: np.ndarray | torch.Tensor
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable

import h5py
import numpy as np
import torch


class RobomimicDataCollector:
    """Data collection interface for robomimic.

    This class implements a data collector interface for saving simulation states to disk.
    The data is stored in `HDF5`_ binary data format. The class is useful for collecting
    demonstrations. The collected data follows the `structure`_ from robomimic.

    All datasets in `robomimic` require the observations and next observations obtained
    from before and after the environment step. These are stored as a dictionary of
    observations in the keys "obs" and "next_obs" respectively.

    For certain agents in `robomimic`, the episode data should have the following
    additional keys: "actions", "rewards", "dones". This behavior can be altered by changing
    the dataset keys required in the training configuration for the respective learning agent.

    For reference on datasets, please check the robomimic `documentation`.

    .. _HDF5: https://www.h5py.org/
    .. _structure: https://robomimic.github.io/docs/datasets/overview.html#dataset-structure
    .. _documentation: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/config/base_config.py#L167-L173
    """

    def __init__(
        self,
        env_name: str,
        directory_path: str,
        filename: str = "demo",
        num_demos: int = 1,
        num_envs: int = 1,
        flush_freq: int = 1,
        env_config: dict | None = None,
    ):
        """Initialize the RobomimicDataCollector.

        Args:
            env_name (str): Name of the environment being recorded
            directory_path (str): Path to directory where data will be stored
            filename (str, optional): Base filename for the demo files. Defaults to "demo".
            num_demos (int, optional): Number of demonstrations to collect. Defaults to 1.
            num_envs (int, optional): Number of parallel environments. Defaults to 1.
            flush_freq (int, optional): Frequency of printing status updates. Defaults to 1.
            env_config (dict, optional): Environment configuration to store with the data. Defaults to None.
        """
        self._env_name = env_name
        self._env_config = env_config
        self._directory = os.path.abspath(directory_path)
        self._filename = filename
        self._num_demos = num_demos
        self._flush_freq = flush_freq
        self._num_envs = num_envs
        self._idx_offset = 0

        print(self.__str__())

        if not os.path.isdir(self._directory):
            os.makedirs(self._directory)

        self._demo_count = 0
        self._is_first_interaction = True
        self._is_stop = False
        self._dataset = dict()
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        """Destructor that ensures data is properly saved before object deletion.

        Calls the close() method if the collector hasn't been explicitly stopped.
        """
        if not self._is_stop:
            self.close()

    def __str__(self) -> str:
        """Generate a string representation of the data collector.

        Returns:
            str: A formatted string with information about the collector configuration.
        """
        msg = "Dataset collector object\n"
        msg += f"\tStoring trajectories in directory: {self._directory}\n"
        msg += f"\tNumber of demos for collection : {self._num_demos}\n"
        msg += f"\tFrequency for saving data to disk: {self._flush_freq}\n"
        return msg

    @property
    def demo_count(self) -> int:
        """Get the current number of collected demonstrations.

        Returns:
            int: Number of demonstrations collected so far.
        """
        return self._demo_count

    def increment_idxs(self):
        """Increment the environment index offset and reset tracking state.

        This method is used when moving to a new batch of environments, resetting
        the internal tracking state and incrementing the index offset by the number
        of environments.
        """
        self._idx_offset += self._num_envs
        self._dataset = dict()

    def is_stopped(self) -> bool:
        """Check if the data collector has been stopped.

        Returns:
            bool: True if the collector has been stopped, False otherwise.
        """
        return self._is_stop

    def reset(self):
        """Reset the data collector's internal state.

        Clears the current dataset dictionary and initializes the collector if this
        is the first interaction. Should be called before starting to collect data
        for a new episode.
        """
        if self._is_first_interaction:
            self._is_first_interaction = False
        self._dataset = dict()

    def add(self, key: str, value: np.ndarray | torch.Tensor):
        """Add data to the current episode collection.

        Stores the provided data under the specified key in the dataset. Supports
        nested keys using "/" as a separator (up to 3 levels deep).

        Args:
            key (str): Key to store the data under. Can be a nested path using "/"
                      (e.g., "obs/camera", "next_obs/joint_pos").
            value (np.ndarray | torch.Tensor): Data to store. Will be converted to
                                              numpy array if it's a torch tensor.

        Raises:
            ValueError: If the key has more than three nested levels.

        Note:
            If the collector is stopped (reached desired demo count), this method
            will log a warning and return without storing data.
        """
        if self._is_first_interaction:
            self.logger.warning("Please call reset before adding new data. Calling reset...")
            self.reset()

        if self._is_stop:
            self.logger.warning(f"Desired number of demonstrations collected: {self._demo_count} >= {self._num_demos}.")
            return

        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        else:
            value = np.asarray(value)

        sub_keys = key.split("/")
        num_sub_keys = len(sub_keys)
        if len(sub_keys) > 3:
            raise ValueError(f"Input key '{key}' has {num_sub_keys} elements which is more than three.")

        for i in range(value.shape[0]):
            if f"env_{i}" not in self._dataset:
                self._dataset[f"env_{i}"] = dict()
            current_dict = self._dataset[f"env_{i}"]

            for j in range(num_sub_keys - 1):
                if sub_keys[j] not in current_dict:
                    current_dict[sub_keys[j]] = dict()
                current_dict = current_dict[sub_keys[j]]

            if sub_keys[-1] not in current_dict:
                current_dict[sub_keys[-1]] = list()
            current_dict[sub_keys[-1]].append(value[i])

    def flush(self, env_ids: Iterable[int] = (0,)):
        """Save the collected data to disk for specified environments.

        Creates HDF5 files for each specified environment ID, storing all collected
        data for that environment in the appropriate format for robomimic.

        Args:
            env_ids (Iterable[int], optional): Environment IDs to flush data for.
                                              Defaults to (0,).

        Note:
            This method will automatically stop the collector if the desired number
            of demonstrations has been reached.
        """
        for index in env_ids:
            actual_index = index + self._idx_offset
            env_dataset = self._dataset[f"env_{actual_index}"]

            # Create a new HDF5 file for this demo
            demo_filename = f"{self._filename}_{self._demo_count}.hdf5"
            demo_filepath = os.path.join(self._directory, demo_filename)

            with h5py.File(demo_filepath, "w") as h5_file:
                data_group = h5_file.create_group("data")
                episode_group = data_group.create_group("demo_0")

                episode_group.attrs["num_samples"] = len(env_dataset["action"])
                episode_group.attrs["sim"] = False
                data_group.attrs["total"] = episode_group.attrs["num_samples"]

                for key, value in env_dataset.items():
                    if isinstance(value, dict):
                        key_group = episode_group.create_group(key)
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict):
                                key_group2 = key_group.create_group(sub_key)
                                for sub_key2, sub_value2 in sub_value.items():
                                    key_group2.create_dataset(sub_key2, data=np.array(sub_value2))
                            else:
                                key_group.create_dataset(sub_key, data=np.array(sub_value))
                    else:
                        episode_group.create_dataset(key, data=np.array(value))

                # Store the environment meta-info
                env_type = 2
                if self._env_config is None:
                    self._env_config = dict()
                data_group.attrs["env_args"] = json.dumps(
                    {
                        "env_name": self._env_name,
                        "type": env_type,
                        "env_kwargs": self._env_config,
                    }
                )

            self._demo_count += 1
            self._dataset[f"env_{index}"] = dict()

            if self._demo_count % self._flush_freq == 0:
                print(f">>> Flushing data to disk. Collected demos: {self._demo_count} / {self._num_demos}")

            if self._demo_count >= self._num_demos:
                print(f">>> Desired number of demonstrations collected: {self._demo_count} >= {self._num_demos}.")
                self.close()
                break

        self._dataset = dict()

    def close(self):
        """Close the data collector and finalize data recording.

        Marks the collector as stopped and logs the final collection status.
        This method should be called when data collection is complete.
        """
        if not self._is_stop:
            print(f">>> Closing recording of data. Collected demos: {self._demo_count} / {self._num_demos}")
            self._is_stop = True
