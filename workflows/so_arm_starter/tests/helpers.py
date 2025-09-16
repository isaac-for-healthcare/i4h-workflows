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
import subprocess
from unittest import skipUnless

import h5py
import numpy as np


def requires_rti(func):
    """Decorator to skip tests if RTI Connext DDS is not available."""
    RTI_AVAILABLE = bool(os.getenv("RTI_LICENSE_FILE") and os.path.exists(os.getenv("RTI_LICENSE_FILE")))
    return skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")(func)


def requires_so101_hardware(func):
    """Decorator to skip tests if SO-ARM101 hardware is not available."""
    # Check if SO-ARM101 ports are available
    try:
        result = subprocess.run(
            ["python", "-c", "import lerobot; print('SO101 available')"], capture_output=True, text=True, timeout=10
        )
        SO101_AVAILABLE = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        SO101_AVAILABLE = False

    return skipUnless(SO101_AVAILABLE, "SO-ARM101 hardware is not available or lerobot not properly installed")(func)


def requires_isaac_sim(func):
    """Decorator to skip tests if Isaac Sim is not available."""
    try:
        import omni  # noqa: F401

        ISAAC_SIM_AVAILABLE = True
    except ImportError:
        ISAAC_SIM_AVAILABLE = False

    return skipUnless(ISAAC_SIM_AVAILABLE, "Isaac Sim is not available. Please install Isaac Sim 4.5.0 or later")(func)


def _get_max_gpu_memory_mib():
    """
    Gets the maximum total memory of any single GPU in MiB using nvidia-smi.
    Returns 0 if nvidia-smi fails or no GPUs are found.
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        lines = result.splitlines()
        gpu_memories_mib = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                try:
                    gpu_memories_mib.append(int(stripped_line))
                except ValueError as e_inner:
                    print(f"Warning: Could not parse line '{stripped_line}' as int. Error: {e_inner}")
        if not gpu_memories_mib:
            return 0
        return max(gpu_memories_mib)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Warning: Could not query GPU memory using nvidia-smi: {e}. Assuming 0 MiB GPU RAM.")
        return 0
    except Exception as e_generic:
        print(f"Warning: An unexpected error occurred while parsing GPU memory: {e_generic}. Assuming 0 MiB GPU RAM.")
        return 0


def requires_gpu_memory(min_gib: int = 25):
    """
    Decorator to skip a test if the maximum available GPU memory on a single GPU
    is less than min_gib.

    Args:
        min_gib: The minimum required GPU memory in GiB.
    """

    def decorator(func):
        max_gpu_memory_mib = _get_max_gpu_memory_mib()
        max_gpu_memory_gib = max_gpu_memory_mib / 1024
        return skipUnless(
            max_gpu_memory_gib >= min_gib,
            f"Requires at least {min_gib} GiB GPU memory, but only {max_gpu_memory_gib:.1f} GiB available",
        )(func)

    return decorator


def cleanup_test_files(test_files):
    """Clean up test files after tests."""
    for file_path in test_files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil

                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Warning: Could not clean up {file_path}: {e}")


def create_dummy_hdf5_files(hdf5_data_dir, num_episodes=2, num_steps=50):
    """Create dummy HDF5 files with the expected structure for SO-ARM101.

    This function creates test datasets that match the format used by the actual
    recording system, based on the pattern from the GR00T training tests.

    Args:
        hdf5_data_dir: Directory to create HDF5 files in
        num_episodes: Number of episodes to create
        num_steps: Number of steps per episode
    """
    os.makedirs(hdf5_data_dir, exist_ok=True)

    for episode_idx in range(num_episodes):
        # Create a dummy HDF5 file (following ultrasound test pattern)
        hdf5_path = os.path.join(hdf5_data_dir, f"data_{episode_idx}.hdf5")

        with h5py.File(hdf5_path, "w") as f:
            # Create the root group (following ultrasound pattern)
            root_name = "data/demo_0"
            root_group = f.create_group(root_name)

            # Mark as successful
            root_group.attrs["success"] = True

            obs_group = root_group.create_group("obs")

            # Create  dataset (num_steps, 6) - SO-ARM101
            actions = np.random.uniform(-1.0, 1.0, (num_steps, 6)).astype(np.float32)
            obs_group.create_dataset("actions", data=actions)

            joint_pos = np.random.uniform(-1.0, 1.0, (num_steps, 6)).astype(np.float32)
            obs_group.create_dataset("joint_pos", data=joint_pos)

            # Create camera images (num_steps, 480, 640, 3) - Match expected dimensions
            room_images = np.random.randint(0, 256, size=(num_steps, 480, 640, 3), dtype=np.uint8)
            obs_group.create_dataset("room", data=room_images)

            wrist_images = np.random.randint(0, 256, size=(num_steps, 480, 640, 3), dtype=np.uint8)
            obs_group.create_dataset("wrist", data=wrist_images)
