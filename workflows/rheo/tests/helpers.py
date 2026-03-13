# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tempfile
from pathlib import Path
from unittest import skipUnless

import yaml


def requires_isaac_sim(func):
    """Decorator to skip tests if Isaac Sim is not available."""
    try:
        import omni  # noqa: F401

        ISAAC_SIM_AVAILABLE = True
    except ImportError:
        ISAAC_SIM_AVAILABLE = False

    return skipUnless(ISAAC_SIM_AVAILABLE, "Isaac Sim is not available. Please install Isaac Sim 5.0.0 or later")(func)


def requires_groot(func):
    """Decorator to skip tests if GR00T is not available."""
    try:
        from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: F401

        GROOT_AVAILABLE = True
    except ImportError:
        GROOT_AVAILABLE = False

    return skipUnless(GROOT_AVAILABLE, "GR00T is not installed. Install with: -g flag in run_docker.sh")(func)


def requires_gpu_memory(min_gib: int = 20):
    """
    Decorator to skip a test if the maximum available GPU memory on a single GPU
    is less than min_gib.

    Args:
        min_gib: The minimum required GPU memory in GiB (default: 20 for G1 tasks).
    """

    def decorator(func):
        max_gpu_memory_mib = _get_max_gpu_memory_mib()
        max_gpu_memory_gib = max_gpu_memory_mib / 1024
        return skipUnless(
            max_gpu_memory_gib >= min_gib,
            f"Requires at least {min_gib} GiB GPU memory, but only {max_gpu_memory_gib:.1f} GiB available",
        )(func)

    return decorator


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


def run_with_monitoring_capture(command, timeout, target_lines=None):
    """
    Run a command with monitoring and capture output.

    Args:
        command: Command string to execute
        timeout: Maximum time in seconds to wait
        target_lines: List of strings to search for in output

    Returns:
        Tuple of (return_code, found_target, output_lines)
    """
    print(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        output_lines = result.stdout.split("\n")

        # Check for target lines
        found_target = True
        if target_lines:
            for target in target_lines:
                if not any(target in line for line in output_lines):
                    found_target = False
                    print(f"Missing target line: {target}")

        return result.returncode, found_target, output_lines

    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return -1, False, []


def cleanup_test_files(test_files):
    """Clean up test files after tests."""
    for file_path in test_files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
            elif os.path.isdir(file_path):
                import shutil

                shutil.rmtree(file_path)
                print(f"Cleaned up directory: {file_path}")
        except Exception as e:
            print(f"Warning: Could not clean up {file_path}: {e}")


def create_temp_closedloop_policy_config(task_name: str, model_path: str, language_instruction: str) -> str:
    """Create a temporary policy config file for testing.

    Args:
        task_name: Name of the task (e.g., 'pick_and_place', 'push_cart')
        model_path: Path to the mock model
        language_instruction: Language instruction for the task

    Returns:
        Path to the temporary config file
    """
    # Use absolute paths for joint config files
    base_path = Path("/workspaces/third_party/IsaacLab-Arena/isaaclab_arena_gr00t/config/g1")

    config_data = {
        "model_path": model_path,
        "language_instruction": language_instruction,
        "action_horizon": 16,
        "embodiment_tag": "new_embodiment",
        "video_backend": "decord",
        "data_config": "unitree_g1_sim_wbc",
        "policy_joints_config_path": str(base_path / "gr00t_43dof_joint_space.yaml"),
        "action_joints_config_path": str(base_path / "43dof_joint_space.yaml"),
        "state_joints_config_path": str(base_path / "43dof_joint_space.yaml"),
        "action_chunk_length": 16,
        "pov_cam_name_sim": "robot_head_cam_rgb",
        "task_mode_name": "g1_locomanipulation",
    }

    # Create temporary file
    fd, temp_path = tempfile.mkstemp(suffix=f"_{task_name}_config.yaml", prefix="test_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    print(f"Created temporary config: {temp_path}")
    return temp_path
