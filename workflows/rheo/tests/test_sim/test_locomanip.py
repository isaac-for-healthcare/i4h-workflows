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
import shutil
import unittest

import numpy as np
import torch
from huggingface_hub import snapshot_download
from simulation.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from simulation.gr00t_closedloop_policy import CustomGr00tClosedloopPolicy
from tests.helpers import create_temp_closedloop_policy_config, requires_groot, requires_isaac_sim


class TestGr00tPolicyRunning(unittest.TestCase):
    """Test GR00T policy with real model from HuggingFace."""

    @classmethod
    def setUpClass(cls):
        """Download model from HuggingFace and configure headless mode before running tests."""
        print("\n" + "=" * 60)
        print("Setting up GR00T Policy Test (Headless Mode)")
        print("=" * 60)

        # Configure headless mode
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["DISPLAY"] = ""
        print("✓ Headless mode configured\n")

        cls.model_path = "/tmp/rheo_test_model/checkpoint-20000"
        cls.download_path = "/tmp/rheo_test_model"

        # Download model from HuggingFace
        print("Downloading Model from HuggingFace...")
        try:
            snapshot_download(
                repo_id="nvidia/orca-dev-test",
                repo_type="dataset",
                allow_patterns="push_cart_new_light_n16/checkpoint-20000/*",
                local_dir=cls.download_path,
                local_dir_use_symlinks=False,
            )
            # Adjust path to match downloaded structure
            cls.model_path = os.path.join(cls.download_path, "push_cart_new_light_n16", "checkpoint-20000")
            print(f"Model downloaded successfully to: {cls.model_path}\n")
        except Exception as e:
            print(f"Failed to download model: {e}\n")
            cls.model_path = None

    @classmethod
    def tearDownClass(cls):
        """Clean up downloaded model after tests."""
        if os.path.exists(cls.download_path):
            shutil.rmtree(cls.download_path)
            print(f"Removed: {cls.download_path}\n")

    def _create_mock_observation(self):
        """Create mock observation data for policy inference."""
        num_envs = 1
        H, W, C = 480, 640, 3
        num_joints = 43  # G1 has 43 DOF for locomanipulation

        obs = {
            "camera_obs": {
                "robot_head_cam_rgb": torch.randint(0, 255, (num_envs, H, W, C), dtype=torch.uint8).cuda(),
            },
            "policy": {
                "robot_joint_pos": torch.randn(num_envs, num_joints, dtype=torch.float32).cuda(),
            },
        }
        return obs

    @requires_isaac_sim
    @requires_groot
    def test_policy_inference_with_real_model(self):
        """Test policy inference with real model and mock observation in headless mode (CUDA only)."""
        if self.model_path is None or not os.path.exists(self.model_path):
            self.skipTest("Model not available - skipping inference test")

        # Check CUDA availability
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available - this test requires GPU")

        # Create mock observation
        obs = self._create_mock_observation()

        # Create temporary config file with the downloaded model path
        temp_config_path = create_temp_closedloop_policy_config(
            task_name="push_cart_test",
            model_path=self.model_path,
            language_instruction="Push the cart forward",
        )

        # Initialize policy (CUDA only)
        try:
            policy = CustomGr00tClosedloopPolicy(
                policy_config_yaml_path=temp_config_path,
                num_envs=1,
                device="cuda",
            )
        finally:
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        # Run inference
        action = policy.get_action(env=None, observation=obs)

        # Validate output
        self.assertIsNotNone(action)
        self.assertIsInstance(action, (np.ndarray, torch.Tensor))


class TestPolicyRunnerCLI(unittest.TestCase):
    """Test policy runner CLI functions."""

    @requires_isaac_sim
    def test_import_policy_runner_cli(self):
        """Test that policy runner CLI can be imported."""

        self.assertTrue(callable(create_policy))
        self.assertTrue(callable(setup_policy_argument_parser))
        print("Policy runner CLI functions imported successfully")

    @requires_isaac_sim
    def test_setup_policy_argument_parser(self):
        """Test setting up policy argument parser."""
        # Don't call setup_policy_argument_parser as it calls parse_args() internally
        # Instead just check that the function exists and is callable
        self.assertTrue(callable(setup_policy_argument_parser))
        print("policy argument parser function is callable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
