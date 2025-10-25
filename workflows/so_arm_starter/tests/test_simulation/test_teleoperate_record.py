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
import sys
import tempfile
import unittest

from helpers import cleanup_test_files


class TestTeleoperation(unittest.TestCase):
    """Test cases for teleoperation_record.py functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = tempfile.mkdtemp()
        self.test_files.append(self.temp_dir)

        script_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "simulation", "environments")
        self.teleop_script = os.path.join(script_dir, "teleoperation_record.py")

        if not os.path.exists(self.teleop_script):
            raise FileNotFoundError(f"Teleoperation script not found at {self.teleop_script}")

    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)

    def _run_teleop_command(self, extra_args=None, timeout=30):
        """Run the teleoperation script with given arguments."""
        cmd = [
            sys.executable,
            self.teleop_script,
            "--task",
            "Isaac-SOARM101-v0",
            "--teleop_device",
            "keyboard",
            "--record",
            "--headless",
        ]

        if extra_args:
            cmd.extend(extra_args)

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        return result.returncode == 0, result.stdout, result.stderr

    def test_invalid_teleop_device(self):
        """Test handling of invalid teleoperation device."""
        success, stdout, stderr = self._run_teleop_command(
            extra_args=[
                "--teleop_device",
                "invalid_device",
            ],
            timeout=15,
        )

        self.assertFalse(success, "Should fail with invalid teleop device")
        self.assertTrue(
            "invalid choice" in stderr or "choose from" in stderr,
            f"Should show argument error for invalid device: {stderr[:200]}",
        )

    def test_keyboard_recording(self):
        """Test automated key press framework and verify headless keyboard limitations."""
        dataset_path = os.path.join(self.temp_dir, "keyboard.hdf5")

        success, stdout, stderr = self._run_teleop_command(
            extra_args=["--dataset_path", dataset_path],
            timeout=120,
        )

        self.assertTrue(success, "Teleoperation script should run successfully")
        print("keyboard recording test passed")


if __name__ == "__main__":
    unittest.main()
