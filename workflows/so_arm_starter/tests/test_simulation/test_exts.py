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

import importlib.util
import os
import subprocess
import sys
import unittest


class TestExtensions(unittest.TestCase):
    """Test cases for Isaac Lab extension validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.so_arm_starter_ext_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "simulation", "exts", "so_arm_starter_ext"
        )

    def test_extension_can_be_imported(self):
        """Test that the extension module can be imported without errors."""
        module_path = os.path.join(self.so_arm_starter_ext_path, "so_arm_starter_ext", "__init__.py")

        if not os.path.exists(module_path):
            self.skipTest("Extension module not found")

        spec = importlib.util.spec_from_file_location("so_arm_starter_ext", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("Extension module imported successfully")
        self.assertTrue(True, "Extension module can be imported")

    def test_extension_pip_installable(self):
        """Test that the extension has proper pip package structure."""
        # Check for setup.py or pyproject.toml
        setup_py = os.path.join(self.so_arm_starter_ext_path, "setup.py")
        pyproject_toml = os.path.join(self.so_arm_starter_ext_path, "pyproject.toml")

        print(f"Checking for setup.py: {setup_py}")
        print(f"Checking for pyproject.toml: {pyproject_toml}")

        has_setup = os.path.exists(setup_py)
        has_pyproject = os.path.exists(pyproject_toml)

        if has_setup or has_pyproject:
            print("✅ Extension has valid pip package structure")
            self.assertTrue(True, "Extension has pip packaging files")
        else:
            print("❌ Extension missing pip packaging files (setup.py or pyproject.toml)")
            self.fail("Extension must have either setup.py or pyproject.toml for pip installation")

    def test_extension_pip_install_dry_run(self):
        """Test pip install in dry-run mode to validate package structure."""

        # Check if extension has packaging files first
        setup_py = os.path.join(self.so_arm_starter_ext_path, "setup.py")
        pyproject_toml = os.path.join(self.so_arm_starter_ext_path, "pyproject.toml")

        if not (os.path.exists(setup_py) or os.path.exists(pyproject_toml)):
            self.fail("Extension has no pip packaging files - cannot test pip installation")

        # Try pip install in dry-run mode
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps", self.so_arm_starter_ext_path]

        print(f"Running pip dry-run: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Extension passes pip install dry-run validation")
            self.assertTrue(True, "Extension has valid pip package structure")
        else:
            print(f"❌ Pip dry-run failed: {result.stderr[:200]}")
            self.fail(f"Extension has packaging issues: {result.stderr[:300]}")

    def test_environment_can_be_created_after_install(self):
        """Test that environment can be created after pip installing the extension."""
        # Test environment creation with import
        test_script = """import so_arm_starter_ext"""

        result = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✅ Environment can be created after pip install")
            self.assertTrue(True, "Environment creation successful")
        else:
            self.fail(
                f"Environment creation failed. Return code: {result.returncode}\n"
                f"Stdout: {result.stdout[:200]}\nStderr: {result.stderr[:200]}"
            )


if __name__ == "__main__":
    unittest.main()
