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
import sys
import tempfile
import unittest

from i4h_asset_helper import get_i4h_local_asset_path
from simulation.utils.assets import Assets
from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets


class TestAssets(unittest.TestCase):
    def setUp(self):
        # Reset the singleton instance before each test
        Assets._instance = None

    def test_import_failure(self):
        # delete the assets module
        from simulation.configs.basic import config

        self.assertEqual(config.main_usd_path, get_i4h_local_asset_path() + "/Test/basic.usda")
        sys.modules.pop("simulation.configs.basic")  # un-import the config

    def test_download_dir_setter(self):
        # Change the download directory using explicit setter
        with tempfile.TemporaryDirectory() as temp_dir:
            robot_us_assets.set_download_dir(temp_dir)
            # import Config to test that change was reflected
            from simulation.configs.basic import config

            self.assertEqual(config.main_usd_path, os.path.join(temp_dir, "Test/basic.usda"))
            sys.modules.pop("simulation.configs.basic")  # un-import the config

    def test_singleton_behavior(self):
        # Create first instance
        assets1 = Assets()
        # Create second instance
        assets2 = Assets()
        # Verify they are the same object
        self.assertIs(assets1, assets2)
        # Verify they share the same download directory
        self.assertEqual(assets1.download_dir, assets2.download_dir)

    def test_initialization_warning(self):
        # Create first instance
        assets1 = Assets("/path/one")

        # Capture warning during second initialization
        with self.assertLogs(level="WARNING") as captured:
            # Try to initialize with different path
            assets2 = Assets("/path/two")

            # Verify warning was logged
            self.assertTrue(any("Assets already initialized" in message for message in captured.output))

        # Verify the download directory wasn't changed
        self.assertEqual(assets1.download_dir, "/path/one")
        self.assertEqual(assets2.download_dir, "/path/one")
        # Verify they are the same instance
        self.assertIs(assets1, assets2)

    def tearDown(self):
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("simulation.utils"):
                sys.modules.pop(module_name)


if __name__ == "__main__":
    unittest.main()
