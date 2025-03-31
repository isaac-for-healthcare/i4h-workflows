import sys
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
        new_dir = "/path/to/new"
        robot_us_assets.set_download_dir(new_dir)
        # import Config to test that change was reflected
        from simulation.configs.basic import config

        self.assertEqual(config.main_usd_path, "/path/to/new/Test/basic.usda")
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
