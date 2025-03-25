import sys
import unittest

from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets
from i4h_asset_helper import get_i4h_local_asset_path

class TestAssets(unittest.TestCase):
    def setUp(self):
        pass

    def test_import_failure(self):
        # delete the assets module
        from simulation.configs.basic import config
        self.assertEqual(config.main_usd_path, get_i4h_local_asset_path() + "/Test/basic.usda")
        sys.modules.pop("simulation.configs.basic")  # un-import the config


    def test_download_dir_setter(self):
        # Change the download directory
        new_dir = "/path/to/new"
        robot_us_assets.download_dir = new_dir
        # import Config to test that change was reflected
        from simulation.configs.basic import config
        self.assertEqual(config.main_usd_path, "/path/to/new/Test/basic.usda")
        sys.modules.pop("simulation.configs.basic")  # un-import the config


    def tearDown(self):
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("simulation.utils"):
                sys.modules.pop(module_name)



if __name__ == "__main__":
    unittest.main()
