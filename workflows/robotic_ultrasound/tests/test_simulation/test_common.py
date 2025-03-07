import os
import unittest
from unittest import mock

import numpy as np
from parameterized import parameterized
from simulation.utils.common import colorize_depth, get_exp_config, list_exp_configs
from test_simulation.common_config import config as expected_config


class TestCommonUtils(unittest.TestCase):
    @parameterized.expand(
        [
            # name, depth_data, near, far, expected_dtype, expected_shape
            ("standard_range", np.array([[2.0, 10.0], [25.0, 40.0]]), 1.0, 50.0, np.uint8, (2, 2)),
            ("out_of_range", np.array([[0.5, 60.0]]), 1.0, 50.0, np.uint8, (1, 2)),
            ("custom_range", np.array([[5.0, 15.0]]), 5.0, 15.0, np.uint8, (1, 2)),
        ]
    )
    def test_colorize_depth(self, name, depth_data, near, far, expected_dtype, expected_shape):
        result = colorize_depth(depth_data, near=near, far=far)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 255))

        # Additional checks for custom range case
        if name == "custom_range":
            self.assertEqual(result[0, 0], 255)  # Closest value should be white (255)
            self.assertEqual(result[0, 1], 0)  # Farthest value should be black (0)

    @mock.patch("os.listdir")
    @mock.patch("os.path.abspath")
    def test_list_exp_configs(self, mock_abspath, mock_listdir):
        # Mock the file list
        mock_listdir.return_value = ["config.py", "__init__.py", "experiment1.py", "experiment2.py", "not_a_config.txt"]
        mock_abspath.return_value = "/mock/path/to/configs"

        # Test with default path
        configs = list_exp_configs()
        self.assertEqual(configs, ["experiment1", "experiment2"])

        # Test with custom path
        custom_path = "/custom/path"
        configs = list_exp_configs(custom_path)
        mock_listdir.assert_called_with(custom_path)
        self.assertEqual(configs, ["experiment1", "experiment2"])

    def test_get_exp_config(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the config file using the function we're testing
        loaded_config = get_exp_config("common_config", test_dir)

        # Compare the loaded config with the expected config
        self.assertEqual(loaded_config.main_usd_path, expected_config.main_usd_path)
        self.assertEqual(loaded_config.room_camera.prim_path, expected_config.room_camera.prim_path)
        self.assertEqual(loaded_config.wrist_camera.prim_path, expected_config.wrist_camera.prim_path)
        self.assertEqual(loaded_config.franka.prim_path, expected_config.franka.prim_path)
        self.assertEqual(loaded_config.target.prim_path, expected_config.target.prim_path)
        self.assertEqual(loaded_config.ultrasound.prim_path, expected_config.ultrasound.prim_path)

    def test_get_exp_config_nonexistent(self):
        """Test that we get an appropriate error for a non-existent config."""
        test_dir = os.path.dirname(os.path.abspath(__file__))

        with self.assertRaises(Exception):
            get_exp_config("nonexistent_config", test_dir)


if __name__ == "__main__":
    unittest.main()
