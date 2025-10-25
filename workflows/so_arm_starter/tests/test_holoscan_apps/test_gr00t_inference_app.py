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

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import holoscan_apps.gr00t_inference_app as appmod
from holoscan_apps.gr00t_inference_app import GR00TCyclicApplication
from holoscan_apps.operators import GR00TInferenceOp, RobotStatusOp


class TestGR00TInferenceApp(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary config.yaml for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text(
            """
robot:
  type: "so101_follower"
  port: "/dev/ttyACM0"
  id: "so101_follower_arm"
  cameras:
    wrist:
      type: "opencv"
      index_or_path: 2
      width: 640
      height: 480
      fps: 30
    room:
      type: "opencv"
      index_or_path: 0
      width: 640
      height: 480
      fps: 30

gr00t:
  host: "localhost"
  port: 5555
  language_instruction: "Pick up the scissors"
  action_horizon: 8
  model_path: "/tmp/model"
  trt_engine_path: "/tmp/engines"
  data_config: "so100_dualcam"
  video_keys: ["video.room", "video.wrist"]
  trt: true
			""".strip()
        )
        # Use a real Application fragment for operator construction
        self.fragment_app = GR00TCyclicApplication(config_path=str(self.config_path))

    def tearDown(self) -> None:
        # Clean up temporary directory
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_robot_status_operator_attributes(self):
        op = RobotStatusOp(fragment=self.fragment_app, robot_config=MagicMock())
        self.assertIsNotNone(op.robot_config)
        self.assertEqual(op.cycle_count, 0)
        self.assertTrue(op.running)
        self.assertFalse(op.action_in_progress)

    def test_gr00t_inference_operator_attributes(self):
        op = GR00TInferenceOp(
            fragment=self.fragment_app,
            policy=MagicMock(),
            language_instruction="Pick up",
            action_horizon=3,
            robot_status_op=MagicMock(),
        )
        self.assertEqual(op.language_instruction, "Pick up")
        self.assertEqual(op.action_horizon, 3)
        self.assertEqual(op.inference_count, 0)

    def test_app_load_config_errors(self):
        with self.assertRaises(ValueError):
            GR00TCyclicApplication(config_path=None)
        with self.assertRaises(FileNotFoundError):
            GR00TCyclicApplication(config_path="/no/such/file.yaml")

    def test_create_robot_config(self):
        app = GR00TCyclicApplication(config_path=str(self.config_path))
        robot_cfg = app.create_robot_config()
        self.assertIsInstance(robot_cfg, appmod.SO101FollowerConfig)
        self.assertIn("wrist", robot_cfg.cameras)
        self.assertIn("room", robot_cfg.cameras)
        self.assertIsInstance(robot_cfg.cameras["wrist"], appmod.OpenCVCameraConfig)

    def test_create_gr00t_policy(self):
        app = GR00TCyclicApplication(config_path=str(self.config_path))
        with (
            patch.object(appmod, "Gr00tPolicy", autospec=True) as MockPolicy,
            patch.object(appmod, "setup_tensorrt_engines") as mock_setup,
            patch("os.path.exists", return_value=True),
        ):
            policy = app.create_gr00t_policy()
            MockPolicy.assert_called_once()
            mock_setup.assert_called_once()
            args, _ = mock_setup.call_args
            self.assertIs(args[0], policy)
            self.assertEqual(args[1], "/tmp/engines")


if __name__ == "__main__":
    unittest.main()
