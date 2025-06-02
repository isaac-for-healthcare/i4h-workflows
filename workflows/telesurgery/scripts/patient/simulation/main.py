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

import argparse
import json
import os

from i4h_asset_helper import BaseI4HAssets
from isaaclab.app import AppLauncher


class Assets(BaseI4HAssets):
    MIRA_ARM = "Robots/MIRA/mira-bipo-size-experiment-smoothing.usd"


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the camera application")
    parser.add_argument("--camera", type=str, default="cv2", choices=["realsense", "cv2"], help="camera type")
    parser.add_argument("--name", type=str, default="robot", help="camera name")
    parser.add_argument("--width", type=int, default=1920, help="width")
    parser.add_argument("--height", type=int, default=1080, help="height")
    parser.add_argument("--framerate", type=int, default=30, help="frame rate")
    parser.add_argument("--encoder", type=str, choices=["nvjpeg", "nvc", "none"], default="nvjpeg")
    parser.add_argument("--encoder_params", type=str, default=json.dumps({"quality": 90}), help="encoder params")
    parser.add_argument("--domain_id", type=int, default=779, help="dds domain id")
    parser.add_argument("--topic", type=str, default="", help="dds topic name")
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="local api server host")
    parser.add_argument("--api_port", type=int, default=8081, help="local api server port")
    args = parser.parse_args()

    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app
    my_assets = Assets()
    usd_path = my_assets.MIRA_ARM

    # Import Isaac/Omni modules after app launch
    import omni.usd
    from omni.timeline import get_timeline_interface
    from pxr import UsdPhysics

    omni.usd.get_context().open_stage(usd_path)

    # Paths and configuration
    robot_usd_root = "/World/A5_GUI_MODEL/A5_GUI_MODEL_001"
    left_arm_base = f"{robot_usd_root}/ASM_L654321"
    right_arm_base = f"{robot_usd_root}/ASM_R654321"
    LJ_PATHS = [
        f"{left_arm_base}/LJ1/LJ1_joint",
        f"{left_arm_base}/ASM_L65432/LJ2/LJ2_joint",
        f"{left_arm_base}/ASM_L65432/ASM_L6543/LJ3/LJ3_joint",
        f"{left_arm_base}/ASM_L65432/ASM_L6543/ASM_L654/LJ4/LJ4_joint",
        f"{left_arm_base}/ASM_L65432/ASM_L6543/ASM_L654/ASM_L65/LJ5/LJ5_joint",
        f"{left_arm_base}/ASM_L65432/ASM_L6543/ASM_L654/ASM_L65/ASM_L61/LJ6/LJ6_1_joint",
    ]
    RJ_PATHS = [
        f"{right_arm_base}/RJ1/RJ1_joint",
        f"{right_arm_base}/ASM_R65432/RJ2/RJ2_joint",
        f"{right_arm_base}/ASM_R65432/ASM_R6543/RJ3/RJ3_joint",
        f"{right_arm_base}/ASM_R65432/ASM_R6543/ASM_R654/RJ4/RJ4_joint",
        f"{right_arm_base}/ASM_R65432/ASM_R6543/ASM_R654/ASM_R65/RJ5/RJ5_joint",
        f"{right_arm_base}/ASM_R65432/ASM_R6543/ASM_R654/ASM_R65/ASM_R6/RJ6/RJ6_joint",
    ]
    camera_base = f"{robot_usd_root}/C_ASM_6543210"
    camera_prim_path = f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65/C_ASM_6/Camera_Tip/Camera"

    stage = omni.usd.get_context().get_stage()
    left_arm_joint_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in LJ_PATHS]
    right_arm_joint_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in RJ_PATHS]

    left_pose = [15.7, 80.6, 45.0, 98.9, 0.0, 0.0]
    right_pose = [15.7, 80.6, 45.0, 98.9, 0.0, 0.0]

    def update_arm_joints():
        for i, api in enumerate(left_arm_joint_apis):
            api.GetTargetPositionAttr().Set(left_pose[i])
        for i, api in enumerate(right_arm_joint_apis):
            api.GetTargetPositionAttr().Set(right_pose[i])

    def on_gamepad_event(message):
        if message["method"] == "set_mira_polar_delta" or message["method"] == "set_mira_cartesian_delta":
            for i in range(6):
                left_pose[i] += message["pose_delta"]["left"][i]
                right_pose[i] += message["pose_delta"]["right"][i]
        elif message["method"] == "set_mira_pose":
            for i in range(6):
                left_pose[i] = message["params"]["left"][i]
                right_pose[i] = message["params"]["right"][i]

        print(f"Update ({message['method']}):: Left: {left_pose}; Right: {right_pose}")

    from patient.simulation.camera.sensor import CameraEx

    camera = CameraEx(
        channels=4 if args.encoder == "nvc" else 3,
        prim_path=camera_prim_path,
        frequency=args.framerate,
        resolution=(args.width, args.height),
    )
    camera.initialize()

    # holoscan app in async mode to consume camera source
    from patient.simulation.camera.app import App as CameraApp

    if os.path.isfile(args.encoder_params):
        with open(args.encoder_params, "r") as f:
            encoder_params = json.load(f)
    else:
        encoder_params = json.loads(args.encoder_params) if args.encoder_params else {}

    camera_app = CameraApp(
        width=args.width,
        height=args.height,
        encoder=args.encoder,
        encoder_params=encoder_params,
        dds_domain_id=args.domain_id,
        dds_topic=args.topic if args.topic else f"telesurgery/{args.name}_camera/rgb",
    )
    f1 = camera_app.run_async()
    camera.set_callback(camera_app.on_new_frame_rcvd)

    from patient.simulation.mira.app import App as MiraApp

    gamepad_app = MiraApp(api_host=args.api_host, api_port=args.api_port, callback=on_gamepad_event)
    f2 = gamepad_app.run_async()
    timeline = get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
    while simulation_app.is_running():
        update_arm_joints()
        simulation_app.update()

    f1.cancel()
    f2.cancel()
    simulation_app.close()


if __name__ == "__main__":
    main()
