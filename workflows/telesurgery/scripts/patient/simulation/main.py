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
import math
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
    parser.add_argument("--encoder", type=str, choices=["nvjpeg", "nvc", "none"], default="nvc", help="encoder type")
    parser.add_argument("--encoder_params", type=str, default=None, help="encoder params")
    parser.add_argument("--domain_id", type=int, default=9, help="dds domain id")
    parser.add_argument("--topic", type=str, default="", help="dds topic name")
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="local api server host")
    parser.add_argument("--api_port", type=int, default=8081, help="local api server port")
    parser.add_argument("--timeline_play", type=bool, default=True, help="play the timeline")
    parser.add_argument("--debug", action='store_true', help="show debug output")
    args = parser.parse_args()

    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app
    my_assets = Assets()
    usd_path = my_assets.MIRA_ARM

    # Import Isaac/Omni modules after app launch
    import omni.usd
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.rotations import euler_angles_to_quat
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
    camera_paths = [
        f"{camera_base}/C_ASM_654321",
        f"{camera_base}/C_ASM_654321/C_ASM_65432",
        f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543",
        f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654",
        f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65",
        f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65/C_ASM_6",
    ]
    camera_prim_path = f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65/C_ASM_6/Camera_Tip/Camera"
    max_camera_angle = 70

    stage = omni.usd.get_context().get_stage()
    left_arm_joint_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in LJ_PATHS]
    right_arm_joint_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in RJ_PATHS]

    left_pose = [15.7, 80.6, 45.0, 98.9, 0.0, 0.0]
    right_pose = [15.7, 80.6, 45.0, 98.9, 0.0, 0.0]

    def update_arm_joints():
        for i, api in enumerate(left_arm_joint_apis):
            if i == 4:
                api.GetTargetPositionAttr().Set(rotations[0])
            elif i == 5:
                api.GetTargetPositionAttr().Set(grasps[0] * (20 / 1350))
            else:
                api.GetTargetPositionAttr().Set(left_pose[i])
        for i, api in enumerate(right_arm_joint_apis):
            if i == 4:
                api.GetTargetPositionAttr().Set(rotations[1])
            elif i == 5:
                api.GetTargetPositionAttr().Set(-grasps[1] * (40 / 1350))
            else:
                api.GetTargetPositionAttr().Set(right_pose[i])

    def update_camera_pose():
        # seems the axis are swapped, match behavior with physical robot
        east = max(-max_camera_angle, min(max_camera_angle, camera_pose[0]))
        north = max(-max_camera_angle, min(max_camera_angle, camera_pose[1]))
        for i in [0]:
            pos, _ = camera_prims[i].get_local_pose()
            quat = euler_angles_to_quat([math.pi / 2, 0, -north * math.pi / 180 / 3])
            camera_prims[i].set_local_pose(translation=pos, orientation=quat)
        for i in [2, 4]:
            pos, _ = camera_prims[i].get_local_pose()
            quat = euler_angles_to_quat([0, -math.pi / 2, -north * math.pi / 180 / 3])
            camera_prims[i].set_local_pose(translation=pos, orientation=quat)
        for i in [1, 3, 5]:
            pos, _ = camera_prims[i].get_local_pose()
            quat = euler_angles_to_quat([0, math.pi / 2, east * math.pi / 180 / 3])
            camera_prims[i].set_local_pose(translation=pos, orientation=quat)

    def on_gamepad_event(message):
        command = message["method"]
        params = message["params"]
        if command == "set_mira_polar_delta" or command == "set_mira_cartesian_delta":
            # flip direction to match left arm orientation, and align with physical robot
            params["right"][2] = -params["right"][2]
            params["right"][0] = -params["right"][0]
            for i in range(4):
                left_pose[i] += params["left"][i]
                right_pose[i] += -params["right"][i]
            if args.debug:
                print(f"Update ({message['method']}):: Left: {left_pose}; Right: {right_pose}")
        elif command == "set_mira_pose":
            for i in range(6):
                left_pose[i] = params["left"][i]
                right_pose[i] = params["right"][i]
            if args.debug:
                print(f"Update ({message['method']}):: Left: {left_pose}; Right: {right_pose}")
        elif command == "set_camera_pose_delta":
            camera_pose[0] += params["north"]
            camera_pose[1] += params["east"]
            if args.debug:
                print(f"Update ({message['method']}):: north: {camera_pose[1]} east: {camera_pose[0]}")
        elif command == "set_left_gripper":
            grasps[0] = params * 1350
            if args.debug:
                print(f"Update ({message['method']}):: left-gripper: {grasps[0]}")
        elif command == "set_right_gripper":
            grasps[1] = params * 1350
            if args.debug:
                print(f"Update ({message['method']}):: right-gripper: {grasps[1]}")
        elif command == "set_mira_roll_delta":
            rotations[0] += params["left"]
            rotations[1] += params["right"]

    from patient.simulation.camera.sensor import CameraEx

    camera_pose = [0.0, 0.0]
    camera_prims = [SingleXFormPrim(p) for p in camera_paths]
    camera = CameraEx(
        prim_path=camera_prim_path,
        frequency=args.framerate,
        resolution=(args.width, args.height),
    )
    camera.initialize()

    left_rotation = left_arm_joint_apis[4].GetTargetPositionAttr().Get()
    right_rotation = right_arm_joint_apis[4].GetTargetPositionAttr().Get()
    rotations = [left_rotation, right_rotation]
    left_gripper = left_arm_joint_apis[5].GetTargetPositionAttr().Get() * (1350 / 20)
    right_gripper = -right_arm_joint_apis[5].GetTargetPositionAttr().Get() * (1350 / 40)
    grasps = [left_gripper, right_gripper]

    # holoscan app in async mode to consume camera source
    from patient.simulation.camera.app import App as CameraApp

    if args.encoder == "nvjpeg" and args.encoder_params is None:
        encoder_params = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nvjpeg_encoder_params.json")
    elif args.encoder == "nvc" and args.encoder_params is None:
        encoder_params = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nvc_encoder_params.json")
    elif os.path.isfile(args.encoder_params):
        encoder_params = args.encoder_params
    else:
        encoder_params = json.loads(args.encoder_params) if args.encoder_params else {}

    print(f"Encoder params: {encoder_params}")

    if os.path.isfile(encoder_params):
        with open(encoder_params) as f:
            encoder_params = json.load(f)

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
    if args.timeline_play:
        timeline = get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
    while simulation_app.is_running():
        update_arm_joints()
        update_camera_pose()
        simulation_app.update()

    f1.cancel()
    f2.cancel()
    simulation_app.close()


if __name__ == "__main__":
    main()
