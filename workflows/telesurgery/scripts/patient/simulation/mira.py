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

from common.utils import strtobool
from i4h_asset_helper import BaseI4HAssets
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
parser.add_argument("--fps", type=int, default=30, help="fps")
parser.add_argument("--width", type=int, default=1920, help="width")
parser.add_argument("--height", type=int, default=1080, help="height")
parser.add_argument("--jpeg_compress", type=strtobool, default=True, help="compress jpeg")
parser.add_argument("--jpeg_quality", type=int, default=90, help="jpeg quality")
parser.add_argument("--domain_id", type=int, default=779, help="dds domain id")
parser.add_argument("--topic", type=str, default="telesurgery/robot_camera/rgb", help="dds topic name")

args = parser.parse_args()

class Assets(BaseI4HAssets):
    """Assets manager for the your workflow."""

    MIRA_ARM = "Robots/MIRA/mira-bipo-size-experiment-smoothing.usd"


def main():
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app
    my_assets = Assets()
    usd_path = my_assets.MIRA_ARM

    # Import required modules after app is launched
    import carb
    import omni.appwindow
    import omni.usd
    from pxr import UsdPhysics

    omni.usd.get_context().open_stage(usd_path)

    # --- Define joint paths directly using strings ---
    # Define the common root path for the robot model in the USD stage
    robot_usd_root_path = "/World/A5_GUI_MODEL/A5_GUI_MODEL_001"

    # Define base paths for left and right arms
    left_arm_base_usd_path = f"{robot_usd_root_path}/ASM_L654321"
    right_arm_base_usd_path = f"{robot_usd_root_path}/ASM_R654321"

    LJ_PATHS = [
        f"{left_arm_base_usd_path}/LJ1/LJ1_joint",
        f"{left_arm_base_usd_path}/ASM_L65432/LJ2/LJ2_joint",
        f"{left_arm_base_usd_path}/ASM_L65432/ASM_L6543/LJ3/LJ3_joint",
        f"{left_arm_base_usd_path}/ASM_L65432/ASM_L6543/ASM_L654/LJ4/LJ4_joint",
        f"{left_arm_base_usd_path}/ASM_L65432/ASM_L6543/ASM_L654/ASM_L65/LJ5/LJ5_joint",
        f"{left_arm_base_usd_path}/ASM_L65432/ASM_L6543/ASM_L654/ASM_L65/ASM_L61/LJ6/LJ6_1_joint",
    ]
    RJ_PATHS = [
        f"{right_arm_base_usd_path}/RJ1/RJ1_joint",
        f"{right_arm_base_usd_path}/ASM_R65432/RJ2/RJ2_joint",
        f"{right_arm_base_usd_path}/ASM_R65432/ASM_R6543/RJ3/RJ3_joint",
        f"{right_arm_base_usd_path}/ASM_R65432/ASM_R6543/ASM_R654/RJ4/RJ4_joint",
        f"{right_arm_base_usd_path}/ASM_R65432/ASM_R6543/ASM_R654/ASM_R65/RJ5/RJ5_joint",
        f"{right_arm_base_usd_path}/ASM_R65432/ASM_R6543/ASM_R654/ASM_R65/ASM_R6/RJ6/RJ6_joint",
    ]

    stage = omni.usd.get_context().get_stage()
    LJ_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in LJ_PATHS]
    RJ_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in RJ_PATHS]

    # --- Initial joint positions ---
    left_pose = [0.0] * 6
    right_pose = [0.0] * 6

    # --- Key mapping: use a single set of keys, switch arm with SPACE ---
    KEY_MAP = {
        "I": (0, 0.1),  # X+
        "K": (0, -0.1),  # X-
        "J": (1, 0.1),  # Y+
        "L": (1, -0.1),  # Y-
        "U": (2, 0.1),  # Z+
        "O": (2, -0.1),  # Z-
        "Z": (3, 0.1),  # elbow+
        "X": (3, -0.1),  # elbow-
        "C": (4, 0.1),  # roll+
        "V": (4, -0.1),  # roll-
        "B": (5, 0.1),  # gripper+
        "N": (5, -0.1),  # gripper-
    }
    SWITCH_KEY = "SPACE"  # Key to switch between left and right arm

    current_arm = ["left"]  # Use a list to allow modification in closure

    def on_keyboard_event(event, *args):
        # Handle keyboard events and update joint positions
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key == SWITCH_KEY:
                # Switch the current arm being controlled
                current_arm[0] = "right" if current_arm[0] == "left" else "left"
                print(f"Switched to {current_arm[0]} arm control!")
                return True
            if key in KEY_MAP:
                idx, delta = KEY_MAP[key]
                if current_arm[0] == "left":
                    left_pose[idx] += delta
                else:
                    right_pose[idx] += delta
                print(
                    f"Key pressed: {key} | {current_arm[0]}_pose: {left_pose if current_arm[0]=='left' else right_pose}"
                )
                return True  # Consume the event, block Isaac Sim default
        return False  # Not our key, let Isaac Sim handle

    # Subscribe to keyboard events
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(
        keyboard, lambda event, *args: on_keyboard_event(event, *args)
    )

    from patient.simulation.annotators.camera import CameraEx

    camera_prim_path = ("/World/A5_GUI_MODEL/A5_GUI_MODEL_001/C_ASM_6543210/C_ASM_654321/C_ASM_65432/C_ASM_6543/"
                        "C_ASM_654/C_ASM_65/C_ASM_6/Camera_Tip/Camera")
    camera = CameraEx(prim_path=camera_prim_path, frequency=args.fps, resolution=(args.width, args.height))
    camera.initialize()

    # holoscan app in async mode to consume camera source
    from patient.simulation.apps.camera import App

    camera_app = App(
        width=args.width,
        height=args.height,
        dds_domain_id=args.domain_id,
        dds_topic=args.topic,
        jpeg_compress=args.jpeg_compress,
        jpeg_quality=args.jpeg_quality,
    )
    camera_app.run_async()
    camera.set_callback(camera_app.on_new_frame)

    # Main Isaac Sim loop
    while simulation_app.is_running():
        camera.set_callback(camera_app.on_new_frame)

        # Per-frame logic: update joint positions
        for i, api in enumerate(LJ_apis):
            api.GetTargetPositionAttr().Set(left_pose[i])
        for i, api in enumerate(RJ_apis):
            api.GetTargetPositionAttr().Set(right_pose[i])

        simulation_app.update()

    # Cleanup after app closes
    keyboard_sub.unsubscribe()
    simulation_app.close()


if __name__ == "__main__":
    main()
