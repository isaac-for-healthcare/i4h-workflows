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

import math

from i4h_asset_helper import BaseI4HAssets
from isaaclab.app import AppLauncher
from PIL import Image


class Assets(BaseI4HAssets):
    """Assets manager for the your workflow."""

    MIRA_ARM = "Robots/MIRA/mira-bipo-size-experiment-smoothing.usd"


def main():
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app
    my_assets = Assets()
    usd_path = my_assets.MIRA_ARM

    # Import Isaac/Omni modules after app launch
    import datetime

    import carb
    import omni
    import omni.appwindow
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.rotations import euler_angles_to_quat
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
    CAMERA_PATHS = [
        f"{camera_base}/C_ASM_654321",
        f"{camera_base}/C_ASM_654321/C_ASM_65432",
        f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543",
        f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654",
        f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65",
        f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65/C_ASM_6",
    ]
    camera_prim_path = f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65/C_ASM_6/Camera_Tip/Camera"
    MAX_CAMERA_ANGLE = 70

    stage = omni.usd.get_context().get_stage()
    left_arm_joint_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in LJ_PATHS]
    right_arm_joint_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in RJ_PATHS]
    camera_prims = [SingleXFormPrim(p) for p in CAMERA_PATHS]

    left_pose = [0.0] * 6
    right_pose = [0.0] * 6
    camera_pose = [0.0, 0.0]  # [north, east]

    KEY_MAP = {
        "I": (0, 0.1),
        "K": (0, -0.1),
        "J": (1, 0.1),
        "L": (1, -0.1),
        "U": (2, 0.1),
        "O": (2, -0.1),
        "Z": (3, 0.1),
        "X": (3, -0.1),
        "C": (4, 0.1),
        "V": (4, -0.1),
        "B": (5, 0.1),
        "N": (5, -0.1),
    }
    CAMERA_KEY_MAP = {
        "UP": (1, 1.0),  # Y (Pan) + (Pan Right)
        "DOWN": (1, -1.0),  # Y (Pan) - (Pan Left)
        "LEFT": (0, -1.0),  # X (Tilt) - (Tilt Down)
        "RIGHT": (0, 1.0),  # X (Tilt) + (Tilt Up)
    }
    SWITCH_KEY = "Y"
    SNAPSHOT_KEY = "F12"
    current_arm = ["left"]

    def save_camera_image(camera_prim_path):
        if not hasattr(save_camera_image, "render_product"):
            save_camera_image.render_product = rep.create.render_product(camera_prim_path, resolution=(1280, 720))
        if not hasattr(save_camera_image, "annotator"):
            save_camera_image.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            save_camera_image.annotator.attach(save_camera_image.render_product)
        rgb_data = save_camera_image.annotator.get_data()
        if rgb_data is None or rgb_data.size == 0:
            print("No image data available. Make sure simulation is running and camera is active.")
            return
        if rgb_data.ndim == 4 and rgb_data.shape[0] == 1:
            rgb_data = rgb_data[0]
        if rgb_data.shape[-1] == 4:
            rgb_data = rgb_data[..., :3]
        img = Image.fromarray(rgb_data, "RGB")
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img.save(f"camera_snapshot_{now}.png")
        print(f"Camera snapshot saved as camera_snapshot_{now}.png")

    def on_keyboard_event(event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key == SWITCH_KEY:
                current_arm[0] = "right" if current_arm[0] == "left" else "left"
                print(f"Switched to {current_arm[0]} arm control!")
                return True
            if key in KEY_MAP:
                idx, delta = KEY_MAP[key]
                (left_pose if current_arm[0] == "left" else right_pose)[idx] += delta
                return True
            if key in CAMERA_KEY_MAP:
                idx, delta = CAMERA_KEY_MAP[key]
                camera_pose[idx] += delta
                return True
            if key == SNAPSHOT_KEY:
                save_camera_image(camera_prim_path)
                return True
        return False

    def update_arm_joints():
        for i, api in enumerate(left_arm_joint_apis):
            api.GetTargetPositionAttr().Set(left_pose[i])
        for i, api in enumerate(right_arm_joint_apis):
            api.GetTargetPositionAttr().Set(right_pose[i])

    def update_camera_pose():
        north = max(-MAX_CAMERA_ANGLE, min(MAX_CAMERA_ANGLE, camera_pose[0]))
        east = max(-MAX_CAMERA_ANGLE, min(MAX_CAMERA_ANGLE, camera_pose[1]))
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

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(
        keyboard, lambda event, *args: on_keyboard_event(event, *args)
    )

    while simulation_app.is_running():
        update_arm_joints()
        update_camera_pose()
        simulation_app.update()

    keyboard_sub.unsubscribe()
    simulation_app.close()


if __name__ == "__main__":
    main()
