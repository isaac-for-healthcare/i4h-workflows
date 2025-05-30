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


from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

import datetime
import math

import carb
import omni
import omni.appwindow
import omni.replicator.core as rep
import omni.usd
from i4h_asset_helper import BaseI4HAssets
from isaaclab.app import AppLauncher
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from mira_configs import (
    camera_key_map,
    camera_paths,
    camera_prim_path,
    key_map,
    lj_paths,
    max_camera_angle,
    rj_paths,
    snapshot_key,
    switch_key,
)
from PIL import Image
from pxr import UsdPhysics


class Assets(BaseI4HAssets):
    """Assets manager for the your workflow."""

    MIRA = "Robots/MIRA/mira-bipo-size-experiment-smoothing.usd"


def main():
    my_assets = Assets()
    usd_path = my_assets.MIRA
    omni.usd.get_context().open_stage(usd_path)

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
            if key == switch_key:
                current_arm[0] = "right" if current_arm[0] == "left" else "left"
                print(f"Switched to {current_arm[0]} arm control!")
                return True
            if key in key_map:
                idx, delta = key_map[key]
                if idx == "rotation":
                    if current_arm[0] == "left":
                        rotations[0] += delta
                    else:
                        rotations[1] += delta
                elif idx == "grasp":
                    if current_arm[0] == "left":
                        grasps[0] = max(0.0, min(1350.0, grasps[0] + delta))  # Clamp to [0, 1350]
                    else:
                        grasps[1] = max(0.0, min(1350.0, grasps[1] + delta))
                else:
                    (left_pose if current_arm[0] == "left" else right_pose)[idx] += delta
                return True
            if key in camera_key_map:
                idx, delta = camera_key_map[key]
                camera_pose[idx] += delta
                return True
            if key == snapshot_key:
                save_camera_image(camera_prim_path)
                return True
        return False

    def update_arm_joints():
        for i, api in enumerate(left_arm_joint_apis):
            if i == 4:  # rotation
                api.GetTargetPositionAttr().Set(rotations[0])
            elif i == 5:  # gripper
                api.GetTargetPositionAttr().Set(grasps[0] * (20 / 1350))
            else:
                api.GetTargetPositionAttr().Set(left_pose[i])
        # Right arm
        for i, api in enumerate(right_arm_joint_apis):
            if i == 4:  # rotation
                api.GetTargetPositionAttr().Set(rotations[1])
            elif i == 5:  # gripper
                api.GetTargetPositionAttr().Set(-grasps[1] * (40 / 1350))
            else:
                api.GetTargetPositionAttr().Set(right_pose[i])

    def update_camera_pose():
        north = max(-max_camera_angle, min(max_camera_angle, camera_pose[0]))
        east = max(-max_camera_angle, min(max_camera_angle, camera_pose[1]))
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

    stage = omni.usd.get_context().get_stage()
    left_arm_joint_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in lj_paths]
    right_arm_joint_apis = [UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(p), "angular") for p in rj_paths]
    camera_prims = [SingleXFormPrim(p) for p in camera_paths]

    left_pose = [api.GetTargetPositionAttr().Get() for api in left_arm_joint_apis[:4]]
    right_pose = [api.GetTargetPositionAttr().Get() for api in right_arm_joint_apis[:4]]
    left_rotation = left_arm_joint_apis[4].GetTargetPositionAttr().Get()
    right_rotation = right_arm_joint_apis[4].GetTargetPositionAttr().Get()
    rotations = [left_rotation, right_rotation]
    left_grasp = left_arm_joint_apis[5].GetTargetPositionAttr().Get() * (1350 / 20)
    right_grasp = -right_arm_joint_apis[5].GetTargetPositionAttr().Get() * (1350 / 40)
    grasps = [left_grasp, right_grasp]
    camera_pose = [0.0, 0.0]

    current_arm = ["left"]

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


if __name__ == "__main__":
    main()
    simulation_app.close()
