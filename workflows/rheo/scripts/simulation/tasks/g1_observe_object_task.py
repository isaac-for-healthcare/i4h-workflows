# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class ObserveObjectTask(G1LocomanipPickAndPlaceTask):
    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([0.5, -0.5, 1.5]),
        )

    def get_events_cfg(self):
        return I4HEventsCfg(
            pick_up_object=self.pick_up_object,
            destination_cart=self.destination_bin,
        )

    def get_termination_cfg(self):
        return TerminationsCfg(
            success=TerminationTermCfg(func=mdp_isaac_lab.time_out),
            object_dropped=TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={
                    "minimum_height": self.background_scene.object_min_z,
                    "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
                },
            ),
        )

    def modify_env_cfg(self, env_cfg):
        """Modify environment configuration for better rendering quality and add room camera."""
        # Call parent class method if it exists
        if hasattr(super(), "modify_env_cfg"):
            env_cfg = super().modify_env_cfg(env_cfg)

        # Set rendering configuration for better visual quality
        env_cfg.sim.render.rendering_mode = "quality"
        env_cfg.sim.render.antialiasing_mode = "DLAA"

        # Enable translucency rendering for transparent/glass objects
        env_cfg.sim.render.enable_translucency = True
        # Use carb_settings for fractionalCutoutOpacity (for transparent object rendering)
        if env_cfg.sim.render.carb_settings is None:
            env_cfg.sim.render.carb_settings = {}
        env_cfg.sim.render.carb_settings["rtx.raytracing.fractionalCutoutOpacity"] = True

        # Add room camera to scene
        self._add_room_camera(env_cfg)

        # Add room camera observation
        self._add_room_camera_observation(env_cfg)

        return env_cfg

    def _add_room_camera(self, env_cfg):
        """Add a fixed room camera to the scene."""
        # Create room camera configuration
        room_camera_cfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/cart/room_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.8), rot=(0.0, 0.707, -0.707, 0.0), convention="ros"),
        )

        # Add camera to scene configuration
        if not hasattr(env_cfg.scene, "room_camera"):
            env_cfg.scene.room_camera = room_camera_cfg

    def _add_room_camera_observation(self, env_cfg):
        """Add room camera observations to the observation configuration."""
        # Check if observations configuration exists
        if not hasattr(env_cfg, "observations"):
            return

        # Add room camera RGB observation to policy group
        if hasattr(env_cfg.observations, "policy"):
            # Add RGB image observation
            room_cam_rgb_obs = ObservationTermCfg(
                func=mdp_isaac_lab.image,
                params={"sensor_cfg": SceneEntityCfg("room_camera"), "data_type": "rgb", "normalize": False},
            )

            # Add to policy observations if not already present
            if not hasattr(env_cfg.observations.policy, "room_camera"):
                env_cfg.observations.policy.room_camera = room_cam_rgb_obs


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING


@configclass
class I4HEventsCfg:
    """Event configuration for i4h-workflow task - adds box and cart reset events."""

    reset_destination_cart_pose: EventTermCfg = MISSING
    reset_pick_up_object_pose: EventTermCfg = MISSING

    def __init__(
        self,
        pick_up_object: Asset,
        destination_cart: Asset,
    ):
        """Initialize reset events for both pick_up_object and destination_cart.

        Args:
            pick_up_object: The object to pick up (e.g., surgical_tray)
            destination_cart: The cart to push (cart)
        """

        object_initial_pose = pick_up_object.get_initial_pose()
        self.reset_pick_up_object_pose = EventTermCfg(
            func=set_object_pose,
            mode="reset",
            params={
                "pose": object_initial_pose,
                "asset_cfg": SceneEntityCfg(pick_up_object.name),
            },
        )

        cart_initial_pose = destination_cart.get_initial_pose()
        self.reset_destination_cart_pose = EventTermCfg(
            func=set_object_pose,
            mode="reset",
            params={
                "pose": cart_initial_pose,
                "asset_cfg": SceneEntityCfg(destination_cart.name),
            },
        )
