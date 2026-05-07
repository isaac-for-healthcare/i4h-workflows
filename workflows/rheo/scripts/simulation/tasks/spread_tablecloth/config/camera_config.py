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

"""Camera presets for the `spread_tablecloth` task."""

from typing import Optional, Sequence, Tuple

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


@configclass
class CameraBaseCfg:
    """Camera base configuration class.

    Provides default configuration for different types of cameras with scene-specific parameter customization.
    """

    @classmethod
    def get_camera_config(
        cls,
        prim_path: str = "/World/envs/env_.*/Robot/d435_link/front_cam",
        update_period: float = 0.02,
        height: int = 480,
        width: int = 640,
        focal_length: float = 7.6,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.0,
        clipping_range: Tuple[float, float] = (0.1, 1.0e5),
        pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rot_offset: Tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5),
        data_types: Optional[Sequence[str]] = None,
    ) -> CameraCfg:
        """Get a pinhole camera configuration.

        Args:
            prim_path: the path of the camera in the scene
            update_period: update period (seconds)
            height: image height (pixels)
            width: image width (pixels)
            focal_length: focal length
            focus_distance: focus distance
            horizontal_aperture: horizontal aperture
            clipping_range: clipping range (near clipping plane, far clipping plane)
            pos_offset: position offset (x, y, z)
            rot_offset: rotation offset quaternion
            data_types: data type list

        Returns:
            CameraCfg: camera configuration
        """
        if data_types is None:
            data_types = ("rgb",)

        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=list(data_types),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range,
            ),
            offset=CameraCfg.OffsetCfg(pos=pos_offset, rot=rot_offset, convention="ros"),
        )


@configclass
class CameraPresets:
    """Camera preset configuration collection."""

    @classmethod
    def g1_front_camera(cls, **overrides) -> CameraCfg:
        """Front camera configuration."""
        params = {"focal_length": 12.0}
        params.update(overrides)
        return CameraBaseCfg.get_camera_config(**params)

    @classmethod
    def left_dex3_wrist_camera(cls, **overrides) -> CameraCfg:
        """Left wrist camera configuration (Dex3 hand)."""
        params = {
            "prim_path": "/World/envs/env_.*/Robot/left_hand_camera_base_link/left_wrist_camera",
            "height": 480,
            "width": 640,
            "update_period": 0.02,
            "data_types": ["rgb"],
            "focal_length": 12.0,
            "focus_distance": 400.0,
            "horizontal_aperture": 20.0,
            "clipping_range": (0.1, 1.0e5),
            "pos_offset": (-0.04012, -0.07441, 0.15711),
            "rot_offset": (0.00539, 0.86024, 0.0424, 0.50809),
        }
        params.update(overrides)
        return CameraBaseCfg.get_camera_config(**params)

    @classmethod
    def right_dex3_wrist_camera(cls, **overrides) -> CameraCfg:
        """Right wrist camera configuration (Dex3 hand)."""
        params = {
            "prim_path": "/World/envs/env_.*/Robot/right_hand_camera_base_link/right_wrist_camera",
            "height": 480,
            "width": 640,
            "update_period": 0.02,
            "data_types": ["rgb"],
            "focal_length": 12.0,
            "focus_distance": 400.0,
            "horizontal_aperture": 20.0,
            "clipping_range": (0.1, 1.0e5),
            "pos_offset": (-0.04012, 0.07441, 0.15711),
            "rot_offset": (0.00539, 0.86024, 0.0424, 0.50809),
        }
        params.update(overrides)
        return CameraBaseCfg.get_camera_config(**params)

    @classmethod
    def left_inspire_wrist_camera(cls, **overrides) -> CameraCfg:
        """Left wrist camera configuration (Inspire hand)."""
        params = {
            "prim_path": "/World/envs/env_.*/Robot/left_hand_camera_base_link/left_wrist_camera",
            "height": 480,
            "width": 640,
            "update_period": 0.02,
            "data_types": ["rgb"],
            "focal_length": 12.0,
            "focus_distance": 400.0,
            "horizontal_aperture": 20.0,
            "clipping_range": (0.1, 1.0e5),
            "pos_offset": (-0.04012, -0.07441, 0.15711),
            "rot_offset": (0.86024, 0.0424, 0.50809, 0.00539),
        }
        params.update(overrides)
        return CameraBaseCfg.get_camera_config(**params)

    @classmethod
    def right_inspire_wrist_camera(cls, **overrides) -> CameraCfg:
        """Right wrist camera configuration (Inspire hand)."""
        params = {
            "prim_path": "/World/envs/env_.*/Robot/right_hand_camera_base_link/right_wrist_camera",
            "height": 480,
            "width": 640,
            "update_period": 0.02,
            "data_types": ["rgb"],
            "focal_length": 12.0,
            "focus_distance": 400.0,
            "horizontal_aperture": 20.0,
            "clipping_range": (0.1, 1.0e5),
            "pos_offset": (-0.04012, 0.07441, 0.15711),
            "rot_offset": (0.86024, 0.0424, 0.50809, 0.00539),
        }
        params.update(overrides)
        return CameraBaseCfg.get_camera_config(**params)
