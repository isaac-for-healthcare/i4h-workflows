# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical live-authoring presets for commonly reused Healthcare assets."""

from __future__ import annotations

import math
from dataclasses import dataclass

from i4h_arena.assets.constants import (
    PANDA_USD,
    SCISSOR_TABLE_USD,
    SCISSOR_TRAY_USD,
    SCISSORS_USD,
    SURGICAL_TWEEZERS_USD,
    UNITREE_G1_29DOF_USD,
)


@dataclass(frozen=True, slots=True)
class AuthoringCameraPreset:
    """Camera that a live asset preview must carry to match its embodiment."""

    relative_parent_path: str
    prim_name: str
    alias: str
    position_m: tuple[float, float, float]
    rotation_opengl_xyzw: tuple[float, float, float, float]
    focal_length: float
    focus_distance: float
    horizontal_aperture: float
    resolution: tuple[int, int]
    clipping_range_m: tuple[float, float]


@dataclass(frozen=True, slots=True)
class AuthoringEmbodimentPreset:
    """Reusable runtime facts a coding agent needs for one embodiment."""

    module: str
    registry_name: str
    manifest_name: str
    action_space: str
    dof: int
    gripper: bool
    control_hz: float
    robot_name: str
    runtime_prim_path: str
    joint_width: int | None
    gripper_index: int | None
    camera_aliases: tuple[tuple[str, str], ...]
    contact_body_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AuthoringAssetPreset:
    """Reusable scale and physics facts for one authored USD."""

    usd_path: str
    scale: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    canonical_size_m: tuple[float, float, float]
    bounds_z_m: tuple[float, float]
    physics: str
    mass_kg: float | None = None
    attached_cameras: tuple[AuthoringCameraPreset, ...] = ()
    embodiment: AuthoringEmbodimentPreset | None = None


AUTHORING_ASSETS: dict[str, AuthoringAssetPreset] = {
    "franka_ultrasound": AuthoringAssetPreset(
        usd_path=PANDA_USD,
        scale=(1.0, 1.0, 1.0),
        rotation_deg=(0.0, 0.0, 0.0),
        canonical_size_m=(0.378920, 0.259306, 1.114663),
        bounds_z_m=(0.0, 1.114663),
        physics="articulation",
        embodiment=AuthoringEmbodimentPreset(
            module="i4h_arena.embodiments.franka",
            registry_name="franka_ultrasound",
            manifest_name="panda",
            action_space="ee_pose",
            dof=6,
            gripper=False,
            control_hz=50.0,
            robot_name="robot",
            runtime_prim_path="Robot",
            joint_width=7,
            gripper_index=None,
            camera_aliases=(("wrist", "wrist_camera"),),
        ),
    ),
    "surgical_table": AuthoringAssetPreset(
        usd_path=SCISSOR_TABLE_USD,
        scale=(0.7, 0.7, 0.52),
        rotation_deg=(0.0, 0.0, 0.0),
        canonical_size_m=(1.28016, 0.80010, 0.475488),
        bounds_z_m=(-0.237744, 0.237744),
        physics="static",
    ),
    "scissors": AuthoringAssetPreset(
        usd_path=SCISSORS_USD,
        scale=(0.006, 0.0065, 0.012),
        rotation_deg=(0.0, 0.0, 90.0),
        canonical_size_m=(0.123452, 0.034816, 0.023982),
        bounds_z_m=(-0.013297, 0.010685),
        physics="rigid",
        mass_kg=0.15,
    ),
    "tweezers": AuthoringAssetPreset(
        usd_path=SURGICAL_TWEEZERS_USD,
        scale=(1.0, 1.0, 1.0),
        rotation_deg=(0.0, 0.0, 90.0),
        canonical_size_m=(0.159888, 0.020162, 0.014943),
        bounds_z_m=(-0.007472, 0.007472),
        physics="rigid",
        mass_kg=0.05,
    ),
    "surgical_tray": AuthoringAssetPreset(
        usd_path=SCISSOR_TRAY_USD,
        scale=(0.7, 0.7, 0.18),
        rotation_deg=(0.0, 0.0, 90.0),
        canonical_size_m=(0.207644, 0.138490, 0.026345),
        bounds_z_m=(-0.013172, 0.013172),
        physics="static",
    ),
    "g1": AuthoringAssetPreset(
        usd_path=UNITREE_G1_29DOF_USD,
        scale=(1.0, 1.0, 1.0),
        rotation_deg=(0.0, 0.0, 0.0),
        canonical_size_m=(0.495829, 0.371414, 1.322845),
        bounds_z_m=(-0.792273, 0.530573),
        physics="articulation",
        embodiment=AuthoringEmbodimentPreset(
            module="i4h_arena.embodiments.g1",
            registry_name="g1_wbc_joint",
            manifest_name="g1",
            action_space="joint_position",
            dof=50,
            gripper=False,
            control_hz=30.0,
            robot_name="robot",
            runtime_prim_path="Robot",
            joint_width=43,
            gripper_index=None,
            camera_aliases=(("head", "robot_head_cam"),),
            contact_body_names=(
                "pelvis",
                "left_hip_pitch_link",
                "left_hip_roll_link",
                "left_hip_yaw_link",
                "left_knee_link",
                "left_ankle_pitch_link",
                "left_ankle_roll_link",
                "right_hip_pitch_link",
                "right_hip_roll_link",
                "right_hip_yaw_link",
                "right_knee_link",
                "right_ankle_pitch_link",
                "right_ankle_roll_link",
                "waist_yaw_link",
                "waist_roll_link",
                "torso_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_wrist_pitch_link",
                "left_wrist_yaw_link",
                "left_hand_palm_link",
                "left_hand_index_0_link",
                "left_hand_index_1_link",
                "left_hand_middle_0_link",
                "left_hand_middle_1_link",
                "left_hand_thumb_0_link",
                "left_hand_thumb_1_link",
                "left_hand_thumb_2_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
                "right_wrist_roll_link",
                "right_wrist_pitch_link",
                "right_wrist_yaw_link",
                "right_hand_palm_link",
                "right_hand_index_0_link",
                "right_hand_index_1_link",
                "right_hand_middle_0_link",
                "right_hand_middle_1_link",
                "right_hand_thumb_0_link",
                "right_hand_thumb_1_link",
                "right_hand_thumb_2_link",
            ),
        ),
        attached_cameras=(
            AuthoringCameraPreset(
                # The live preview nests the referenced USD below ``Asset``.
                # The coding agent replaces this preview with the registered
                # G1 embodiment, whose equivalent prim is
                # ``Robot/head_link/RobotHeadCam``.
                relative_parent_path="Asset/head_link",
                prim_name="RobotHeadCam",
                alias="head",
                position_m=(0.04485, 0.0, 0.35325),
                # Stock G1CameraCfg's ROS quaternion
                # (-0.62721, 0.62721, -0.32651, 0.32651), converted to the
                # OpenGL convention required by UsdGeom.Camera.
                rotation_opengl_xyzw=(0.32651, -0.32651, -0.62721, 0.62721),
                focal_length=15.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                resolution=(640, 480),
                clipping_range_m=(0.1, 5.0),
            ),
        ),
    ),
}


def authoring_asset(name: str) -> AuthoringAssetPreset:
    """Return a named preset or a useful error listing known assets."""
    try:
        return AUTHORING_ASSETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown authoring asset {name!r}; choose from {sorted(AUTHORING_ASSETS)}") from exc


def spawn_pose_facing_rectangle(
    *,
    center_xy: tuple[float, float],
    half_extents_xy: tuple[float, float],
    edge_distance_m: float,
    base_z_m: float,
    side: str = "-y",
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Place a +X-forward embodiment outside one rectangle edge, facing its center."""
    if edge_distance_m < 0.0:
        raise ValueError("edge_distance_m must be non-negative")
    if any(value <= 0.0 for value in half_extents_xy):
        raise ValueError("half_extents_xy must be positive")
    cx, cy = center_xy
    hx, hy = half_extents_xy
    placements = {
        "-x": ((cx - hx - edge_distance_m, cy, base_z_m), 0.0),
        "+x": ((cx + hx + edge_distance_m, cy, base_z_m), math.pi),
        "-y": ((cx, cy - hy - edge_distance_m, base_z_m), math.pi / 2.0),
        "+y": ((cx, cy + hy + edge_distance_m, base_z_m), -math.pi / 2.0),
    }
    try:
        position, yaw = placements[side]
    except KeyError as exc:
        raise ValueError("side must be one of '-x', '+x', '-y', '+y'") from exc
    return position, (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
