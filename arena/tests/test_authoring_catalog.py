# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from i4h_arena.assets.authoring_catalog import AUTHORING_ASSETS, authoring_asset, spawn_pose_facing_rectangle


def test_known_healthcare_assets_have_canonical_metric_sizes() -> None:
    assert set(AUTHORING_ASSETS) == {
        "franka_ultrasound",
        "g1",
        "scissors",
        "surgical_table",
        "surgical_tray",
        "tweezers",
    }
    for preset in AUTHORING_ASSETS.values():
        assert all(value > 0.0 for value in preset.scale)
        assert all(value > 0.0 for value in preset.canonical_size_m)
        assert preset.bounds_z_m[0] < preset.bounds_z_m[1]


def test_unknown_asset_lists_available_presets() -> None:
    with pytest.raises(KeyError, match="surgical_table"):
        authoring_asset("unknown")


def test_g1_live_preset_carries_the_standard_head_camera() -> None:
    preset = authoring_asset("g1")
    camera = preset.attached_cameras[0]
    embodiment = preset.embodiment

    assert camera.alias == "head"
    assert camera.relative_parent_path == "Asset/head_link"
    assert camera.prim_name == "RobotHeadCam"
    assert camera.resolution == (640, 480)
    assert embodiment is not None
    assert embodiment.registry_name == "g1_wbc_joint"
    assert embodiment.manifest_name == "g1"
    assert embodiment.runtime_prim_path == "Robot"
    assert embodiment.dof == 50
    assert not embodiment.gripper
    assert embodiment.camera_aliases == (("head", "robot_head_cam"),)
    assert embodiment.contact_body_names[0] == "pelvis"
    assert len(embodiment.contact_body_names) == len(set(embodiment.contact_body_names))
    assert "left_ankle_roll_link" in embodiment.contact_body_names
    assert "right_hand_thumb_2_link" in embodiment.contact_body_names
    assert "imu_in_pelvis" not in embodiment.contact_body_names


def test_franka_ultrasound_preset_matches_runtime_contract() -> None:
    preset = authoring_asset("franka_ultrasound")
    embodiment = preset.embodiment

    assert preset.physics == "articulation"
    assert embodiment is not None
    assert embodiment.registry_name == "franka_ultrasound"
    assert embodiment.manifest_name == "panda"
    assert embodiment.action_space == "ee_pose"
    assert embodiment.runtime_prim_path == "Robot"
    assert embodiment.dof == 6
    assert embodiment.joint_width == 7
    assert not embodiment.gripper
    assert embodiment.camera_aliases == (("wrist", "wrist_camera"),)


def test_spawn_pose_uses_edge_distance_and_faces_rectangle_center() -> None:
    position, rotation = spawn_pose_facing_rectangle(
        center_xy=(1.0, 2.0),
        half_extents_xy=(0.6, 0.4),
        edge_distance_m=2.0,
        base_z_m=0.8,
        side="-y",
    )

    assert position == pytest.approx((1.0, -0.4, 0.8))
    assert rotation == pytest.approx((0.0, 0.0, 2**-0.5, 2**-0.5))


def test_spawn_pose_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="side"):
        spawn_pose_facing_rectangle(
            center_xy=(0.0, 0.0),
            half_extents_xy=(0.5, 0.5),
            edge_distance_m=1.0,
            base_z_m=0.8,
            side="diagonal",
        )
