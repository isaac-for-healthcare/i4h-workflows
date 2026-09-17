# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.cli import build_parser
from i4h_arena.scenes.base import load_scene
from i4h_common.config import get_robot_config


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_xr_is_configured_before_asset_imports(robot):
    args = build_parser().parse_args(["--workflow", f"spread_tablecloth_{robot}", "--mode", "teleop"])
    scene = load_scene(f"{robot}_tablecloth", args, register_assets=False)
    scene.configure_args(args)
    assert args.xr and args.enable_pinocchio and args.presets == "newton"
    assert args.usd_thread_limit == 1
    args.headless = True
    with pytest.raises(ValueError, match="visible Kit"):
        scene.configure_args(args)


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_rejects_unsupported_cloth_backend_before_kit(robot):
    args = build_parser().parse_args(["--workflow", f"spread_tablecloth_{robot}", "--presets", "physx"])
    scene = load_scene(f"{robot}_tablecloth", args, register_assets=False)
    with pytest.raises(ValueError, match="requires --presets newton"):
        scene.configure_args(args)


@pytest.mark.parametrize("robot,width", [("g1", 38), ("h2", 58)])
def test_seed_uses_live_wrist_poses_and_explicit_finger_order(robot, width):
    args = build_parser().parse_args(
        ["--workflow", f"spread_tablecloth_{robot}", "--mode", "replay", "--device", "cpu"]
    )
    scene = load_scene(f"{robot}_tablecloth", args, register_assets=False)
    robot_cfg = get_robot_config(scene.spec.embodiment)
    names = list(reversed(robot_cfg.joint_names))
    hand_names = [name.removesuffix(".pos") for name in robot_cfg.action_names[14:]]
    poses = np.array([[[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]]], dtype=np.float32)
    data = SimpleNamespace(
        body_link_pose_w=poses,
        body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        joint_names=names,
        joint_pos=np.arange(len(names), dtype=np.float32)[None, :],
    )

    class FakeScene(dict):
        env_origins = np.array([[0.5, 1, 0]])

    env = SimpleNamespace(
        action_space=SimpleNamespace(shape=(1, width)),
        unwrapped=SimpleNamespace(
            num_envs=1,
            scene=FakeScene(robot=SimpleNamespace(data=data)),
            cfg=SimpleNamespace(actions=SimpleNamespace(pink_ik_cfg=SimpleNamespace(hand_joint_names=hand_names))),
        ),
    )
    action = scene.make_actuation(env).numpy()
    np.testing.assert_allclose(action[0, :7], [0.5, 1, 3, 0, 0, 0, 1])
    np.testing.assert_allclose(action[0, 7:14], [3.5, 4, 6, 0, 0, 1, 0])
    np.testing.assert_array_equal(action[0, 14:], [names.index(name) for name in hand_names])
