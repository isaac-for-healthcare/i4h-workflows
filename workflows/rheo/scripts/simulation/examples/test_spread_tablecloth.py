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

"""Minimal test script to visualize the G1 spread_tablecloth environment."""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test G1 Spread Tablecloth Environment")
parser.add_argument("--num_envs", type=int, default=1, help="number of environments")
parser.add_argument("--num_steps", type=int, default=5000, help="number of simulation steps")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simulation.tasks.spread_tablecloth  # noqa: F401, E402


def _build_hold_pose_action(env: gym.Env) -> torch.Tensor:
    """Construct a Pink-IK action that asks the controller to keep both wrists
    at their current pose, with hand joints zeroed.

    The teleop env exposes a 38-D action whose first 14 entries are the left /
    right EE targets ``(pos_x, pos_y, pos_z, q_x, q_y, q_z, q_w)``. Feeding all
    zeros yields a degenerate quaternion ``(0, 0, 0, 0)`` which has no inverse
    -> the IK solver returns NaN every step (hence the spam of
    ``Solution to IK contains NaN`` in the terminal). Setting the targets to
    the wrists' current poses gives the solver a trivial fixed point.
    """
    robot = env.unwrapped.scene["robot"]
    left_idx = robot.find_bodies("left_wrist_yaw_link")[0][0]
    right_idx = robot.find_bodies("right_wrist_yaw_link")[0][0]

    # ``body_link_pose_w`` is a warp.array (dtype wp.transformf) with shape
    # (num_envs, num_bodies). ``wp.to_torch`` gives a zero-copy torch view
    # of shape (num_envs, num_bodies, 7) where the trailing dim is
    # (pos_x, pos_y, pos_z, q_x, q_y, q_z, q_w) -- xyzw, which already matches
    # the Pink IK action layout, so no quaternion reordering is needed.
    body_pose = wp.to_torch(robot.data.body_link_pose_w)
    left_ee = body_pose[:, left_idx, :]
    right_ee = body_pose[:, right_idx, :]

    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    action[:, 0:7] = left_ee
    action[:, 7:14] = right_ee
    return action


def main():
    task_name = "Isaac-Spread-Tablecloth-G129-Inspire-Teleop"

    env_cfg = parse_env_cfg(
        task_name, device=args_cli.device, num_envs=args_cli.num_envs,
    )
    if not args_cli.enable_cameras:
        env_cfg.observations.camera_images = None

    env = gym.make(task_name, cfg=env_cfg).unwrapped

    obs, info = env.reset()
    env.sim.set_camera_view(
        eye=[1.0, 0.5, 2.7],
        target=[-0.5, 0.0, 0.70],
    )
    print(f"Environment created: {task_name}")
    print(f"Observation keys: {list(obs['policy'].keys())}")
    for key, val in obs["policy"].items():
        print(f"  {key}: shape={val.shape}")
    if args_cli.enable_cameras and "camera_images" in obs:
        print(f"Camera image keys: {list(obs['camera_images'].keys())}")
        for key, val in obs["camera_images"].items():
            print(f"  {key}: shape={val.shape}")

    # Hold-pose action captured AFTER reset, so the IK target equals the
    # robot's actual initial wrist pose. Recomputed once outside the loop --
    # the wrists shouldn't drift if the IK target stays fixed.
    hold_action = _build_hold_pose_action(env)

    print("[INFO]: Entering step loop. Timing first 10 steps to spot stalls...")
    step = 0
    loop_start = time.perf_counter()
    env.reset()
    while simulation_app.is_running() and step < args_cli.num_steps:
        t0 = time.perf_counter()
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(hold_action)
        step += 1
        dt = time.perf_counter() - t0

        if step <= 10 or step % 50 == 0:
            elapsed = time.perf_counter() - loop_start
            print(f"[STEP {step:5d}] dt={dt*1000:7.1f} ms  total={elapsed:6.1f} s", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
