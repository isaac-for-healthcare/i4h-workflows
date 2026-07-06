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
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test G1/H2 Spread Tablecloth Environment")
parser.add_argument("--num_envs", type=int, default=1, help="number of environments")
parser.add_argument("--num_steps", type=int, default=5000, help="number of simulation steps")
parser.add_argument(
    "--physics_backend",
    type=str,
    default="newton",
    choices=["newton", "physx"],
    help="Physics backend preset: 'newton' (coupled MJWarp+VBD) or 'physx'.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="h2",
    choices=["h2", "g1"],
    help="Robot preset: 'h2' (H2 + Sharpa) or 'g1' (G129 + Inspire).",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="teleop",
    choices=["teleop", "joint"],
    help=(
        "Control mode: 'teleop' uses the Pink-IK action env (hold both wrists), "
        "'joint' uses the direct JointPositionAction env (hold current joints). "
        "Use 'joint' to isolate IK from the physics/NaN behaviour."
    ),
)
parser.add_argument(
    "--no_cloth",
    action="store_true",
    help="Drop the deformable cloth (and its inner-body reset event) to isolate the robot/physics.",
)
parser.add_argument(
    "--repro_cuda700",
    action="store_true",
    help=(
        "Reproduce the CUDA-700 crash fixed in commit 8e8f6860: run a render-only "
        "warmup (defers Newton's first CUDA-graph capture to the first env.step), "
        "then call env.sim.reset() + env.reset() right before the first step. With "
        "use_cuda_graph=True this poisons CUDA state and the first step crashes with "
        "'Warp CUDA error 700: an illegal memory access' inside narrow_phase / "
        "create_soft_contacts."
    ),
)
parser.add_argument(
    "--repro_warmup",
    type=int,
    default=60,
    help="Render-only frames before the poison env.sim.reset() (only used with --repro_cuda700).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# CUDA-700 diagnostics: enable BEFORE Kit starts a CUDA context so faults leave a
# stack trace and CUDA errors surface at the real offending kernel launch.
if args_cli.repro_cuda700:
    import faulthandler  # noqa: E402

    faulthandler.enable(all_threads=True)
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    print("[REPRO CUDA-700] faulthandler + CUDA_LAUNCH_BLOCKING=1 enabled", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# When this script is launched directly (``python .../examples/xxx.py``), Python
# puts this examples/ dir at sys.path[0], where examples/utils.py shadows the
# scripts-level ``utils`` package that the spread_tablecloth mdp imports
# (``from utils.logging import make_logger``). Drop the examples/ entry and make
# the rheo "scripts" root authoritative so ``utils`` resolves correctly.
_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _this_dir]
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import simulation.tasks.spread_tablecloth  # noqa: F401, E402
from simulation.tasks.spread_tablecloth.cloth_physics import select_physics_backend  # noqa: E402


# Robot/mode -> registered task id. The base ("-Joint") env uses a direct
# JointPositionAction; the "-Teleop" env wraps the same scene in a Pink-IK
# action space.
_ROBOT_TASK = {
    "h2": "Isaac-Spread-Tablecloth-H2-Sharpa",
    "g1": "Isaac-Spread-Tablecloth-G129-Inspire",
}
_MODE_SUFFIX = {"teleop": "-Teleop", "joint": "-Joint"}


def _as_torch(x):
    """Return a torch view of a sim data field regardless of backend.

    Newton-backed ``ArticulationData`` exposes warp arrays via a ``.torch``
    accessor (e.g. ``robot.data.joint_pos.torch``); some fields are raw
    ``wp.array``; PhysX returns plain torch tensors.
    """
    if hasattr(x, "torch"):
        return x.torch
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return x


def _nan_count(x) -> tuple[int, int]:
    t = _as_torch(x)
    return int(torch.isnan(t).sum().item()), int(t.numel())


def _nan_report(env: gym.Env, action: torch.Tensor, tag: str) -> None:
    """Print NaN counts for the robot state + the action being applied."""
    robot = env.unwrapped.scene["robot"]
    jp_n, jp_t = _nan_count(robot.data.joint_pos)
    bp_n, bp_t = _nan_count(robot.data.body_link_pose_w)
    ac_n, ac_t = _nan_count(action)
    print(
        f"[NAN {tag}] joint_pos={jp_n}/{jp_t}  body_pose={bp_n}/{bp_t}  action={ac_n}/{ac_t}",
        flush=True,
    )


def _cloth_report(env: gym.Env, tag: str) -> None:
    """Print the cloth nodal COM / z-extent and the cloth_inner body position so we
    can tell whether the cloth floats (z rises), slides (xy moves), and which way."""
    scene = env.unwrapped.scene
    parts = []
    try:
        cloth = scene["cloth"]
        npw = _as_torch(cloth.data.nodal_pos_w)[0]  # (V, 3)
        com = npw.mean(dim=0)
        zmin = npw[:, 2].min().item()
        zmax = npw[:, 2].max().item()
        parts.append(
            f"cloth COM=({com[0]:.3f},{com[1]:.3f},{com[2]:.3f}) z[{zmin:.3f},{zmax:.3f}]"
        )
    except Exception as e:  # noqa: BLE001
        parts.append(f"cloth N/A ({type(e).__name__})")
    try:
        inner = scene["cloth_inner"]
        rp = _as_torch(inner.data.root_pos_w)[0]
        parts.append(f"inner pos=({rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f})")
    except Exception as e:  # noqa: BLE001
        parts.append(f"inner N/A ({type(e).__name__})")
    print(f"[CLOTH {tag}] " + "  ".join(parts), flush=True)


def _build_joint_hold_action(env: gym.Env) -> torch.Tensor:
    """Construct a JointPositionAction that holds the robot at its current pose.

    The base env's ``joint_pos`` action term uses ``use_default_offset=False``
    and ``scale=1.0``, so the absolute joint targets equal the action. Feeding
    the robot's current joint positions (in the term's joint order) gives a
    trivial fixed point -- any NaN that appears is from the physics, not the
    controller.
    """
    robot = env.unwrapped.scene["robot"]
    term = env.unwrapped.action_manager.get_term("joint_pos")
    joint_ids = term._joint_ids  # slice(None) when all joints are driven
    jp = _as_torch(robot.data.joint_pos)
    return jp[:, joint_ids].clone()


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
    task_name = _ROBOT_TASK[args_cli.robot] + _MODE_SUFFIX[args_cli.action_mode]
    print(f"[INFO]: Task = {task_name}  (mode={args_cli.action_mode})", flush=True)

    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    selected_backend = select_physics_backend(env_cfg, args_cli.physics_backend)
    print(f"[INFO]: Physics backend = {selected_backend}", flush=True)

    if args_cli.no_cloth:
        # Remove the deformable cloth, its standalone inner rigid body, and the
        # event that resets that body, to isolate whether NaNs come from the
        # robot/physics or from the cloth coupling.
        env_cfg.scene.cloth = None
        if getattr(env_cfg.scene, "cloth_inner", None) is not None:
            env_cfg.scene.cloth_inner = None
        if getattr(env_cfg.events, "reset_cloth_inner", None) is not None:
            env_cfg.events.reset_cloth_inner = None
        print("[INFO]: Cloth + inner body removed (--no_cloth).", flush=True)

    if not args_cli.enable_cameras:
        # Drop camera observations AND the camera sensors from the scene, so RTX
        # is never instantiated (avoids native Kit/RTX crashes in headless runs).
        if getattr(env_cfg.observations, "camera_images", None) is not None:
            env_cfg.observations.camera_images = None
        for _attr in list(vars(env_cfg.scene).keys()):
            _val = getattr(env_cfg.scene, _attr)
            if _val is not None and "Camera" in type(_val).__name__:
                setattr(env_cfg.scene, _attr, None)
                print(f"[INFO]: Dropped camera sensor '{_attr}' (cameras disabled).", flush=True)

    env = gym.make(task_name, cfg=env_cfg).unwrapped

    obs, info = env.reset()
    # Frame the table + cloth: target the tabletop center (H2 cloth rests at
    # z=0.79; use ~0.70 for the G1 task) and view it from a front-right 3/4 angle.
    env.sim.set_camera_view(
        eye=[0.9, 0.7, 1.6],
        target=[-0.5, 0.0, 0.79],
    )
    print(f"Environment created: {task_name}")
    print(f"Observation keys: {list(obs['policy'].keys())}")
    for key, val in obs["policy"].items():
        print(f"  {key}: shape={val.shape}")
    if args_cli.enable_cameras and "camera_images" in obs:
        print(f"Camera image keys: {list(obs['camera_images'].keys())}")
        for key, val in obs["camera_images"].items():
            print(f"  {key}: shape={val.shape}")

    # Hold-pose action captured AFTER reset, so the target equals the robot's
    # actual initial pose. Recomputed once outside the loop -- nothing should
    # drift if the target stays fixed.
    if args_cli.action_mode == "teleop":
        hold_action = _build_hold_pose_action(env)
    else:
        hold_action = _build_joint_hold_action(env)

    import numpy as np  # noqa: E402

    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print(f"[ACTION {args_cli.action_mode}] shape={tuple(hold_action.shape)}", flush=True)
    if args_cli.action_mode == "teleop":
        print(f"  left_ee  (pos+quat xyzw) = {hold_action[0, 0:7].cpu().numpy()}", flush=True)
        print(f"  right_ee (pos+quat xyzw) = {hold_action[0, 7:14].cpu().numpy()}", flush=True)
        print(f"  hand/other dims (14:)    = {hold_action[0, 14:].cpu().numpy()}", flush=True)
    else:
        print(f"  joint targets            = {hold_action[0].cpu().numpy()}", flush=True)

    _nan_report(env, hold_action, "pre-step0")
    _cloth_report(env, "pre-step0")

    print("[INFO]: Entering step loop. Timing first 10 steps to spot stalls...")
    step = 0
    loop_start = time.perf_counter()

    # Reproduce CUDA-700 (see commit 8e8f6860 "resolve CUDA700 by removing env.sim.reset"):
    # with use_cuda_graph=True, the sequence
    #   render/wait -> env.sim.reset() -> env.reset() -> first env.step()
    # poisons Newton/CUDA state and the first step faults with
    # "Warp CUDA error 700: an illegal memory access" inside narrow_phase /
    # create_soft_contacts. The render-only warmup defers Newton's first CUDA-graph
    # capture to the first env.step() (RTX mode), which is required for the poison
    # to land on that specific step.
    if args_cli.repro_cuda700:
        print(
            f"[REPRO CUDA-700] render-only warmup {args_cli.repro_warmup} frames "
            "(defer CUDA-graph capture to first env.step)",
            flush=True,
        )
        for _ in range(args_cli.repro_warmup):
            env.sim.render()
        print("[REPRO CUDA-700] env.sim.reset() (the poison) -> env.reset() -> step", flush=True)
        env.sim.reset()

    env.reset()
    try:
        while simulation_app.is_running() and step < args_cli.num_steps:
            t0 = time.perf_counter()
            with torch.inference_mode():
                obs, reward, terminated, truncated, info = env.step(hold_action)
            step += 1
            dt = time.perf_counter() - t0

            if step == 1:
                _nan_report(env, hold_action, "post-step0")
                _cloth_report(env, "post-step0")
            elif step % 50 == 0:
                _nan_report(env, hold_action, f"step{step}")
                _cloth_report(env, f"step{step}")

            if step <= 10 or step % 50 == 0:
                elapsed = time.perf_counter() - loop_start
                print(f"[STEP {step:5d}] dt={dt*1000:7.1f} ms  total={elapsed:6.1f} s", flush=True)
    except (ReferenceError, RuntimeError):
        print(f"[INFO]: Simulation shut down after {step} steps.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
