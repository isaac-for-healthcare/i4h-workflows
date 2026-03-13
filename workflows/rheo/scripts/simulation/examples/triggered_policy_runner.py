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


import enum
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tqdm
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.policy_runner_cli import get_isaaclab_arena_environments_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli
from scripts.utils.keyboard_env_reseter import KeyboardHandler, disable_terminations_and_recorders
from scripts.utils.trigger_server import RemoteTrigger
from scripts.utils.webrtc_cam import setup_webrtc_cam
from simulation.examples.webrtc_runner_cli import add_trigger_cli_args, add_webrtc_cli_args
from simulation.gr00t_closedloop_policy import CustomGr00tClosedloopPolicy
from simulation.register_and_patch import register_workflow_assets, register_workflow_cli


class RunnerState(enum.Enum):
    """States for the triggered policy runner."""

    IDLE = "idle"
    RUNNING = "running"


register_workflow_cli()

# Policy configurations - hardcoded for triggered runner
# Keys are the sequence names used in trigger requests (e.g. ?sequence=tray_pick_and_place)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
POLICY_CONFIGS = {
    "tray_pick_and_place": {
        "max_steps": 1500,
        "prompt": (
            "Grasp the box from the shelf, turn around 180 degrees, "
            "place it onto the cart surface, take two steps backward, and return to initial posture."
        ),
        "config_path": CONFIG_DIR / "g1_gr00t_closedloop_pick_and_place_config.yaml",
    },
    "cart_push": {
        "max_steps": 2000,
        "prompt": (
            "Go to the left side of the cart, turn around to face the cart, "
            "grasp the cart handles with both hands, and push the cart forward to the destination."
        ),
        "config_path": CONFIG_DIR / "g1_gr00t_closedloop_push_cart_config.yaml",
    },
}


def add_room_camera_to_env_cfg(env_cfg):
    """Add a room camera to the environment configuration.

    This mimics what ObserveObjectTask does, but can be applied to any env_cfg.
    """
    import isaaclab.envs.mdp as mdp_isaac_lab
    import isaaclab.sim as sim_utils
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
    from isaaclab.sensors import CameraCfg

    room_camera_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/room_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-1.2, -0.1, 0.9), rot=(-0.21535, 0.21917, -0.78648, 0.53576), convention="ros"),
    )

    # Add camera to scene configuration
    if not hasattr(env_cfg.scene, "room_camera"):
        env_cfg.scene.room_camera = room_camera_cfg
        print("[INFO] Added room_camera to scene")

    # Add room camera observation
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        if not hasattr(env_cfg.observations.policy, "room_camera"):
            room_cam_rgb_obs = ObservationTermCfg(
                func=mdp_isaac_lab.image,
                params={"sensor_cfg": SceneEntityCfg("room_camera"), "data_type": "rgb", "normalize": False},
            )
            env_cfg.observations.policy.room_camera = room_cam_rgb_obs
            print("[INFO] Added room_camera observation to policy")


def _print_idle_controls(trigger_host: str, trigger_port: int):
    """Print available controls when the robot is idle."""
    base = f"http://{trigger_host}:{trigger_port}"
    print("\n[INFO] Robot is idle. Controls:")
    for name in POLICY_CONFIGS:
        print(f"  - POST {base}/trigger?sequence={name}")
    print(f"  - POST {base}/reset  (reset environment)")
    print("  - Press 'R' to reset environment")
    print("  - Press Ctrl+C to stop\n")


def _build_idle_action(obs, env) -> torch.Tensor:
    """Build an idle action that holds the current joint configuration."""
    joint_pos_key = "robot_joint_pos"
    if "policy" not in obs or joint_pos_key not in obs["policy"]:
        raise ValueError(f"Observation '{joint_pos_key}' not found in observation")

    joint_pos = obs["policy"][joint_pos_key]  # (1, num_joints)
    action_dim = env.action_space.shape[-1]
    idle_action = torch.zeros(1, action_dim, device=env.device)

    num_joints = joint_pos.shape[1]
    idle_action[:, :num_joints] = joint_pos.to(env.device)
    # Last 7 dims: [navigate_cmd(3), base_height(1), torso_rpy(3)]
    if action_dim >= num_joints + 7:
        idle_action[:, -4] = 0.75  # base_height at 0.75 as in WBC default

    return idle_action


def _load_policy(policy_name: str, device: str) -> tuple:
    """Load a policy by name. Returns (policy, max_steps)."""
    cfg = POLICY_CONFIGS[policy_name]
    print(f"[INFO] Loading policy: {policy_name}")
    policy = CustomGr00tClosedloopPolicy(
        policy_config_yaml_path=cfg["config_path"],
        device=device,
    )
    policy.set_task_description(cfg["prompt"])
    return policy, cfg["max_steps"]


def _unload_policy(policy) -> None:
    """Delete a policy and free GPU memory."""
    del policy
    torch.cuda.empty_cache()


def main():
    """State-machine runner: IDLE <-> RUNNING.

    IDLE:
        Steps the simulation with an idle (hold-position) action.
        Transitions to RUNNING when a trigger is consumed.

    RUNNING:
        Executes a policy for up to ``max_steps``.
        If a new trigger arrives during execution it is saved; when the
        current policy finishes the saved trigger is consumed immediately
        (RUNNING -> RUNNING) instead of returning to IDLE.
    """
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        register_workflow_assets()
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        add_webrtc_cli_args(args_parser)
        add_trigger_cli_args(args_parser)
        args_cli = args_parser.parse_args()

        # Build environment
        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
        print(f"[INFO] Environment: {env_name}")

        add_room_camera_to_env_cfg(env_cfg)
        disable_terminations_and_recorders(env_cfg)
        env_cfg.is_finite_horizon = False

        env = gym.make(env_name, cfg=env_cfg).unwrapped
        print(f"[INFO] Environment created: {env_name}")

        _maybe_publish_cam = setup_webrtc_cam(args_cli, camera_name="room_camera")

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        keyboard = KeyboardHandler()
        remote_trigger = RemoteTrigger(
            host=getattr(args_cli, "trigger_host", "0.0.0.0"),
            port=getattr(args_cli, "trigger_port", 8081),
        )
        trigger_host = "localhost" if args_cli.trigger_host == "0.0.0.0" else args_cli.trigger_host
        policy_device = getattr(args_cli, "policy_device", "cuda:0")

        def check_any_reset() -> bool:
            return keyboard.check_reset() or remote_trigger.check_reset()

        def resolve_policy_name(sequence: str) -> str | None:
            if sequence in POLICY_CONFIGS:
                return sequence
            print(f"[WARNING] Unknown sequence '{sequence}', ignoring trigger")
            return None

        # ------------------------------------------------------------------
        # Initial reset
        # ------------------------------------------------------------------
        with torch.inference_mode():
            obs, _ = env.reset()
        idle_action = _build_idle_action(obs, env)
        _maybe_publish_cam(obs)

        state = RunnerState.IDLE
        remote_trigger.set_state("idle")
        _print_idle_controls(trigger_host, args_cli.trigger_port)

        # Per-RUNNING-episode variables
        policy = None
        step_count = 0
        max_steps = 0
        progress_bar = None

        # ------------------------------------------------------------------
        # Main state-machine loop (one sim step per iteration)
        # ------------------------------------------------------------------
        try:
            while True:
                # --- Reset handling (works in any state) ---
                if check_any_reset():
                    print("[INFO] Resetting environment...")
                    if policy is not None:
                        _unload_policy(policy)
                        policy = None
                    if progress_bar is not None:
                        progress_bar.close()
                        progress_bar = None
                    with torch.inference_mode():
                        obs, _ = env.reset()
                    idle_action = _build_idle_action(obs, env)
                    _maybe_publish_cam(obs)
                    remote_trigger.clear()
                    remote_trigger.set_state("idle")
                    state = RunnerState.IDLE
                    _print_idle_controls(trigger_host, args_cli.trigger_port)
                    continue

                # --- IDLE state ---
                if state == RunnerState.IDLE:
                    sequence = remote_trigger.consume_trigger()
                    if sequence is not None:
                        # Transition: IDLE -> RUNNING (only if sequence is known)
                        policy_name = resolve_policy_name(sequence)
                        if policy_name is not None:
                            policy, max_steps = _load_policy(policy_name, policy_device)
                            step_count = 0
                            progress_bar = tqdm.tqdm(total=max_steps, desc=policy_name)
                            remote_trigger.set_state("running")
                            state = RunnerState.RUNNING
                            print(f"[INFO] Running policy for {max_steps} steps...")
                    else:
                        with torch.inference_mode():
                            obs, _, _, _, _ = env.step(idle_action)
                            _maybe_publish_cam(obs)

                # --- RUNNING state ---
                elif state == RunnerState.RUNNING:
                    with torch.inference_mode():
                        actions = policy.get_action(env, obs)
                        obs, _, _, _, _ = env.step(actions)
                        _maybe_publish_cam(obs)

                    step_count += 1
                    progress_bar.update(1)

                    if step_count >= max_steps:
                        progress_bar.close()
                        progress_bar = None
                        print(f"[INFO] Policy complete ({step_count} steps)")
                        _unload_policy(policy)
                        policy = None

                        # Check for a trigger that arrived during execution
                        sequence = remote_trigger.consume_trigger()
                        policy_name = resolve_policy_name(sequence) if sequence is not None else None
                        if policy_name is not None:
                            # Transition: RUNNING -> RUNNING (chain next policy)
                            policy, max_steps = _load_policy(policy_name, policy_device)
                            step_count = 0
                            progress_bar = tqdm.tqdm(total=max_steps, desc=policy_name)
                            print(f"[INFO] Pending trigger found. Running policy for {max_steps} steps...")
                        else:
                            # Transition: RUNNING -> IDLE
                            idle_action = _build_idle_action(obs, env)
                            remote_trigger.set_state("idle")
                            state = RunnerState.IDLE
                            _print_idle_controls(trigger_host, args_cli.trigger_port)

        except KeyboardInterrupt:
            print("\n[INFO] Simulation interrupted by user")
        finally:
            if progress_bar is not None:
                progress_bar.close()
            if policy is not None:
                _unload_policy(policy)
            keyboard.close()
            env.close()


if __name__ == "__main__":
    main()
