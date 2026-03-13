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

import random

import gymnasium as gym
import numpy as np
import torch
import tqdm
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli
from scripts.utils.webrtc_cam import setup_webrtc_cam
from simulation.examples.webrtc_runner_cli import add_webrtc_cli_args
from simulation.register_and_patch import register_workflow_assets, register_workflow_cli

register_workflow_cli()


def main():
    """Script to run an IsaacLab Arena environment in observation mode without policy."""
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.add_argument(
        "--num_steps",
        type=int,
        default=10000,
        help="Number of simulation steps to run",
    )
    # Add WebRTC streaming arguments
    add_webrtc_cli_args(args_parser)
    # Add environment-specific CLI arguments (task, object, embodiment, etc.)
    add_example_environments_cli_args(args_parser)
    args_cli = args_parser.parse_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):
        # Register Workflow-specific assets
        register_workflow_assets()

        # Build environment configuration
        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()

        print("[INFO] Running in observation mode - no policy, no terminations")
        print(f"[INFO] Environment: {env_name}")
        print(f"[INFO] Number of environments: {args_cli.num_envs}")
        print(f"[INFO] Number of steps: {args_cli.num_steps}")

        # Disable all terminations (like in record_demos_keyboard_23d.py)
        if hasattr(env_cfg, "terminations"):
            if hasattr(env_cfg.terminations, "success"):
                env_cfg.terminations.success = None
                print("[INFO] Disabled success termination")
            if hasattr(env_cfg.terminations, "time_out"):
                env_cfg.terminations.time_out = None
                print("[INFO] Disabled timeout termination")
            # Disable any other terminations
            for attr in dir(env_cfg.terminations):
                if not attr.startswith("_") and attr not in ["success", "time_out"]:
                    setattr(env_cfg.terminations, attr, None)

        # Set infinite horizon
        env_cfg.is_finite_horizon = False

        # Disable recorder to avoid HDF5 file lock conflicts
        if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
            print("[INFO] Disabling recorders (not needed for observe mode)")
            env_cfg.recorders = None

        # Create environment
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        print(f"[INFO] Environment created: {env_name}")

        # Optional WebRTC livestream of the room camera.
        _maybe_publish_webrtc = setup_webrtc_cam(args_cli, camera_name="room_camera")

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)
            print(f"[INFO] Random seed set to: {args_cli.seed}")

        obs, _ = env.reset()
        _maybe_publish_webrtc(obs)

        # Get action space dimensions
        action_dim = env.action_space.shape[-1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]

        initial_action = torch.zeros(env.num_envs, action_dim, device=env.device)
        initial_action[:, 0] = 0.0  # left gripper (open)
        initial_action[:, 1] = 0.0  # right gripper (open)
        initial_action[:, 2:5] = torch.tensor([0.15, 0.15, 0.2], device=env.device)  # left wrist pos
        initial_action[:, 5:9] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)  # left wrist quat (wxyz)
        initial_action[:, 9:12] = torch.tensor([0.15, -0.15, 0.2], device=env.device)  # right wrist pos
        initial_action[:, 12:16] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)  # right wrist quat (wxyz)
        initial_action[:, 16:19] = 0.0  # no base movement
        initial_action[:, 19] = 0.75  # default base height
        initial_action[:, 20:23] = 0.0  # no torso tilt

        print("[INFO] Starting observation mode - maintaining default posture")
        print("[INFO] Press Ctrl+C to stop\n")

        try:
            # Run simulation with initial action to maintain posture
            for _ in tqdm.tqdm(range(args_cli.num_steps), desc="Running simulation"):
                with torch.inference_mode():
                    # Step environment with initial action to maintain initial posture
                    obs, _, _, _, _ = env.step(initial_action)
                    _maybe_publish_webrtc(obs)

        except KeyboardInterrupt:
            print("\n[INFO] Simulation interrupted by user")

        print("\n[INFO] Simulation complete")
        env.close()


if __name__ == "__main__":
    main()
