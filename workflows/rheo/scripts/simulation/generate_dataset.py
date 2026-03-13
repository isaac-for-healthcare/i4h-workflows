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

"""Workflow-specific wrapper around the IsaacLab Arena dataset generation script."""

from typing import Any

from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli
from simulation.register_and_patch import register_workflow_assets, register_workflow_cli
from utils.policy_tasks import create_success_hold_wrapper

register_workflow_cli()

# add argparse arguments
parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--generation_num_trials", type=int, help="Number of demos to be generated.", default=None, required=True
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--keep_failed",
    action="store_true",
    default=False,
    help="Keep failed demos in a separate file (useful for debugging and replaying unsuccessful demos).",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1,
    help="Number of consecutive steps the success condition must be satisfied to consider task successful.",
)

# Add the example environments CLI args
add_example_environments_cli_args(parser)

# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import asyncio
import inspect
import random

import gymnasium as gym
import isaaclab_mimic.envs  # noqa: F401
import numpy as np
import omni
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation
from isaaclab_mimic.datagen.utils import setup_output_paths

register_workflow_assets()


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step."""

    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Add the camera observation recorder term."""

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()


def setup_env_config(
    args_cli: Any,
    output_dir: str,
    output_file_name: str,
    num_envs: int,
    device: str,
    generation_num_trials: int | None = None,
) -> tuple[Any, str, Any]:
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    if generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = generation_num_trials

    if hasattr(args_cli, "keep_failed") and args_cli.keep_failed:
        env_cfg.datagen_config.generation_keep_failed = True

    env_cfg.env_name = env_name

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    # Remove camera observations from policy group since we use camera_obs group instead
    if hasattr(env_cfg.observations.policy, "robot_head_cam"):
        env_cfg.observations.policy.robot_head_cam = None
    if hasattr(env_cfg.observations.policy, "robot_head_cam_seg"):
        env_cfg.observations.policy.robot_head_cam_seg = None

    if args_cli.enable_cameras:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, env_name, success_term


def main():
    num_envs = args_cli.num_envs
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)

    env_cfg, env_name, success_term = setup_env_config(
        args_cli=args_cli,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )

    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    if args_cli.num_steps > 1 and success_term is not None:
        success_term = create_success_hold_wrapper(success_term, args_cli.num_steps, num_envs)

    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    env.reset()

    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    simulation_app.close()
