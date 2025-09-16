# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

import so_arm_starter_ext  # noqa: F401
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="so101leader",
    choices=["keyboard", "so101leader"],
    help="Device for teleoperating the robot in simulation. Supported: keyboard, so101leader",
)
parser.add_argument(
    "--port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for connecting to the teleop device SO-ARM101 leader arm, default is /dev/ttyACM0",
)
parser.add_argument("--task", type=str, default="Isaac-SOARM101-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor for teleoperation.")

# recorder_parameter
parser.add_argument("--record", action="store_true", default=False, help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_path", type=str, default="./datasets/test_data.hdf5", help="Path to export recorded dataset."
)
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument("--recalibrate", action="store_true", default=False, help="recalibrate SO101-Leader")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.devices import Se3Keyboard, SO101Leader
from leisaac.enhance.managers import StreamingRecorderManager


class TeleopEvent(Enum):
    RESET_RECORDING = "reset_recording"
    TASK_SUCCESS = "task_success"


@dataclass
class TeleopState:
    recording_instance_reset_pending: bool = False
    task_success_pending: bool = False
    start_record_state: bool = False

    def reset_flags(self):
        self.recording_instance_reset_pending = False
        self.task_success_pending = False


class TeleopEventHandler:
    def __init__(self, state: TeleopState):
        self.state = state

    def handle_reset_recording(self):
        self.state.recording_instance_reset_pending = True

    def handle_task_success(self):
        self.state.task_success_pending = True
        self.state.recording_instance_reset_pending = True


# Force robot to zero pose
def manual_reset_robot_pose(env, joint_positions=[0.0, -1.6, 1.4, 1.4, -1.8, 0.0]):
    """Reset the robot to a specified initial joint configuration."""
    robot = env.scene["robot"]
    target_positions = torch.tensor([joint_positions], device=robot.device, dtype=torch.float32)
    target_velocities = torch.zeros_like(target_positions)
    robot.write_joint_state_to_sim(target_positions, target_velocities)

    for _ in range(5):
        env.sim.step()

    return robot.data.joint_pos[0].cpu().numpy()


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def _success_termination_false(env):
    """Helper function for success termination that returns False."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _success_termination_true(env):
    """Helper function for success termination that returns True."""
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


@torch.inference_mode()
def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_path)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_path))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    task_name = args_cli.task

    # modify configuration
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if args_cli.record:
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(func=_success_termination_false)
    else:
        env_cfg.recorders = None
    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    # replace the original recorder manager with the streaming recorder manager
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"

    # create controller
    if args_cli.teleop_device == "keyboard":
        teleop_interface = Se3Keyboard(env, sensitivity=0.25 * args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'so101leader'.")

    # add teleoperation key for env reset
    teleop_state = TeleopState()
    event_handler = TeleopEventHandler(teleop_state)

    # add teleoperation key callbacks
    teleop_interface.add_callback("R", event_handler.handle_reset_recording)
    teleop_interface.add_callback("N", event_handler.handle_task_success)
    print(teleop_interface)

    rate_limiter = RateLimiter(args_cli.step_hz)

    env.reset()
    teleop_interface.reset()

    # Apply manual reset
    actual_positions = manual_reset_robot_pose(env)

    current_recorded_demo_count = 0

    while simulation_app.is_running():
        # get actions from teleop interface
        actions = teleop_interface.advance()
        if teleop_state.task_success_pending:
            print("Task Success!!!")
            teleop_state.task_success_pending = False
            if args_cli.record:
                env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=_success_termination_true))
                env.termination_manager.compute()
        if teleop_state.recording_instance_reset_pending:
            env.reset()
            teleop_state.reset_flags()

            # MANUAL RESET: Force robot to surgical pose when R/N is pressed
            actual_positions = manual_reset_robot_pose(env)
            print(f"ROBOT POSITIONS AFTER MANUAL RESET (R/N): {actual_positions[:6]}")

            if teleop_state.start_record_state:
                if args_cli.record:
                    print("Stop Recording!!!")
                teleop_state.start_record_state = False
            if args_cli.record:
                env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=_success_termination_false))
            # print out the current demo count if it has changed
            if args_cli.record and env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
            if (
                args_cli.record
                and args_cli.num_demos > 0
                and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos
            ):
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

        elif actions is None:
            env.render()
        # apply actions
        else:
            if not teleop_state.start_record_state:
                teleop_state.start_record_state = True
                print("Start Recording!!!")
            if teleop_state.start_record_state:
                env.step(actions)
            else:
                env.render()

        if rate_limiter:
            rate_limiter.sleep(env)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
