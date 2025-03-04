import os
import argparse
import collections
import gymnasium as gym
import torch
import numpy as np

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script evaluate the pi0 model in a single-arm manipulator.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--ckpt_path", type=str, help="checkpoint path.")
parser.add_argument("--repo_id", type=str, help="the LeRobot repo id for the dataset norm.")

# append AppLauncher cli argr
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
reset_flag = False

from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401
from simulation.policies.state_machine.act_policy.act_utils import get_np_images
from simulation.policies.state_machine.meta_state_machine.ultrasound_state_machine import (
    RobotPositions,
    RobotQuaternions
)
from simulation.policies.state_machine.utils import (
    get_robot_obs,
    compute_relative_action,
    get_joint_states,
    KeyboardHandler
)
from policy_runner import PI0PolicyRunner

    
def get_reset_action(env, use_rel: bool=True):
    """Get the reset action."""
    reset_pos = torch.tensor(RobotPositions.SETUP, device="cuda:0")
    reset_quat = torch.tensor(RobotQuaternions.DOWN, device="cuda:0")
    reset_tensor = torch.cat([reset_pos, reset_quat], dim=-1)
    reset_tensor = reset_tensor.repeat(env.unwrapped.num_envs, 1)
    if not use_rel:
        return reset_tensor
    else:
        robot_obs = get_robot_obs(env)
        return compute_relative_action(reset_tensor, robot_obs)


def main():
    """Main function."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # modify configuration
    env_cfg.terminations.time_out = None
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    obs = env.reset()

    reset_steps = 20
    max_timesteps = 250

    # allow environment to settle
    for _ in range(reset_steps):
        reset_tensor = get_reset_action(env)
        obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    policy_runner = PI0PolicyRunner(
        ckpt_path=args_cli.ckpt_path,
        repo_id=args_cli.repo_id,
        task_description="Conduct a ultrasound scan on the liver.",
    )
    # Number of steps played before replanning
    replan_steps = 5

    # simulate environment
    while simulation_app.is_running():
        global reset_flag
        with torch.inference_mode():
            action_plan = collections.deque()

            for t in range(max_timesteps):
                if not action_plan:
                    # get images
                    room_img, wrist_img = get_np_images(env)
                    action_chunk = policy_runner.infer(
                        room_img=room_img,
                        wrist_img=wrist_img,
                        current_state=get_joint_states(env)[0]
                    )
                    action_plan.extend(action_chunk[: replan_steps])

                action = action_plan.popleft()

                action = action.astype(np.float32)

                # convert to torch
                action = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

                # step the environment
                obs, rew, terminated, truncated, info_ = env.step(action)

            env.reset()
            for _ in range(reset_steps):
                reset_tensor = get_reset_action(env)
                obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 
