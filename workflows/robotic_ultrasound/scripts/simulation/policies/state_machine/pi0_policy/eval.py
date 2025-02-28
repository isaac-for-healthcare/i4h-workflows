import os
import argparse
import collections
import gymnasium as gym
import torch
import numpy as np

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates a single-arm manipulator.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--send_joints", action="store_true", default=False, help="Send joint states to the model.")
parser.add_argument("--rti_license_file", type=str, default=None, help="Path to the RTI license file.")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address.")
parser.add_argument("--port", type=int, default=8000, help="Port number.")
parser.add_argument("--domain_id", type=int, default=int(os.environ.get("OVH_DDS_DOMAIN_ID", 0)), help="Domain ID.")
parser.add_argument("--topic_out", type=str, default="topic_franka_ctrl", help="topic name to publish generated joint velocities")

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

    # Start the keyboard listener in a separate thread
    keyboard_handler = KeyboardHandler()
    keyboard_handler.start_keyboard_listener()

    # allow environment to settle
    for _ in range(reset_steps):
        reset_tensor = get_reset_action(env)
        obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    policy_runner = PI0PolicyRunner(
        host=args_cli.host,
        port=args_cli.port,
        task_description="Conduct a ultrasound scan on the liver.",
        send_joints=args_cli.send_joints,
        rti_license_file=args_cli.rti_license_file,
        domain_id=args_cli.domain_id,
        topic_out=args_cli.topic_out,
    )
    # Number of steps played before replanning
    replan_steps = 5

    # simulate environment
    while simulation_app.is_running():
        global reset_flag
        with torch.inference_mode():
            action_plan = collections.deque()


            for t in range(max_timesteps):
                # Reset the environment on 'r' key press
                if keyboard_handler.reset_flag:
                    keyboard_handler.reset_flag = False
                    break

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

                print(action)
                action = action.astype(np.float32)

                # convert to torch
                action = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

                # step the environment
                obs, rew, terminated, truncated, info_ = env.step(action)

                if args_cli.send_joints:
                    policy_runner.send_joint_states(get_joint_states(env)[0])

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
