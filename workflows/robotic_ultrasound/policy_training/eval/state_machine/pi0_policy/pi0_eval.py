import os
import sys
import argparse
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

"""Rest everything follows."""
import gymnasium as gym
import torch

from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import numpy as np


# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401

# append parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from meta_state_machine.ultrasound_state_machine import RobotPositions, RobotQuaternions
from utils import get_robot_obs, compute_relative_action, get_joint_states, KeyboardHandler
from act_policy.act_utils import get_np_images

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import collections
    
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

if args_cli.send_joints:
    assert args_cli.rti_license_file is not None, "RTI license file is required when sending joint states."
    assert os.path.isabs(args_cli.rti_license_file), "RTI license file must be an absolute path."
    os.environ["RTI_LICENSE_FILE"] = args_cli.rti_license_file
    import rti.connextdds as dds
    from rti_dds.schemas.franka_ctrl import FrankaCtrlInput

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


    host = args_cli.host
    port = args_cli.port
    # Connect to the model
    client = _websocket_client_policy.WebsocketClientPolicy(host, port)
    # Prompt for the model
    task_description =  "Conduct a ultrasound scan on the liver."
    # Number of steps played before replanning
    replan_steps = 5

    if args_cli.send_joints:
        participant = dds.DomainParticipant(domain_id=args_cli.domain_id)
        topic = dds.Topic(participant, args_cli.topic_out, FrankaCtrlInput)
        writer = dds.DataWriter(participant.implicit_publisher, topic)

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
                # get images
                room_img, wrist_img = get_np_images(env)
                room_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(room_img, 224, 224)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, 224, 224)
                )

                if not action_plan:
                    current_state = get_joint_states(env)[0] # Get 1st env
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
                    element = {
                        "observation/image": room_img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": current_state,
                        "prompt": task_description,
                    }

                    # Query model to get action
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[: replan_steps])

                action = action_plan.popleft()

                print(action)
                action = action.astype(np.float32)

                # convert to torch
                action = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

                # step the environment
                obs, rew, terminated, truncated, info_ = env.step(action)

                if args_cli.send_joints:
                    joint_states = get_joint_states(env)[0].astype(float).tolist()
                    writer.write(FrankaCtrlInput(joint_positions=joint_states))

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
