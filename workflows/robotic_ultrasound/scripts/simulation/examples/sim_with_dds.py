import argparse
import collections
import os

import gymnasium as gym
import numpy as np
import torch
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.schemas.usp_info import UltraSoundProbeInfo
from dds.subscriber import SubscriberWithQueue
from omni.isaac.lab.app import AppLauncher
from simulation.environments.state_machine.act_policy.act_utils import get_np_images
from simulation.environments.state_machine.utils import (
    compute_relative_action,
    compute_transform_matrix,
    get_joint_states,
    get_probe_pos_ori,
    get_robot_obs,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Run simulation in a single-arm manipulator, communication via DDS.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--rti_license_file", type=str, help="the path of rti_license_file.")
parser.add_argument("--infer_domain_id", type=int, default=0, help="domain id to publish data for inference.")
parser.add_argument("--viz_domain_id", type=int, default=1, help="domain id to publish data for visualization.")
parser.add_argument(
    "--topic_in_room_camera",
    type=str,
    default="topic_room_camera_data_rgb",
    help="topic name to consume room camera rgb",
)
parser.add_argument(
    "--topic_in_wrist_camera",
    type=str,
    default="topic_wrist_camera_data_rgb",
    help="topic name to consume wrist camera rgb",
)
parser.add_argument(
    "--topic_in_franka_pos",
    type=str,
    default="topic_franka_info",
    help="topic name to consume franka pos",
)
parser.add_argument(
    "--topic_in_probe_pos",
    type=str,
    default="topic_ultrasound_info",
    help="topic name to consume probe pos",
)
parser.add_argument(
    "--topic_out",
    type=str,
    default="topic_franka_ctrl",
    help="topic name to publish generated franka actions",
)

# append AppLauncher cli argr
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
reset_flag = False

from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401, E402
from simulation.environments.state_machine.meta_state_machine.ultrasound_state_machine import (  # noqa: E402
    RobotPositions,
    RobotQuaternions,
)

pub_data = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,
    "probe_pos": None,
    "probe_ori": None,
}
hz = 30


class RoomCamPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_room_camera, CameraInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 12.0
        output.height = 224
        output.width = 224
        output.data = pub_data["room_cam"].tobytes()
        return output


class WristCamPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_wrist_camera, CameraInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.height = 224
        output.width = 224
        output.data = pub_data["wrist_cam"].tobytes()
        return output


class PosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_franka_pos, FrankaInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = FrankaInfo()
        output.joints_state_positions = pub_data["joint_pos"].tolist()
        return output


class ProbePosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_probe_pos, UltraSoundProbeInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = UltraSoundProbeInfo()
        output.position = pub_data["probe_pos"].tolist()
        output.orientation = pub_data["probe_ori"].tolist()
        return output


def get_reset_action(env, use_rel: bool = True):
    """Get the reset action."""
    reset_pos = torch.tensor(RobotPositions.SETUP, device=args_cli.device)
    reset_quat = torch.tensor(RobotQuaternions.DOWN, device=args_cli.device)
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

    # get transform matrix from isaac sim to organ coordinate system
    transform_matrix = compute_transform_matrix(
        ov_point=[0.6 * 1000, 0.0, 0.09 * 1000],  # initial position of the organ in isaac sim
        nifti_point=[-0.7168, -0.7168, -330.6],  # corresponding position in nifti coordinate system
    )
    print(f"[INFO]: Coordinate transform matrix: {transform_matrix}")
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    obs = env.reset()

    if args_cli.rti_license_file is None or not os.path.isabs(args_cli.rti_license_file):
        raise ValueError("RTI license file must be an existing absolute path.")
    os.environ["RTI_LICENSE_FILE"] = args_cli.rti_license_file

    reset_steps = 20
    max_timesteps = 250

    # allow environment to settle
    for _ in range(reset_steps):
        reset_tensor = get_reset_action(env)
        obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    infer_r_cam_writer = RoomCamPublisher(args_cli.infer_domain_id)
    infer_w_cam_writer = WristCamPublisher(args_cli.infer_domain_id)
    infer_pos_writer = PosPublisher(args_cli.infer_domain_id)
    viz_r_cam_writer = RoomCamPublisher(args_cli.viz_domain_id)
    viz_w_cam_writer = WristCamPublisher(args_cli.viz_domain_id)
    viz_pos_writer = PosPublisher(args_cli.viz_domain_id)
    viz_probe_pos_writer = ProbePosPublisher(args_cli.viz_domain_id)
    infer_reader = SubscriberWithQueue(args_cli.infer_domain_id, args_cli.topic_out, FrankaCtrlInput, 1 / hz)
    infer_reader.start()

    # Number of steps played before replanning
    replan_steps = 5

    # simulate environment
    while simulation_app.is_running():
        global reset_flag
        with torch.inference_mode():
            action_plan = collections.deque()

            for t in range(max_timesteps):
                # get and publish the current images and joint positions
                pub_data["room_cam"], pub_data["wrist_cam"] = get_np_images(env)
                pub_data["joint_pos"] = get_joint_states(env)[0]
                pub_data["probe_pos"], pub_data["probe_ori"] = get_probe_pos_ori(
                    env, transform_matrix=transform_matrix, scale=1000.0, log=True
                )
                viz_r_cam_writer.write(0.1, 1.0)
                viz_w_cam_writer.write(0.1, 1.0)
                viz_pos_writer.write(0.1, 1.0)
                viz_probe_pos_writer.write(0.1, 1.0)
                if not action_plan:
                    # publish the images and joint positions when run policy inference
                    infer_r_cam_writer.write(0.1, 1.0)
                    infer_w_cam_writer.write(0.1, 1.0)
                    infer_pos_writer.write(0.1, 1.0)

                    ret = None
                    while ret is None:
                        ret = infer_reader.read_data()
                    o: FrankaCtrlInput = ret
                    action_chunk = np.array(o.joint_positions, dtype=np.float32).reshape(50, 6)
                    action_plan.extend(action_chunk[:replan_steps])

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

    infer_reader.stop()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
