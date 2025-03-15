# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import os

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment",
)
parser.add_argument("--task", type=str, default="Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0", help="Name of the task. Default is Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--topic_in_probe_pos",
    type=str,
    default="topic_ultrasound_info",
    help="topic name to consume probe pos",
)

parser.add_argument("--rti_license_file", type=str, default=None, help="Path to the RTI license file.")
parser.add_argument(
    "--viz_domain_id",
    type=int,
    default=0,
    help="domain id to publish data for visualization.",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Add after existing parser arguments
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

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import time  # noqa: F401, E402

import gymnasium as gym  # noqa: F401, E402
import numpy as np
import omni.isaac.lab.utils.math as math_utils  # noqa: F401, E402
import omni.isaac.lab_tasks  # noqa: F401, E402
import omni.log  # noqa: F401, E402
import torch  # noqa: F401, E402
from dds.publisher import Publisher  # noqa: F401, E402
from dds.schemas.camera_info import CameraInfo  # noqa: F401, E402
from dds.schemas.franka_info import FrankaInfo  # noqa: F401, E402
from dds.schemas.usp_info import UltraSoundProbeInfo  # noqa: F401, E402
from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse  # noqa: F401, E402
from omni.isaac.lab.managers import SceneEntityCfg  # noqa: F401, E402
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm  # noqa: F401, E402
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp  # noqa: F401, E402
from omni.isaac.lab_tasks.utils import parse_env_cfg  # noqa: F401, E402
# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401, E402
from simulation.environments.state_machine.utils import get_joint_states

# Add RTI DDS imports
if args_cli.rti_license_file is not None:
    if not os.path.isabs(args_cli.rti_license_file):
        raise ValueError("RTI license file must be an existing absolute path.")
    os.environ["RTI_LICENSE_FILE"] = args_cli.rti_license_file


def capture_camera_images(env, cam_names, device="cuda"):
    """Captures RGB and depth images from specified cameras"""
    depths, rgbs = [], []
    for cam_name in cam_names:
        camera_data = env.unwrapped.scene[cam_name].data
        rgb = camera_data.output["rgb"][..., :3].squeeze(0)
        depth = camera_data.output["distance_to_image_plane"].squeeze(0)
        rgbs.append(rgb)
        depths.append(depth)

    return torch.stack(rgbs).unsqueeze(0), torch.stack(depths).unsqueeze(0)


# Add publisher classes before main()
pub_data = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,  # Added for robot pose
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


# Add new publisher class with existing publishers
class PosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__("topic_franka_info", FrankaInfo, 1 / hz, domain_id)

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


def make_pose(pos, rot):
    """
    Make homogeneous pose matrices from a set of translation vectors and rotation matrices.

    Args:
        pos (torch.Tensor): batch of position vectors with last dimension of 3
        rot (torch.Tensor): batch of rotation matrices with last 2 dimensions of (3, 3)

    Returns:
        pose (torch.Tensor): batch of pose matrices with last 2 dimensions of (4, 4)
    """
    assert isinstance(pos, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(rot, torch.Tensor), "Input must be a torch tensor"
    assert pos.shape[:-1] == rot.shape[:-2]
    assert pos.shape[-1] == rot.shape[-2] == rot.shape[-1] == 3
    pose = torch.zeros(pos.shape[:-1] + (4, 4), dtype=pos.dtype, device=pos.device)
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def matrix_from_pos_quat(pos, quat, device="cuda"):
    """Convert position and quaternion to a 4x4 transformation matrix.

    Args:
        pos (torch.Tensor): Position vector of shape (1, 3)
        quat (torch.Tensor): Quaternion of shape (1, 4) in (w, x, y, z) format

    Returns:
        torch.Tensor: 4x4 homogeneous transformation matrix
    """
    # Type assertions
    assert isinstance(pos, torch.Tensor), "Position must be a torch.Tensor"
    assert isinstance(quat, torch.Tensor), "Quaternion must be a torch.Tensor"

    # Shape assertions
    assert pos.shape == (1, 3), f"Position must have shape (1, 3), got {pos.shape}"
    assert quat.shape == (1, 4), f"Quaternion must have shape (1, 4), got {quat.shape}"

    # Datatype assertions
    assert pos.dtype == torch.float64, "Position must be double precision (float64)"
    assert quat.dtype == torch.float64, "Quaternion must be double precision (float64)"

    # put on device
    pos = pos.to(device)
    quat = quat.to(device)

    # Convert quaternion to rotation matrix
    rot = math_utils.matrix_from_quat(quat)

    # Create transformation matrix
    transform = make_pose(pos, rot)

    return transform


def quat_from_euler_xyz_deg(roll, pitch, yaw, device="cuda"):
    euler_angles = np.array([roll, pitch, yaw])
    euler_angles_rad = np.radians(euler_angles)
    euler_angles_rad = torch.tensor(euler_angles_rad, device=device).double()
    quat_sim_to_nifti = math_utils.quat_from_euler_xyz(
        roll=euler_angles_rad[0], pitch=euler_angles_rad[1], yaw=euler_angles_rad[2]
    )
    quat_sim_to_nifti = quat_sim_to_nifti.unsqueeze(0)
    return quat_sim_to_nifti


def get_joint_states(env):
    """Get the robot joint states from the environment."""
    robot_data = env.unwrapped.scene["robot"].data
    robot_joint_pos = robot_data.joint_pos
    return robot_joint_pos.cpu().numpy()


def make_pose(pos, rot):
    """
    Make homogeneous pose matrices from a set of translation vectors and rotation matrices.

    Args:
        pos (torch.Tensor): batch of position vectors with last dimension of 3
        rot (torch.Tensor): batch of rotation matrices with last 2 dimensions of (3, 3)

    Returns:
        pose (torch.Tensor): batch of pose matrices with last 2 dimensions of (4, 4)
    """
    assert isinstance(pos, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(rot, torch.Tensor), "Input must be a torch tensor"
    assert pos.shape[:-1] == rot.shape[:-2]
    assert pos.shape[-1] == rot.shape[-2] == rot.shape[-1] == 3
    pose = torch.zeros(pos.shape[:-1] + (4, 4), dtype=pos.dtype, device=pos.device)
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity,
            rot_sensitivity=0.15 * args_cli.sensitivity,
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity,
            rot_sensitivity=0.015 * args_cli.sensitivity,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # Add rate limiting variables
    target_hz = 30.0
    target_dt = 1.0 / target_hz
    last_step_time = time.time()

    # Create publishers for cameras
    viz_r_cam_writer = RoomCamPublisher(args_cli.viz_domain_id)
    viz_w_cam_writer = WristCamPublisher(args_cli.viz_domain_id)
    # viz_pos_writer = PosPublisher(args_cli.viz_domain_id)
    viz_probe_pos_writer = ProbePosPublisher(args_cli.viz_domain_id)
    # Added robot pose publisher

    # Describe the orientation of the organ frame in the nifti frame (mesh coords system)
    quat_MeshToOrgan = quat_from_euler_xyz_deg(90.0, 180.0, 0.0)
    # Describe how to position the mesh objects in the organ frame so that it roughly match
    mesh_offset = torch.zeros(1, 3, device=env.unwrapped.device)
    mesh_offset[0, 2] = -360.0 / 1000.0

    # Fixed orientation of the end-effector to the ultrasound probe
    quat_EEToUS = quat_from_euler_xyz_deg(0.0, 0.0, -90.0)

    # simulate environment
    while simulation_app.is_running():
        current_time = time.time()
        dt = current_time - last_step_time
        if dt < target_dt:
            time.sleep(target_dt - dt)
            continue
        last_step_time = current_time

        with torch.inference_mode():
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = torch.tensor(delta_pose.astype("float32"), device=env.unwrapped.device).repeat(
                env.unwrapped.num_envs, 1
            )

            delta_pos = delta_pose[:, :3]
            delta_rot = math_utils.quat_from_euler_xyz(delta_pose[:, 3], delta_pose[:, 4], delta_pose[:, 5])

            # get the robot's TCP pose
            robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
            robot_entity_cfg.resolve(env.unwrapped.scene)
            ee_pose_w = env.unwrapped.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7].clone()
            ee_pos_w = ee_pose_w[:, :3]
            ee_rot_w = ee_pose_w[:, 3:7]

            # compute the target pose in the world frame
            target_pos_w, target_rot_w = math_utils.combine_frame_transforms(ee_pos_w, ee_rot_w, delta_pos, delta_rot)

            # compute the error in the pose in the world frame
            delta_pos, delta_rot = math_utils.compute_pose_error(
                ee_pos_w, ee_rot_w, target_pos_w, target_rot_w, rot_error_type="quat"
            )
            delta_rot[delta_rot.abs() < 1e-6] = 0.0

            # convert to angle-axis because it's needed for the action space
            target_rot_angle_axis = math_utils.axis_angle_from_quat(delta_rot)
            actions = torch.cat([delta_pos, target_rot_angle_axis], dim=1)

            env.step(actions)

            # Get the end-effector pose in the organ frame
            pos_OrganToEE = env.unwrapped.scene["organ_to_robot_transform"].data.target_pos_source[0].double()
            quat_OrganToEE = env.unwrapped.scene["organ_to_robot_transform"].data.target_quat_source[0]

            # apply the transformation from sim_to_nifti frame -> returns the orientation of the end-effector in the nifti frame, using quaternion transformation
            quat_MeshToEE = math_utils.quat_mul(quat_MeshToOrgan, quat_OrganToEE)
            pos_MeshToEE = math_utils.quat_apply(quat_MeshToOrgan, pos_OrganToEE) + mesh_offset

            # print("pos_ee_from_organ shape:", pos_ee_from_organ.shape)
            # print(f"dtype of pos_ee_from_organ: {pos_ee_from_organ.dtype}")

            # Add an additional rotation to the end-effector pose from sim to nifti frame
            # add additional rotations here if needed
            # qnew​=qold​ × qz​(α).   
            # # create a matrix from pos quat for MeshToEE
            # matrix_EEToUS = matrix_from_pos_quat(torch.zeros(1, 3).double(), quat_EEToUS)

            # # create a matrix from pos quat
            # matrix_MeshToEE = matrix_from_pos_quat(pos_MeshToEE, quat_MeshToEE)

            # # multiply with local transform probe to probe us
            # matrix_MeshToUS = torch.matmul(matrix_MeshToEE, matrix_EEToUS)
            # # convert matrix to quat and compare with below
            # quat_MeshToUS = math_utils.quat_from_matrix(matrix_MeshToUS[:, :3, :3])
            # pos_MeshToUS = matrix_MeshToUS[:, :3, 3]

            quat_MeshToUS = math_utils.quat_mul(quat_MeshToEE, quat_EEToUS)
            pos_MeshToUS = pos_MeshToEE

            # quat_MeshToUS = math_utils.normalize(quat_MeshToUS)
            # print("quat:", quat_MeshToUS)
            # print("quat shape:", quat_MeshToUS.shape)
            # normalize the quat

            # apply the transformation to the end-effector pose
            # quat = math_utils.quat_mul(quat, probe_to_probe_us_quat)

            # # print both and compare
            # print("quat_local:", quat_local)
            # print("quat:", quat)
            # assert torch.allclose(quat_local, quat)
            # pos =  math_utils.quat_apply(probe_to_probe_us_quat, pos)

            # scale the position from m to mm
            pos = pos_MeshToUS * 1000.0
            pos_np = pos.cpu().numpy().squeeze()

            # convert the quat to euler angles
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_MeshToUS)
            # stack the euler angles into roll pich yaw tensor
            euler_angles = np.array(
                [roll.squeeze().cpu().numpy(), pitch.squeeze().cpu().numpy(), yaw.squeeze().cpu().numpy()]
            )
            euler_angles_deg = np.degrees(euler_angles)
            # print the results in euler angles in degrees
            print("pos:", pos_np)
            print("pos shape:", pos_np.shape)
            print("euler angles:", euler_angles_deg)
            print("euler shape:", euler_angles.shape)

            # convert to numpy and publish
            pub_data["probe_pos"] = pos_np
            pub_data["probe_ori"] = euler_angles

            # Get and publish camera images
            rgb_images, _ = capture_camera_images(env, ["room_camera", "wrist_camera"], device=env.unwrapped.device)
            pub_data["room_cam"] = rgb_images[0, 0, ...].cpu().numpy()
            pub_data["wrist_cam"] = rgb_images[0, 1, ...].cpu().numpy()
            # pub_data["probe_pos"], pub_data["probe_ori"] = get_probe_pos_ori(
            #         env, transform_matrix=transform_matrix, scale=1000.0, log=True
            #     )
            # # Get and publish joint positions
            pub_data["joint_pos"] = get_joint_states(env)[0]

            viz_r_cam_writer.write(0.1, 1.0)
            viz_w_cam_writer.write(0.1, 1.0)
            viz_probe_pos_writer.write(0.1, 1.0)

    env.close()


if __name__ == "__main__":
    # run the main function
    # rti.asyncio.run(main())
    main()
    # close sim app
    simulation_app.close()
