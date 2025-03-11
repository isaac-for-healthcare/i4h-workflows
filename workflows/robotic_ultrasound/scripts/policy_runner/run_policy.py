import argparse
import os

import numpy as np
from PIL import Image
from policy_runner.runners import PI0PolicyRunner
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.subscriber import SubscriberWithCallback
from openpi_client import image_tools

current_state = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,
}

# Ensure pi0 doesn't take all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="checkpoint path.")
    parser.add_argument(
        "--repo_id", type=str, default="i4h/robotic_ultrasound", help="the LeRobot repo id for the dataset norm."
    )
    parser.add_argument("--rti_license_file", type=str, help="the path of rti_license_file.")
    parser.add_argument("--domain_id", type=int, default=0, help="domain id.")
    parser.add_argument("--height", type=int, default=224, help="input image height.")
    parser.add_argument("--width", type=int, default=224, help="input image width.")
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
        "--topic_out",
        type=str,
        default="topic_franka_ctrl",
        help="topic name to publish generated franka actions",
    )
    parser.add_argument("--verbose", type=bool, default=False, help="whether to print the log.")
    args = parser.parse_args()

    pi0_policy = PI0PolicyRunner(ckpt_path=args.ckpt_path, repo_id=args.repo_id)

    if args.rti_license_file is None or not os.path.isabs(args.rti_license_file):
        raise ValueError("RTI license file must be an existing absolute path.")
    os.environ["RTI_LICENSE_FILE"] = args.rti_license_file
    hz = 20

    class PolicyPublisher(Publisher):
        def __init__(self, topic: str, domain_id: int):
            super().__init__(topic, FrankaCtrlInput, 1 / hz, domain_id)

        def produce(self, dt: float, sim_time: float):
            r_cam_buffer = np.frombuffer(current_state["room_cam"], dtype=np.uint8)
            room_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(r_cam_buffer, 224, 224))
            w_cam_buffer = np.frombuffer(current_state["wrist_cam"], dtype=np.uint8)
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(w_cam_buffer, 224, 224))
            joint_pos = current_state["joint_pos"]
            actions = pi0_policy.infer(
                room_img=room_img,
                wrist_img=wrist_img,
                current_state=joint_pos,
            )
            i = FrankaCtrlInput()
            # actions are relative positions, if run with absolute positions, need to add the current joint positions
            # actions shape is (50, 6), must reshape to (300,)
            i.joint_positions = (
                np.array(actions)
                .astype(np.float32)
                .reshape(
                    350,
                )
                .tolist()
            )
            return i

    writer = PolicyPublisher(args.topic_out, args.domain_id)

    def dds_callback(topic, data):
        if args.verbose:
            print(f"[INFO]: Received data from {topic}")
        if topic == args.topic_in_room_camera:
            o: CameraInfo = data
            current_state["room_cam"] = o.data

        if topic == args.topic_in_wrist_camera:
            o: CameraInfo = data
            current_state["wrist_cam"] = o.data

        if topic == args.topic_in_franka_pos:
            o: FrankaInfo = data
            current_state["joint_pos"] = o.joints_state_positions
        if (
            current_state["room_cam"] is not None
            and current_state["wrist_cam"] is not None
            and current_state["joint_pos"] is not None
        ):
            writer.write(0.1, 1.0)
            if args.verbose:
                print(f"[INFO]: Published joint position to {args.topic_out}")
            # clean the buffer
            current_state["room_cam"] = current_state["wrist_cam"] = current_state["joint_pos"] = None

    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_room_camera, CameraInfo, 1 / hz).start()
    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_wrist_camera, CameraInfo, 1 / hz).start()
    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_franka_pos, FrankaInfo, 1 / hz).start()


if __name__ == "__main__":
    main()
