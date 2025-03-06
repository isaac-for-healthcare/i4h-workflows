import argparse
import os

import numpy as np
from PIL import Image
from policy_runner.runners import PI0PolicyRunner
from rti_dds.publisher import Publisher
from rti_dds.schemas.camera_info import CameraInfo
from rti_dds.schemas.franka_ctrl import FrankaCtrlInput
from rti_dds.schemas.franka_info import FrankaInfo
from rti_dds.subscriber import SubscriberWithCallback

current_state = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="checkpoint path.")
    parser.add_argument("--repo_id", type=str, help="the LeRobot repo id for the dataset norm.")
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
    args = parser.parse_args()

    pi0_policy = PI0PolicyRunner(ckpt_path=args.ckpt_path, repo_id=args.repo_id)

    if args.rti_license_file is None or not os.path.isabs(args.rti_license_file):
        raise ValueError("RTI license file must be an existing absolute path.")
    os.environ["RTI_LICENSE_FILE"] = args.rti_license_file
    hz = 1 / 30

    class PolicyPublisher(Publisher):
        def __init__(self, topic: str, domain_id: int):
            super().__init__(topic, FrankaCtrlInput, hz, domain_id)

        def produce(self, dt: float, sim_time: float):
            r_cam_buffer = np.frombuffer(current_state["room_cam"], dtype=np.uint8)
            room_img = Image.fromarray(r_cam_buffer.reshape(args.height, args.width, 3), "RGB")
            w_cam_buffer = np.frombuffer(current_state["wrist_cam"], dtype=np.uint8)
            wrist_img = Image.fromarray(w_cam_buffer.reshape(args.height, args.width, 3), "RGB")
            joint_pos = current_state["joint_pos"]
            actions = pi0_policy.infer(
                room_img=np.array(room_img),
                wrist_img=np.array(wrist_img),
                current_state=np.array(joint_pos[:7]),
            )
            i = FrankaCtrlInput()
            # actions are relative positions, if run with absolute positions, need to add the current joint positions
            # actions shape is (50, 6), must reshape to (300,)
            i.joint_positions = (
                np.array(actions)
                .astype(np.float32)
                .reshape(
                    300,
                )
                .tolist()
            )
            return i

    writer = PolicyPublisher(args.topic_out, args.domain_id)

    def dds_callback(topic, data):
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
            print(f"[INFO]: Published joint position to {args.topic_out}")
            # clean the buffer
            current_state["room_cam"] = current_state["wrist_cam"] = current_state["joint_pos"] = None

    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_room_camera, CameraInfo, hz).start()
    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_wrist_camera, CameraInfo, hz).start()
    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_franka_pos, FrankaInfo, hz).start()


if __name__ == "__main__":
    main()
