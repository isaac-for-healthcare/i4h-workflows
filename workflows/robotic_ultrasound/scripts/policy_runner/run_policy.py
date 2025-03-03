import argparse
import os
import numpy as np
from PIL import Image

from rti_dds.schemas.camera_info import CameraInfo
from rti_dds.schemas.franka_ctrl import FrankaCtrlInput
from rti_dds.schemas.franka_info import FrankaInfo

from rti_dds.publisher import Publisher
from rti_dds.subscriber import SubscriberWithCallback
from policy_runner.runners import PI0PolicyRunner

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

    if args.rti_license_file is None or not os.path.isabs(args.rti_license_file):
        raise ValueError("RTI license file must be an existing absolute path.")
    os.environ["RTI_LICENSE_FILE"] = args.rti_license_file
    hz = 1 / 30

    class PolicyPublisher(Publisher):
        def __init__(self, topic: str, domain_id: int):
            super().__init__(topic, FrankaCtrlInput, hz, domain_id)

        def produce(self, dt: float, sim_time: float):
            r_cam_buffer = np.frombuffer(current_state["room_cam"], dtype=np.uint8)
            room_cam = Image.fromarray(r_cam_buffer.reshape(args.height, args.width, 4), "RGBA").convert("RGB")
            w_cam_buffer = np.frombuffer(current_state["wrist_cam"], dtype=np.uint8)
            wrist_cam = Image.fromarray(w_cam_buffer.reshape(args.height, args.width, 4), "RGBA").convert("RGB")
            joint_pos = current_state["joint_pos"]

            actions = pi0_policy.infer(
                room_cam=np.array(room_cam),
                wrist_cam=np.array(wrist_cam),
                joint_pos=np.array(joint_pos[:7]),
            )
            actions = np.array(actions).astype(np.float32)

            # First action only
            for action in actions:
                target_jp = [i for i in joint_pos]
                for i in range(len(action)):
                    target_jp[i] = target_jp[i] + action[i]

                i = FrankaCtrlInput()
                i.joint_positions = target_jp
                return i


    writer = PolicyPublisher(args.topic_out, args.domain_id)

    def dds_callback(topic, data):
        if topic == args.topic_in_room_camera:
            o: CameraInfo = data
            current_state["room_cam"] = o.data

        if topic == args.topic_in_wrist_camera:
            o: CameraInfo = data
            current_state["wrist_cam"] = o.data

        if topic == args.topic_in_franka_pos:
            o: FrankaInfo = data
            current_state["joint_pos"] = o.joints_state_positions
            if current_state["room_cam"] and current_state["wrist_cam"] and current_state["joint_pos"]:
                writer.write(0, 0)

    r_cam_reader = SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_room_camera, CameraInfo, hz)
    w_cam_reader = SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_wrist_camera, CameraInfo, hz)
    f_pos_reader = SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_franka_pos, FrankaInfo, hz)


    pi0_policy = PI0PolicyRunner(ckpt_path=args.ckpt_path, repo_id=args.repo_id)

    r_cam_reader.start()
    w_cam_reader.start()
    f_pos_reader.start()


if __name__ == "__main__":
    main()
