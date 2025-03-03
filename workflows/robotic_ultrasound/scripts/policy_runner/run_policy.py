import argparse
import asyncio
import os
import numpy as np
import rti.asyncio
import rti.connextdds as dds
from PIL import Image

from rti_dds.schemas.camera_info import CameraInfo
from rti_dds.schemas.franka_ctrl import FrankaCtrlInput
from rti_dds.schemas.franka_info import FrankaInfo

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
    dp = dds.DomainParticipant(domain_id=args.domain_id)
    r_cam_reader = dds.DataReader(dp.implicit_subscriber, dds.Topic(dp, args.topic_in_room_camera, CameraInfo))
    w_cam_reader = dds.DataReader(dp.implicit_subscriber, dds.Topic(dp, args.topic_in_wrist_camera, CameraInfo))
    f_pos_reader = dds.DataReader(dp.implicit_subscriber, dds.Topic(dp, args.topic_in_franka_pos, FrankaInfo))
    writer = dds.DataWriter(dp.implicit_publisher, dds.Topic(dp, args.topic_out, FrankaCtrlInput))

    pi0_policy = PI0PolicyRunner(ckpt_path=args.ckpt_path, repo_id=args.repo_id)

    async def read_room():
        print(f"{args.domain_id}:{args.topic_in_room_camera} => Waiting for data...")
        async for data in r_cam_reader.take_data_async():
            o: CameraInfo = data
            current_state["room_cam"] = o.data

    async def read_wrist():
        print(f"{args.domain_id}:{args.topic_in_room_camera} => Waiting for data...")
        async for data in w_cam_reader.take_data_async():
            o: CameraInfo = data
            current_state["wrist_cam"] = o.data

    async def read_pos():
        print(f"{args.domain_id}:{args.topic_in_room_camera} => Waiting for data...")
        async for data in f_pos_reader.take_data_async():
            o: FrankaInfo = data
            current_state["joint_pos"] = o.joints_state_positions

            mc = current_state["room_cam"]
            wc = current_state["wrist_cam"]
            jp = current_state["joint_pos"]

            if mc is not None and wc is not None and jp is not None:
                r_cam_buffer = np.frombuffer(mc, dtype=np.uint8).reshape(args.height, args.width, 4)
                room_cam = Image.fromarray(r_cam_buffer, "RGBA").convert("RGB")
                w_cam_buffer = np.frombuffer(wc, dtype=np.uint8).reshape(args.height, args.width, 4)
                wrist_cam = Image.fromarray(w_cam_buffer, "RGBA").convert("RGB")
                joint_pos = jp

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
                    writer.write(i)
                    break

    async def run_simulation():
        await asyncio.gather(
            read_room(),
            read_wrist(),
            read_pos(),
        )

    rti.asyncio.run(run_simulation())


if __name__ == "__main__":
    main()
