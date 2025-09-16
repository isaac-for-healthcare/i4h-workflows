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

import argparse
import os

import numpy as np
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.soarm_ctrl import SOARM101CtrlInput
from dds.schemas.soarm_info import SOARM101Info
from dds.subscriber import SubscriberWithCallback
from PIL import Image
from policy_runner.gr00tn1_5.runners import GR00TN1_5_PolicyRunner

current_state = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,
}

# Prevent JAX from preallocating all memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main():
    parser = argparse.ArgumentParser(description="Run the GR00T N1.5 policy runner")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/so101_sim_scissors_finetune_30k/checkpoint-30000",
        help="checkpoint path. Default will use the policy model in the downloaded assets.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="Grip the scissors and put it into the tray",
        help="Task description for the policy.",
    )
    gr00tn1_5_group = parser.add_argument_group("GR00T N1.5 Policy Arguments")
    gr00tn1_5_group.add_argument(
        "--data_config",
        type=str,
        default="so100_dualcam",
        help="Data config name for GR00T N1.5 policy",
    )
    gr00tn1_5_group.add_argument(
        "--embodiment_tag",
        type=str,
        default="new_embodiment",
        help="The embodiment tag for the GR00T N1.5 model.",
    )
    parser.add_argument(
        "--rti_license_file", type=str, default=os.getenv("RTI_LICENSE_FILE"), help="the path of rti_license_file."
    )
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        default="gr00t_engine",
        help="Path to tensorrt engine",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Use tensorrt engine",
    )
    parser.add_argument("--domain_id", type=int, default=0, help="domain id.")
    parser.add_argument("--height", type=int, default=480, help="input image height")
    parser.add_argument("--width", type=int, default=640, help="input image width")
    parser.add_argument(
        "--topic_in_room_camera",
        type=str,
        default="topic_room_camera_data_rgb",
        help="topic name to consume room camera rgb.",
    )
    parser.add_argument(
        "--topic_in_wrist_camera",
        type=str,
        default="topic_wrist_camera_data_rgb",
        help="topic name to consume wrist camera rgb.",
    )
    parser.add_argument(
        "--topic_in_soarm_pos",
        type=str,
        default="topic_soarm_info",
        help="topic name to consume soarm pos.",
    )
    parser.add_argument(
        "--topic_out",
        type=str,
        default="topic_soarm_ctrl",
        help="topic name to publish generated soarm actions.",
    )
    parser.add_argument("--verbose", type=bool, default=False, help="whether to print the log.")
    parser.add_argument("--policy", type=str, default="gr00tn1.5", choices=["gr00tn1.5"], help="policy type to use.")
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=16,
        help="Length of the action chunk inferred by the policy.",
    )
    args = parser.parse_args()

    if args.trt:
        if not os.path.exists(args.trt_engine_path):
            raise ValueError(f"Tensorrt engine path {args.trt_engine_path} does not exist.")
        else:
            policy = GR00TN1_5_PolicyRunner(
                ckpt_path=args.ckpt_path,
                data_config=args.data_config,
                embodiment_tag=args.embodiment_tag,
                task_description=args.task_description,
                trt_engine_path=args.trt_engine_path,
                trt=True,
            )
    else:
        policy = GR00TN1_5_PolicyRunner(
            ckpt_path=args.ckpt_path,
            data_config=args.data_config,
            embodiment_tag=args.embodiment_tag,
            task_description=args.task_description,
            trt=False,
        )

    if args.rti_license_file is not None:
        if not os.path.isabs(args.rti_license_file):
            raise ValueError("RTI license file must be an existing absolute path.")
        os.environ["RTI_LICENSE_FILE"] = args.rti_license_file

    hz = 60

    class PolicyPublisher(Publisher):
        def __init__(self, topic: str, domain_id: int):
            super().__init__(topic, SOARM101CtrlInput, 1 / hz, domain_id)

        def produce(self, dt: float, sim_time: float):
            r_cam_buffer = np.frombuffer(current_state["room_cam"], dtype=np.uint8)
            room_img = Image.fromarray(r_cam_buffer.reshape(args.height, args.width, 3), "RGB")
            w_cam_buffer = np.frombuffer(current_state["wrist_cam"], dtype=np.uint8)
            wrist_img = Image.fromarray(w_cam_buffer.reshape(args.height, args.width, 3), "RGB")
            joint_pos = current_state["joint_pos"]
            # print(f"joint_pos: {joint_pos}")

            actions = policy.infer(
                room_img=np.array(room_img),
                wrist_img=np.array(wrist_img),
                current_state=np.array(joint_pos[:6]),
            )
            # print(f"Actions: {actions * np.pi / 180}")
            i = SOARM101CtrlInput()

            # if run with absolute positions, need to add the current joint positions
            # actions shape is (chunk_length, 6), must reshape to (chunk_length * 6,)
            i.joint_positions = (
                np.array(actions)
                .astype(np.float32)
                .reshape(
                    args.chunk_length * 6,
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

        if topic == args.topic_in_soarm_pos:
            o: SOARM101Info = data
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
    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_soarm_pos, SOARM101Info, 1 / hz).start()


if __name__ == "__main__":
    main()
