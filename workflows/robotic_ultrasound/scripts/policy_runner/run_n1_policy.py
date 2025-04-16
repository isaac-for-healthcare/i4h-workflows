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

import torch
import numpy as np
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.subscriber import SubscriberWithCallback
from PIL import Image

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy

current_state = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,
}

# Prevent JAX from preallocating all memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main():
    parser = argparse.ArgumentParser(description="Run the openpi0 policy runner")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/yunliu/Workspace/Code/Isaac-GR00T/demo_data/us-checkpoints-n1-640-wo/checkpoint-2500/",
        help="checkpoint path. Default to use policy checkpoint in the latest assets.",
    )
    parser.add_argument(
        "--rti_license_file", type=str, default=os.getenv("RTI_LICENSE_FILE"), help="the path of rti_license_file."
    )
    parser.add_argument("--domain_id", type=int, default=0, help="domain id.")
    parser.add_argument("--height", type=int, default=480, help="input image height.")
    parser.add_argument("--width", type=int, default=640, help="input image width.")
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
        "--topic_in_franka_pos",
        type=str,
        default="topic_franka_info",
        help="topic name to consume franka pos.",
    )
    parser.add_argument(
        "--topic_out",
        type=str,
        default="topic_franka_ctrl",
        help="topic name to publish generated franka actions.",
    )
    parser.add_argument("--verbose", type=bool, default=False, help="whether to print the log.")
    parser.add_argument(
        "--data_config",
        type=str,
        default="single_panda_us",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--modality_keys", nargs="+", type=str, default=["panda_hand"])
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy: BasePolicy = Gr00tPolicy(
        model_path=args.ckpt_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    

    if args.rti_license_file is not None:
        if not os.path.isabs(args.rti_license_file):
            raise ValueError("RTI license file must be an existing absolute path.")
        os.environ["RTI_LICENSE_FILE"] = args.rti_license_file

    hz = 30

    class PolicyPublisher(Publisher):
        def __init__(self, topic: str, domain_id: int):
            super().__init__(topic, FrankaCtrlInput, 1 / hz, domain_id)

        def produce(self, dt: float, sim_time: float):
            # Process camera images directly to numpy arrays
            def _process_camera_image(buffer, height, width):
                img_buffer = np.frombuffer(buffer, dtype=np.uint8)
                return img_buffer.reshape(height, width, 3)
            
            # Get images from camera buffers
            room_img = _process_camera_image(current_state["room_cam"], args.height, args.width)
            wrist_img = _process_camera_image(current_state["wrist_cam"], args.height, args.width)
            joint_pos = current_state["joint_pos"]
            
            # Prepare input data with batch dimension for model
            data_point = {
                "video.room": np.expand_dims(room_img, axis=0),
                "video.wrist": np.expand_dims(wrist_img, axis=0),
                "state.panda_hand": np.expand_dims(np.array(joint_pos), axis=0),
                "action.panda_hand": np.zeros((16, 6), dtype=np.float32),
                "annotation.human.task_description": "Perform a liver ultrasound.",
            }
            actions = policy.get_action(data_point)
            i = FrankaCtrlInput()
            # actions are relative positions, if run with absolute positions, need to add the current joint positions
            # actions shape is (16, 6), must reshape to (96,)
            i.joint_positions = (
                np.array(actions["action.panda_hand"])
                .astype(np.float32)
                .reshape(
                    16 * 6,
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
