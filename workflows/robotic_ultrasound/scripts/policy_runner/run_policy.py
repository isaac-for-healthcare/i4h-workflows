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
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.subscriber import SubscriberWithCallback
from PIL import Image
from simulation.utils.common import resolve_checkpoint_path

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
        default="nvidia/Liver_Scan_Pi0_Cosmos_Rel",
        help="Checkpoint path or HF repo id for the policy model.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="Perform a liver ultrasound.",
        help="Task description for the policy.",
    )

    pi0_group = parser.add_argument_group("PI0 Policy Arguments")
    pi0_group.add_argument(
        "--repo_id",
        type=str,
        default="i4h/sim_liver_scan",
        help="LeRobot repo id for the dataset norm. Default is `i4h/sim_liver_scan`.",
    )

    gr00tn1_group = parser.add_argument_group("GR00T N1 Policy Arguments")
    gr00tn1_group.add_argument(
        "--data_config",
        type=str,
        default="single_panda_us",
        help="Data config name for GR00T N1 policy.",
    )
    gr00tn1_group.add_argument(
        "--embodiment_tag",
        type=str,
        default="new_embodiment",
        help="The embodiment tag for the GR00T N1 model.",
    )
    parser.add_argument(
        "--rti_license_file", type=str, default=os.getenv("RTI_LICENSE_FILE"), help="the path of rti_license_file."
    )
    parser.add_argument("--domain_id", type=int, default=0, help="domain id.")
    parser.add_argument("--height", type=int, default=224, help="input image height.")
    parser.add_argument("--width", type=int, default=224, help="input image width.")
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
        "--policy", type=str, default="pi0", choices=["pi0", "gr00tn1"], help="policy type to use (pi0 or gr00tn1)."
    )
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=50,
        help="Length of the action chunk inferred by the policy.",
    )
    args = parser.parse_args()
    args.ckpt_path = resolve_checkpoint_path(args.ckpt_path)

    if args.policy == "pi0":
        from policy_runner.pi0.runners import PI0PolicyRunner

        policy = PI0PolicyRunner(ckpt_path=args.ckpt_path, repo_id=args.repo_id, task_description=args.task_description)
    elif args.policy == "gr00tn1":
        from policy_runner.gr00tn1.runners import GR00TN1PolicyRunner

        policy = GR00TN1PolicyRunner(
            ckpt_path=args.ckpt_path,
            data_config=args.data_config,
            embodiment_tag=args.embodiment_tag,
            task_description=args.task_description,
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
            r_cam_buffer = np.frombuffer(current_state["room_cam"], dtype=np.uint8)
            room_img = Image.fromarray(r_cam_buffer.reshape(args.height, args.width, 3), "RGB")
            w_cam_buffer = np.frombuffer(current_state["wrist_cam"], dtype=np.uint8)
            wrist_img = Image.fromarray(w_cam_buffer.reshape(args.height, args.width, 3), "RGB")
            joint_pos = current_state["joint_pos"]
            actions = policy.infer(
                room_img=np.array(room_img),
                wrist_img=np.array(wrist_img),
                current_state=np.array(joint_pos[:7]),
            )
            i = FrankaCtrlInput()
            # actions are relative positions, if run with absolute positions, need to add the current joint positions
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
