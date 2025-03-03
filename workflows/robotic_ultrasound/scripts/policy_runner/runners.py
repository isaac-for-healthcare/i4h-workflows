import os
import torch

import rti.connextdds as dds
from rti_dds.schemas.franka_ctrl import FrankaCtrlInput
from openpi.models import pi0
from openpi.policies import policy_config
from openpi.training.config import DataConfig, TrainConfig
from openpi_client import image_tools
from policy_runner.utils import LeRobotDataConfig


class PI0PolicyRunner:
    """
    Policy runner for PI0 policy.
    This is the client side, should work together with the PI0 server.

    """
    def __init__(
        self,
        ckpt_path,
        repo_id,
        task_description="Conduct a ultrasound scan on the liver.",
        send_joints=False,
        rti_license_file=None,
        domain_id=0,
        topic_out="topic_franka_ctrl",
    ):
        config = TrainConfig(
            name="pi0_scan",
            model=pi0.Pi0Config(),
            data=LeRobotDataConfig(
                repo_id=repo_id,
                base_config=DataConfig(
                    local_files_only=True,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
            ),
        )
        self.model = policy_config.create_trained_policy(config, ckpt_path)
        # Prompt for the model
        self.task_description = task_description
        self.writer = None
        if send_joints:
            if rti_license_file is None or not os.path.isabs(rti_license_file):
                raise ValueError("RTI license file must be an existing absolute path.")
            os.environ["RTI_LICENSE_FILE"] = rti_license_file
            participant = dds.DomainParticipant(domain_id=domain_id)
            topic = dds.Topic(participant, topic_out, FrankaCtrlInput)
            self.writer = dds.DataWriter(participant.implicit_publisher, topic)

    def infer(self, room_img, wrist_img, current_state) -> torch.Tensor:
        room_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(room_img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )

        element = {
            "observation/image": room_img,
            "observation/wrist_image": wrist_img,
            "observation/state": current_state,
            "prompt": self.task_description,
        }
        # Query model to get action
        return self.model.infer(element)["actions"]

    def send_joint_states(self, joint_states):
        joint_states = joint_states.astype(float).tolist()
        self.writer.write(FrankaCtrlInput(joint_positions=joint_states))
