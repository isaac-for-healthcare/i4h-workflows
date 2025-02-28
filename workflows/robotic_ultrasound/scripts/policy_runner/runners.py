import os
import torch
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


class PI0PolicyRunner:
    """
    Policy runner for PI0 policy.
    This is the client side, should work together with the PI0 server.

    """
    def __init__(
        self,
        host="0.0.0.0",
        port=8000,
        task_description="Conduct a ultrasound scan on the liver.",
        send_joints=False,
        rti_license_file=None,
        domain_id=0,
        topic_out="topic_franka_ctrl",
    ):
        self.client = _websocket_client_policy.WebsocketClientPolicy(host, port)
        # Prompt for the model
        self.task_description = task_description
        self.writer = None
        if send_joints:
            if rti_license_file is None or not os.path.isabs(rti_license_file):
                raise ValueError("RTI license file must be an existing absolute path.")
            os.environ["RTI_LICENSE_FILE"] = rti_license_file
            import rti.connextdds as dds
            from rti_dds.schemas.franka_ctrl import FrankaCtrlInput

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
        return self.client.infer(element)["actions"]

    def send_joint_states(self, joint_states):
        joint_states = joint_states.astype(float).tolist()
        self.writer.write(FrankaCtrlInput(joint_positions=joint_states))
