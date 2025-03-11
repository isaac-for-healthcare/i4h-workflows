import torch
from openpi.policies import policy_config
from openpi_client import image_tools
from policy_runner.config import get_config


class PI0PolicyRunner:
    """
    Policy runner for PI0 policy, based on the openpi library.

    Args:
        ckpt_path: Path to the checkpoint file.
        repo_id: Repository ID of the original training dataset.
        task_description: Task description. Default is "Conduct a ultrasound scan on the liver."

    """

    def __init__(
        self,
        ckpt_path,
        repo_id,
        task_description="Conduct a ultrasound scan on the liver.",
    ):
        config = get_config(name="robotic_ultrasound", repo_id=repo_id)
        print(f'loading model from {ckpt_path}...')
        self.model = policy_config.create_trained_policy(config, ckpt_path)
        # Prompt for the model
        self.task_description = task_description

    def infer(self, room_img, wrist_img, current_state) -> torch.Tensor:
        room_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(room_img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

        element = {
            "observation/image": room_img,
            "observation/wrist_image": wrist_img,
            "observation/state": current_state,
            "prompt": self.task_description,
        }
        # Query model to get action
        return self.model.infer(element)["actions"]
