import numpy as np
from omni.isaac.lab.utils import convert_dict_to_backend


def get_np_images(env):
    """Get numpy images from the environment."""
    third_person_img = convert_dict_to_backend(env.unwrapped.scene["room_camera"].data.output, backend="numpy")["rgb"]
    third_person_img = third_person_img[0, :, :, :3].astype(np.uint8)

    wrist_img1 = convert_dict_to_backend(env.unwrapped.scene["wrist_camera"].data.output, backend="numpy")["rgb"]
    wrist_img1 = wrist_img1[0, :, :, :3].astype(np.uint8)

    return third_person_img, wrist_img1
