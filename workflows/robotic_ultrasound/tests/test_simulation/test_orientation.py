import unittest

import numpy as np
import torch

# Define the default rotation matrix used in the function
DEFAULT_ROTATION_MATRIX = torch.tensor([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float64)
# Define explicit test values for defaults (not using function defaults)
TEST_OV_DOWN_QUAT = [0, 1, 0, 0]  # [w, x, y, z]
TEST_ORGAN_DOWN_QUAT = [-np.pi / 2, 0, 0]  # [x, y, z]

# Define test cases for parameterized tests
TEST_CASES = [
    # name, ov_quat, rotation_matrix, ov_down_quat, organ_down_quat, expected_result
    (
        "default_down_quaternion",  # Test with default down settings
        [0, 1, 0, 0],  # ov_quat: Omniverse "down" quaternion [w, x, y, z]
        DEFAULT_ROTATION_MATRIX,  # Use explicit rotation matrix
        TEST_OV_DOWN_QUAT,  # Use explicit ov_down_quat
        TEST_ORGAN_DOWN_QUAT,  # Use explicit organ_down_quat
        np.array([-np.pi / 2, 0.0, 0.0]),  # Expected result
    ),
    (
        "rotated_quaternion",  # Test with a quaternion rotated 90 degrees around Y
        [0.7071, 0, 0.7071, 0],  # ov_quat: 90° rotation around Y axis [w, x, y, z]
        DEFAULT_ROTATION_MATRIX,  # Use explicit rotation matrix
        TEST_OV_DOWN_QUAT,  # Use explicit ov_down_quat
        TEST_ORGAN_DOWN_QUAT,  # Use explicit organ_down_quat
        None,  # Will verify using rotation matrices
    ),
    (
        "custom_down_quaternions",  # Test with custom down definitions
        [0.5, 0.5, 0.5, 0.5],  # ov_quat: arbitrary rotation
        DEFAULT_ROTATION_MATRIX,  # Use explicit rotation matrix
        [0, 0, 1, 0],  # ov_down_quat: different "down" direction in Omniverse
        [0, -np.pi / 2, 0],  # organ_down_quat: different "down" in organ system
        None,  # Will be calculated in test
    ),
    (
        "identity_quaternion",  # Test with identity quaternion
        [1, 0, 0, 0],  # ov_quat: identity quaternion [w, x, y, z]
        DEFAULT_ROTATION_MATRIX,  # Use explicit rotation matrix
        TEST_OV_DOWN_QUAT,  # Use explicit ov_down_quat
        TEST_ORGAN_DOWN_QUAT,  # Use explicit organ_down_quat
        None,  # Will be calculated in test
    ),
    (
        "real_world_example",
        [0.3758, 0.8235, 0.3686, 0.2114],
        DEFAULT_ROTATION_MATRIX,  # Use explicit rotation matrix
        TEST_OV_DOWN_QUAT,  # Use explicit ov_down_quat
        TEST_ORGAN_DOWN_QUAT,  # Use explicit organ_down_quat
        None,  # Will be calculated in test
    ),
    (
        "custom_rotation_matrix",
        [0, 1, 0, 0],  # ov_quat: Omniverse "down" quaternion [w, x, y, z]
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64),  # Identity rotation matrix
        TEST_OV_DOWN_QUAT,  # Use explicit ov_down_quat
        TEST_ORGAN_DOWN_QUAT,  # Use explicit organ_down_quat
        None,  # Will be calculated in test
    ),
]

INVERSE_TEST_CASES = [
    # ov_quat, rotation_matrix
    ([1, 0, 0, 0], DEFAULT_ROTATION_MATRIX),  # Identity
    ([0, 1, 0, 0], DEFAULT_ROTATION_MATRIX),  # 180° around X
    ([0.7071, 0, 0.7071, 0], DEFAULT_ROTATION_MATRIX),  # 90° around Y
    ([0.7071, 0, 0, 0.7071], DEFAULT_ROTATION_MATRIX),  # 90° around Z
    ([0.9239, 0.3827, 0, 0], DEFAULT_ROTATION_MATRIX),  # 45° around X
]


if __name__ == "__main__":
    unittest.main()
