import unittest

import numpy as np
import torch
from parameterized import parameterized
from scipy.spatial.transform import Rotation
from simulation.environments.state_machine.utils import ov_to_nifti_orientation

# Define the default rotation matrix used in the function
DEFAULT_ROTATION_MATRIX = torch.tensor([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float64)

# Define test cases for parameterized tests
TEST_CASES = [
    # name, ov_quat, rotation_matrix, ov_down_quat, organ_down_quat, expected_result
    (
        "default_down_quaternion",  # Test with default down settings
        [0, 1, 0, 0],  # ov_quat: Omniverse "down" quaternion [w, x, y, z]
        None,  # Use default rotation matrix
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        np.array([-np.pi / 2, 0.0, 0.0]),  # Expected result
    ),
    (
        "rotated_quaternion",  # Test with a quaternion rotated 90 degrees around Y
        [0.7071, 0, 0.7071, 0],  # ov_quat: 90° rotation around Y axis [w, x, y, z]
        None,  # Use default rotation matrix
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        None,  # Will verify using rotation matrices
    ),
    (
        "custom_down_quaternions",  # Test with custom down definitions
        [0.5, 0.5, 0.5, 0.5],  # ov_quat: arbitrary rotation
        None,  # Use default rotation matrix
        [0, 0, 1, 0],  # ov_down_quat: different "down" direction in Omniverse
        [0, -np.pi / 2, 0],  # organ_down_quat: different "down" in organ system
        None,  # Will be calculated in test
    ),
    (
        "identity_quaternion",  # Test with identity quaternion
        [1, 0, 0, 0],  # ov_quat: identity quaternion [w, x, y, z]
        None,  # Use default rotation matrix
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        None,  # Will be calculated in test
    ),
    (
        "real_world_example",
        [0.3758, 0.8235, 0.3686, 0.2114],
        None,  # Use default rotation matrix
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        None,  # Will be calculated in test
    ),
    (
        "custom_rotation_matrix",
        [0, 1, 0, 0],  # ov_quat: Omniverse "down" quaternion [w, x, y, z]
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64),  # Identity rotation matrix
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        None,  # Will be calculated in test
    ),
]

INVERSE_TEST_CASES = [
    # ov_quat, rotation_matrix
    ([1, 0, 0, 0], None),  # Identity
    ([0, 1, 0, 0], None),  # 180° around X
    ([0.7071, 0, 0.7071, 0], None),  # 90° around Y
    ([0.7071, 0, 0, 0.7071], None),  # 90° around Z
    ([0.9239, 0.3827, 0, 0], None),  # 45° around X
]


class TestOrientationConversion(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_orientation_conversion(
        self, name, ov_quat, rotation_matrix, ov_down_quat, organ_down_quat, expected_result
    ):
        # Call the function with the test parameters
        result_euler = ov_to_nifti_orientation(
            ov_quat=ov_quat, rotation_matrix=rotation_matrix, ov_down_quat=ov_down_quat, organ_down_quat=organ_down_quat
        )

        # For cases with expected Euler angles, compare directly
        if expected_result is None:
            # Calculate the expected result based on the new implementation
            ov_rot = Rotation.from_quat(ov_quat, scalar_first=True)

            if ov_down_quat is None:
                ov_down_quat = [0, 1, 0, 0]  # Default [w, x, y, z]
            if organ_down_quat is None:
                organ_down_quat = [-np.pi / 2, 0, 0]  # Default [x, y, z]
            if rotation_matrix is None:
                rotation_matrix = DEFAULT_ROTATION_MATRIX

            # Convert to numpy for calculations
            if isinstance(rotation_matrix, torch.Tensor):
                coord_transform = rotation_matrix.numpy()
            else:
                coord_transform = rotation_matrix

            ov_down_rot = Rotation.from_quat(ov_down_quat, scalar_first=True)

            organ_down_rot = Rotation.from_euler("xyz", organ_down_quat, degrees=False)

            # Apply the coordinate transformation to both orientations
            ov_matrix = ov_rot.as_matrix()
            transformed_matrix = coord_transform @ ov_matrix @ coord_transform.T
            transformed_rot = Rotation.from_matrix(transformed_matrix)

            ov_down_matrix = ov_down_rot.as_matrix()
            transformed_down_matrix = coord_transform @ ov_down_matrix @ coord_transform.T
            transformed_down_rot = Rotation.from_matrix(transformed_down_matrix)

            # Calculate the final rotation
            final_rot = organ_down_rot * transformed_down_rot.inv() * transformed_rot
            expected_result = final_rot.as_euler("xyz", degrees=False)

        # Compare directly with the result
        self.assertTrue(
            np.allclose(result_euler, expected_result, rtol=1e-5, atol=1e-3),
            f"Failed for {name}.\nExpected: {expected_result}\nGot: {result_euler}",
        )

    def test_orientation_mapping_consistency(self):
        # Define a reference orientation and a rotated orientation in Omniverse
        ov_ref = np.array([0, 1, 0, 0])  # Reference "down" orientation [w, x, y, z]
        ov_rotated = np.array([0, 0.9239, 0, 0.3827])  # 45° around Z axis

        # Convert both to organ system
        organ_ref = ov_to_nifti_orientation(ov_ref)
        organ_rotated = ov_to_nifti_orientation(ov_rotated)

        # Calculate the angular difference between the two in Omniverse
        # Convert to Rotation objects for scipy
        rot_ov_ref = Rotation.from_quat(ov_ref, scalar_first=True)
        rot_ov_rotated = Rotation.from_quat(ov_rotated, scalar_first=True)
        ov_angle = rot_ov_ref.inv() * rot_ov_rotated
        ov_angle_deg = np.linalg.norm(ov_angle.as_rotvec(degrees=True))

        # Calculate the angular difference in the organ system using rotation matrices
        rot_organ_ref = Rotation.from_euler("xyz", organ_ref, degrees=False)
        rot_organ_rotated = Rotation.from_euler("xyz", organ_rotated, degrees=False)
        organ_angle = rot_organ_ref.inv() * rot_organ_rotated
        organ_angle_deg = np.linalg.norm(organ_angle.as_rotvec(degrees=True))

        # The angles should be approximately the same (the transformation preserves angles)
        self.assertAlmostEqual(
            ov_angle_deg,
            organ_angle_deg,
            delta=2.0,
            msg=f"Angle in Omniverse: {ov_angle_deg}, Angle in organ system: {organ_angle_deg}",
        )

    @parameterized.expand(INVERSE_TEST_CASES)
    def test_inverse_mapping(self, ov_quat, rotation_matrix):
        mapped_euler = ov_to_nifti_orientation(
            ov_quat=ov_quat,
            rotation_matrix=rotation_matrix,
            ov_down_quat=ov_quat,  # Use the input quaternion as reference
            organ_down_quat=[0, 0, 0],  # Target neutral orientation
        )

        # The result should be close to [0,0,0]
        self.assertTrue(
            np.allclose(mapped_euler, [0, 0, 0], rtol=1e-5, atol=1e-3),
            f"Inverse mapping failed for {ov_quat}.\nGot: {mapped_euler}",
        )


if __name__ == "__main__":
    unittest.main()
