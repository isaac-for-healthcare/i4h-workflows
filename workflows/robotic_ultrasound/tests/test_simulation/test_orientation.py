import unittest

import numpy as np
from parameterized import parameterized
from scipy.spatial.transform import Rotation
from simulation.environments.state_machine.utils import ov_to_nifti_orientation

# Define test cases for parameterized tests
TEST_CASES = [
    # name, ov_quat, ov_down_quat, organ_down_quat, expected_result
    (
        "default_down_quaternion",  # Test with default down settings
        [0, 1, 0, 0],  # ov_quat: Omniverse "down" quaternion [w, x, y, z]
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        np.array([0.0, 0.0, -90.0]),  # Expected result
    ),
    (
        "rotated_quaternion",  # Test with a quaternion rotated 90 degrees around Y
        [0.7071, 0, 0.7071, 0],  # ov_quat: 90° rotation around Y axis [w, x, y, z]
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        None,  # Will verify using rotation matrices
    ),
    (
        "custom_down_quaternions",  # Test with custom down definitions
        [0.5, 0.5, 0.5, 0.5],  # ov_quat: arbitrary rotation
        [0, 0, 1, 0],  # ov_down_quat: different "down" direction in Omniverse
        [0, -90, 0],  # organ_down_quat: different "down" in organ system
        None,  # Will be calculated in test
    ),
    (
        "identity_quaternion",  # Test with identity quaternion
        [1, 0, 0, 0],  # ov_quat: identity quaternion [w, x, y, z]
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        None,  # Will be calculated in test
    ),
    (
        "real_world_example",
        [0.3758, 0.8235, 0.3686, 0.2114],
        None,  # Use default ov_down_quat
        None,  # Use default organ_down_quat
        None,  # Will be calculated in test
    ),
]

INVERSE_TEST_CASES = [
    ([1, 0, 0, 0],),  # Identity
    ([0, 1, 0, 0],),  # 180° around X
    ([0.7071, 0, 0.7071, 0],),  # 90° around Y
    ([0.7071, 0, 0, 0.7071],),  # 90° around Z
    ([0.9239, 0.3827, 0, 0],),  # 45° around X
]


class TestOrientationConversion(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_orientation_conversion(self, name, ov_quat, ov_down_quat, organ_down_quat, expected_result):
        # Call the function with the test parameters
        result_euler = ov_to_nifti_orientation(
            ov_quat=ov_quat, ov_down_quat=ov_down_quat, organ_down_quat=organ_down_quat
        )

        # For cases with expected Euler angles, compare directly
        if expected_result is None:
            # Calculate the expected result directly
            ov_quat_scipy = [ov_quat[1], ov_quat[2], ov_quat[3], ov_quat[0]]
            ov_rot = Rotation.from_quat(ov_quat_scipy)

            if ov_down_quat is None:
                ov_down_quat = [0, 1, 0, 0]  # Default [w, x, y, z]
            if organ_down_quat is None:
                organ_down_quat = [0, 0, -90]  # Default [x, y, z]

            ov_down_scipy = [ov_down_quat[1], ov_down_quat[2], ov_down_quat[3], ov_down_quat[0]]
            ov_down_rot = Rotation.from_quat(ov_down_scipy)

            organ_down_rot = Rotation.from_euler("xyz", organ_down_quat, degrees=True)

            reference_mapping = organ_down_rot * ov_down_rot.inv()
            expected_rot = reference_mapping * ov_rot
            expected_result = expected_rot.as_euler("xyz", degrees=True)

        # Compare directly with the result
        self.assertTrue(
            np.allclose(result_euler, expected_result, rtol=1e-5, atol=1e-3),
            f"Failed for {name}.\nExpected: {expected_result}\nGot: {result_euler}",
        )

    def test_orientation_mapping_consistency(self):
        """Test that relative orientations are preserved in the mapping."""
        # Define a reference orientation and a rotated orientation in Omniverse
        ov_ref = np.array([0, 1, 0, 0])  # Reference "down" orientation [w, x, y, z]
        # Use a rotation that's less likely to cause gimbal lock
        ov_rotated = np.array([0, 0.9239, 0, 0.3827])  # 45° around Z axis

        # Convert both to organ system
        organ_ref = ov_to_nifti_orientation(ov_ref)
        organ_rotated = ov_to_nifti_orientation(ov_rotated)

        # Calculate the angular difference between the two in Omniverse
        # Convert to Rotation objects for scipy
        rot_ov_ref = Rotation.from_quat([ov_ref[1], ov_ref[2], ov_ref[3], ov_ref[0]])
        rot_ov_rotated = Rotation.from_quat([ov_rotated[1], ov_rotated[2], ov_rotated[3], ov_rotated[0]])
        ov_angle = rot_ov_ref.inv() * rot_ov_rotated
        ov_angle_deg = np.linalg.norm(ov_angle.as_rotvec(degrees=True))

        # Calculate the angular difference in the organ system using rotation matrices
        rot_organ_ref = Rotation.from_euler("xyz", organ_ref, degrees=True)
        rot_organ_rotated = Rotation.from_euler("xyz", organ_rotated, degrees=True)
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
    def test_inverse_mapping(self, ov_quat):
        mapped_euler = ov_to_nifti_orientation(
            ov_quat=ov_quat,
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
