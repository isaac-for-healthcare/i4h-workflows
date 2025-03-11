import unittest

import numpy as np
import torch
from parameterized import parameterized
from simulation.environments.state_machine.utils import compute_transform_matrix

# Define test cases for parameterized tests
TEST_CASES = [
    # name, ov_point, nifti_point, rotation_matrix, expected_result
    (
        "default_transform",  # Default behavior
        [0.1, 0.2, 0.3],  # ov_point
        [100.0, -300.0, 200.0],  # nifti_point
        None,  # rotation_matrix
        torch.tensor(
            [[1.0, 0.0, 0.0, 99.9], [0.0, 0.0, -1.0, -299.7], [0.0, -1.0, 0.0, 200.2], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),  # expected_matrix
    ),
    (
        "custom_rotation_matrix",
        [0.1, 0.2, 0.3],
        [100.0, 300.0, 200.0],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),  # Identity rotation
        None,  # Will be calculated in the test
    ),
    (
        "real_world_example",
        [0.6, 0.0, 0.09],  # From example in codebase
        [-0.7168, -0.7168, -330.6],  # From example in codebase
        None,
        None,  # Will be calculated in the test
    ),
]


class TestTransformMatrix(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transform_matrix_creation(self, name, ov_point, nifti_point, rotation_matrix, expected_matrix):
        result = compute_transform_matrix(ov_point, nifti_point, rotation_matrix=rotation_matrix)

        # For cases where we need to calculate the expected matrix
        if expected_matrix is None:
            # Get the rotation matrix
            if rotation_matrix is None:
                R = torch.tensor([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float64)
            else:
                R = rotation_matrix

            # Calculate translation
            ov_point_tensor = torch.tensor(ov_point, dtype=torch.float64).unsqueeze(-1)
            nifti_point_tensor = torch.tensor(nifti_point, dtype=torch.float64)
            t = nifti_point_tensor - (R @ ov_point_tensor).squeeze(-1)

            # Create expected matrix
            expected_matrix = torch.eye(4, dtype=torch.float64)
            expected_matrix[:3, :3] = R
            expected_matrix[:3, 3] = t

        # Assert equality with a small tolerance for floating point errors
        self.assertTrue(
            torch.allclose(result, expected_matrix, rtol=1e-5, atol=1e-8),
            f"Failed for {name}.\nExpected:\n{expected_matrix}\nGot:\n{result}",
        )

    def test_transform_point_correctness(self):
        test_mappings = [
            # ov_point, nifti_point, rotation_matrix
            ([0.1, 0.2, 0.3], [100.0, -300.0, 200.0], None),
            ([0.5, 0.0, 0.1], [-0.7168, -0.7168, -330.6], None),
        ]

        for ov_point, nifti_point, rotation_matrix in test_mappings:
            # Create the transformation matrix
            transformation = compute_transform_matrix(ov_point, nifti_point, rotation_matrix=rotation_matrix)

            # Convert Omniverse point to homogeneous coordinates
            ov_homogeneous = torch.tensor(ov_point + [1.0], dtype=torch.float64)

            # Apply transformation
            transformed_point = transformation @ ov_homogeneous
            transformed_point = transformed_point[:3].cpu().numpy()

            # Compare with expected NIFTI point
            expected_point = np.array(nifti_point, dtype=np.float64)
            self.assertTrue(
                np.allclose(transformed_point, expected_point, rtol=1e-5, atol=1e-8),
                f"Transformation failed to map point correctly.\n"
                f"Omniverse point: {ov_point}\n"
                f"Expected NIFTI point: {expected_point}\n"
                f"Got: {transformed_point}",
            )


if __name__ == "__main__":
    unittest.main()
