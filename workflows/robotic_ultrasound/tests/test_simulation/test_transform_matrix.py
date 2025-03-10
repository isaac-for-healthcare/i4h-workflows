import unittest

import numpy as np
import torch
from parameterized import parameterized
from simulation.environments.state_machine.utils import compute_transform_matrix

# Define test cases for parameterized tests
TEST_CASES = [
    # name, ov_point, nifti_point, rotation_matrix, scale, expected_result
    (
        "default_transform",  # Default behavior
        [0.1, 0.2, 0.3],  # ov_point
        [100.0, -300.0, 200.0],  # nifti_point
        None,  # rotation_matrix
        1000.0,  # scale
        torch.tensor(
            [[1000.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1000.0, 0.0], [0.0, -1000.0, 0.0, 400.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),  # expected_matrix
    ),
    (
        "custom_rotation_matrix",
        [0.1, 0.2, 0.3],
        [100.0, 300.0, 200.0],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),  # Identity rotation
        1000.0,
        None,  # Will be calculated in the test
    ),
    (
        "custom_scale",
        [0.1, 0.2, 0.3],
        [50.0, -150.0, 100.0],
        None,
        500.0,  # Half the default scale
        torch.tensor(
            [[500.0, 0.0, 0.0, 0.0], [0.0, 0.0, -500.0, 0.0], [0.0, -500.0, 0.0, 200.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    ),
    (
        "no_scale",
        [0.1, 0.2, 0.3],
        [0.1, -0.3, 0.2],
        None,
        None,  # No scaling
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    ),
    (
        "custom_transform_and_scale",
        [0.1, 0.2, 0.3],
        [250.0, 500.0, 750.0],
        torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64),  # Double each axis
        500.0,  # Custom scale
        None,  # Will be calculated in the test
    ),
    (
        "real_world_example",
        [0.6, 0.0, 0.09],
        [-0.7168, -0.7168, -330.6],
        None,
        1000.0,
        None,  # Will be calculated in the test
    ),
]


class TestTransformMatrix(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transform_matrix_creation(self, name, ov_point, nifti_point, rotation_matrix, scale, expected_matrix):
        result = compute_transform_matrix(ov_point, nifti_point, rotation_matrix=rotation_matrix, scale=scale)

        # For cases where we need to calculate the expected matrix
        if expected_matrix is None:
            # Get the rotation matrix
            if rotation_matrix is None:
                R = torch.tensor([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float64)
            else:
                R = rotation_matrix

            # Apply scaling if needed
            if scale is not None:
                R = R * scale

            # Calculate translation
            ov_point_tensor = torch.tensor(ov_point, dtype=torch.float64).unsqueeze(-1)
            nifti_point_tensor = torch.tensor(nifti_point, dtype=torch.float64)
            t = nifti_point_tensor - (R @ ov_point_tensor).squeeze(-1)

            # Create expected matrix
            expected_matrix = torch.eye(4)
            expected_matrix[:3, :3] = R
            expected_matrix[:3, 3] = t

        # Assert equality with a small tolerance for floating point errors
        self.assertTrue(
            torch.allclose(result, expected_matrix, rtol=1e-5, atol=1e-8),
            f"Failed for {name}.\nExpected:\n{expected_matrix}\nGot:\n{result}",
        )

    def test_transform_point_correctness(self):
        # Define test cases with expected mappings
        test_mappings = [
            # ov_point, nifti_point, rotation_matrix, scale
            ([0.1, 0.2, 0.3], [100.0, -300.0, 200.0], None, 1000.0),
            ([0.5, 0.0, 0.1], [-0.7168, -0.7168, -330.6], None, 1000.0),
            ([0.1, 0.2, 0.3], [50.0, -150.0, 100.0], None, 500.0),
        ]

        for ov_point, nifti_point, rotation_matrix, scale in test_mappings:
            # Create the transformation matrix
            transformation = compute_transform_matrix(
                ov_point, nifti_point, rotation_matrix=rotation_matrix, scale=scale
            )

            # Convert Omniverse point to homogeneous coordinates
            ov_homogeneous = torch.tensor(ov_point + [1.0], dtype=torch.float32)

            # Apply transformation
            transformed_point = transformation @ ov_homogeneous
            transformed_point = transformed_point[:3].cpu().numpy()

            # Compare with expected NIFTI point
            expected_point = np.array(nifti_point)
            self.assertTrue(
                np.allclose(transformed_point, expected_point, rtol=1e-5, atol=1e-8),
                f"Transformation failed to map point correctly.\n"
                f"Omniverse point: {ov_point}\n"
                f"Expected NIFTI point: {expected_point}\n"
                f"Got: {transformed_point}",
            )


if __name__ == "__main__":
    unittest.main()
