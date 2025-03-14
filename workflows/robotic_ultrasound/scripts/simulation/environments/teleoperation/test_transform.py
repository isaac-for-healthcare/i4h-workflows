import unittest
import omni.isaac.lab.utils.math as math_utils  
import numpy as np 
import torch
import argparse

def make_pose(pos, rot):
    """
    Make homogeneous pose matrices from a set of translation vectors and rotation matrices.

    Args:
        pos (torch.Tensor): batch of position vectors with last dimension of 3
        rot (torch.Tensor): batch of rotation matrices with last 2 dimensions of (3, 3)

    Returns:
        pose (torch.Tensor): batch of pose matrices with last 2 dimensions of (4, 4)
    """
    assert isinstance(pos, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(rot, torch.Tensor), "Input must be a torch tensor"
    assert pos.shape[:-1] == rot.shape[:-2]
    assert pos.shape[-1] == rot.shape[-2] == rot.shape[-1] == 3
    pose = torch.zeros(pos.shape[:-1] + (4, 4), dtype=pos.dtype, device=pos.device)
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose

def matrix_from_pos_quat(pos, quat):
    """Convert position and quaternion to a 4x4 transformation matrix.
    
    Args:
        pos (torch.Tensor): Position vector of shape (1, 3)
        quat (torch.Tensor): Quaternion of shape (1, 4) in (w, x, y, z) format
        
    Returns:
        torch.Tensor: 4x4 homogeneous transformation matrix
    """
    # Type assertions
    assert isinstance(pos, torch.Tensor), "Position must be a torch.Tensor"
    assert isinstance(quat, torch.Tensor), "Quaternion must be a torch.Tensor"
    
    # Shape assertions
    assert pos.shape == (1, 3), f"Position must have shape (1, 3), got {pos.shape}"
    assert quat.shape == (1, 4), f"Quaternion must have shape (1, 4), got {quat.shape}"
    
    # Datatype assertions
    assert pos.dtype == torch.float64, "Position must be double precision (float64)"
    assert quat.dtype == torch.float64, "Quaternion must be double precision (float64)"
    
    # Convert quaternion to rotation matrix
    rot = math_utils.matrix_from_quat(quat)
    
    # Create transformation matrix
    transform = make_pose(pos, rot)
    
    return transform

def quat_from_euler_xyy_deg(roll, pitch, yaw):
    euler_angles = np.array([roll, pitch, yaw])
    euler_angles_rad = np.radians(euler_angles)
    euler_angles_rad = torch.tensor(euler_angles_rad).double()
    quat_sim_to_nifti = math_utils.quat_from_euler_xyz(roll=euler_angles_rad[0], pitch=euler_angles_rad[1], yaw=euler_angles_rad[2])
    quat_sim_to_nifti = quat_sim_to_nifti.unsqueeze(0)
    return quat_sim_to_nifti

class TestQuaternionVsMatrixTransforms(unittest.TestCase):
    def setUp(self):
        # Initialize test data
        self.pos_ee_from_organ = torch.tensor([12.0, 23.0, -19.0]).unsqueeze(0).double()
        self.quat_ee_from_organ = quat_from_euler_xyy_deg(12.0, 23.0, -19.0)
        self.pose_ee_from_organ = matrix_from_pos_quat(self.pos_ee_from_organ, self.quat_ee_from_organ)

        # transform probe orientations 
        self.quat_probe_to_probe_us = quat_from_euler_xyy_deg(0.0, 0.0, -90.0)
        self.pose_probe_to_probe_us = matrix_from_pos_quat(torch.zeros(1, 3).double(), self.quat_probe_to_probe_us)
        
        # Setup sim to nifti transformation
        self.quat_sim_to_nifti = quat_from_euler_xyy_deg(90.0, 180.0, 0.0)
        self.trans_sim_to_nifti = torch.zeros(1, 3).double()
        self.trans_sim_to_nifti[0, 2] = -390.0 / 1000.0
        self.pose_sim_to_nifti = matrix_from_pos_quat(self.trans_sim_to_nifti, self.quat_sim_to_nifti)

        self.verbose = False  # Add verbose flag

    def test_matrix_vs_quaternion_transformation(self):
        # Method 1: Using 4x4 matrix multiplication
        pose_ee_in_nifti_frame = torch.matmul(self.pose_sim_to_nifti, self.pose_ee_from_organ)

        # Method 2: Using quaternion operations
        quat = math_utils.quat_mul(self.quat_sim_to_nifti, self.quat_ee_from_organ)
        pos = math_utils.quat_apply(self.quat_sim_to_nifti, self.pos_ee_from_organ) + self.trans_sim_to_nifti
        pose_quat_pos = matrix_from_pos_quat(pos, quat)

        # Assert the results are equal
        self.assertTrue(
            torch.allclose(pose_ee_in_nifti_frame, pose_quat_pos),
            "Matrix multiplication and quaternion transformation methods produce different results"
        )

        # Additional shape checks
        self.assertEqual(
            pose_ee_in_nifti_frame.shape, 
            (1, 4, 4), 
            f"Expected shape (1, 4, 4), got {pose_ee_in_nifti_frame.shape}"
        )

    def test_three_transformations(self):
        # Method 1: Using 4x4 matrix multiplication
        pose_ee_in_nifti_frame = torch.matmul(self.pose_sim_to_nifti, torch.matmul(self.pose_probe_to_probe_us, self.pose_ee_from_organ))

        # Method 2: Using quaternion operations
        quat = math_utils.quat_mul(self.quat_probe_to_probe_us, self.quat_ee_from_organ)
        pos = math_utils.quat_apply(self.quat_probe_to_probe_us, self.pos_ee_from_organ)

        quat = math_utils.quat_mul(self.quat_sim_to_nifti, quat)
        pos = math_utils.quat_apply(self.quat_sim_to_nifti, pos) + self.trans_sim_to_nifti
        pose_quat_pos = matrix_from_pos_quat(pos, quat)

        print("\nMatrix multiplication method result:")
        print(pose_ee_in_nifti_frame)
        print("\nQuaternion transformation method result:")
        print(pose_quat_pos)
        print("\nDifference between methods:")
        print(torch.abs(pose_ee_in_nifti_frame - pose_quat_pos))

        # Assert the results are equal
        self.assertTrue(
            torch.allclose(pose_ee_in_nifti_frame, pose_quat_pos),
            "Matrix multiplication and quaternion transformation methods produce different results"
        )

        # Additional shape checks
        self.assertEqual(
            pose_ee_in_nifti_frame.shape, 
            (1, 4, 4), 
            f"Expected shape (1, 4, 4), got {pose_ee_in_nifti_frame.shape}"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed transformation results')
    args = parser.parse_args()
    
    # Set verbose flag for all test cases
    TestQuaternionVsMatrixTransforms.verbose = args.verbose
    
    unittest.main(argv=['first-arg-is-ignored'], exit=False)



