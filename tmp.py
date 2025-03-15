import numpy as np

# Define basis vectors for USD coordinate system
v_usd_si = np.array([0, -1, 0])
v_usd_lr = np.array([-1, 0, 0])
v_usd_ap = np.array([0, 0, -1])

# Define basis vectors for mesh coordinate system
v_mesh_si = np.array([0, 0, 1])
v_mesh_lr = np.array([1, 0, 0])
v_mesh_ap = np.array([0, 1, 0])

# def get_organ_to_usd_transform():
#     """
#     Derives a 4x4 transformation matrix to align organ mesh coordinates with USD coordinates.
    
#     Based on the following basis vectors:
#     USD coordinate system:
#         v_usd_si = (0, -1, 0)
#         v_usd_lr = (-1, 0, 0)
#         v_usd_ap = (0, 0, -1)
    
#     Mesh coordinate system:
#         v_mesh_si = (0, 0, 1)
#         v_mesh_lr = (1, 0, 0)
#         v_mesh_ap = (0, 1, 0)
    
#     Returns:
#         numpy.ndarray: 4x4 transformation matrix
#     """    
#     # Create rotation matrices from basis vectors
#     # Each column of the rotation matrix represents where each basis vector maps to
#     R_mesh = np.column_stack((v_mesh_lr, v_mesh_ap, v_mesh_si))
#     R_usd = np.column_stack((v_usd_lr, v_usd_ap, v_usd_si))
    
#     # Calculate rotation matrix from mesh to USD
#     # R_mesh * R = R_usd
#     # R = R_mesh^-1 * R_usd
#     R = np.linalg.inv(R_mesh) @ R_usd
    
#     # Create 4x4 transformation matrix
#     transform = np.eye(4)
#     transform[:3, :3] = R
    
#     return transform


# from math import atan2, degrees


# def rotation_matrix_to_euler_angles(R):
#     """
#     Convert a rotation matrix to Euler angles (in degrees).
#     Uses the convention of rotations around X, Y, Z axes (roll, pitch, yaw).
    
#     Args:
#         R: 3x3 rotation matrix
        
#     Returns:
#         tuple: (roll, pitch, yaw) in degrees
#     """
#     # Check if we're in the singularity known as "Gimbal lock"
#     # This happens when the middle rotation aligns the first and third rotation axes
#     sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
#     singular = sy < 1e-6
    
#     if not singular:
#         x = atan2(R[2,1], R[2,2])  # Roll
#         y = atan2(-R[2,0], sy)     # Pitch
#         z = atan2(R[1,0], R[0,0])  # Yaw
#     else:
#         x = atan2(-R[1,2], R[1,1])
#         y = atan2(-R[2,0], sy)
#         z = 0
    
#     # Convert to degrees
#     return degrees(x), degrees(y), degrees(z)

# # Example usage
# if __name__ == "__main__":
#     transform_matrix = get_organ_to_usd_transform()
#     print("Transformation matrix from organ mesh to USD coordinates:")
#     print(transform_matrix)
    
#     # # Test the transformation with a sample point
#     # test_point_mesh = np.array([v_mesh_lr[0], v_mesh_lr[1], v_mesh_lr[2], 1])  # Homogeneous coordinates
#     # test_point_usd = transform_matrix @ test_point_mesh
#     # print("\nTest point in mesh coordinates:", test_point_mesh[:3])
#     # print("Test point in USD coordinates:", test_point_usd[:3])

#     rotation_matrix = transform_matrix[:3, :3]
#     # Extract Euler angles
#     roll, pitch, yaw = rotation_matrix_to_euler_angles(transform_matrix[:3, :3])
#     print("\nEuler angles (in degrees):")
#     print(f"Roll (X-axis): {roll:.2f}")
#     print(f"Pitch (Y-axis): {pitch:.2f}")
#     print(f"Yaw (Z-axis): {yaw:.2f}")

import numpy as np
from math import atan2, degrees

def get_usd_to_mesh_transform():
    """
    Derives a 4x4 transformation matrix to align USD coordinates with organ mesh coordinates.
    
    Based on the following basis vectors:
    USD coordinate system:
        v_usd_si = (0, -1, 0)
        v_usd_lr = (-1, 0, 0)
        v_usd_ap = (0, 0, -1)
    
    Mesh coordinate system:
        v_mesh_si = (0, 0, 1)
        v_mesh_lr = (1, 0, 0)
        v_mesh_ap = (0, 1, 0)
    
    Returns:
        numpy.ndarray: 4x4 transformation matrix from USD to mesh coordinates
    """
    # Define basis vectors for USD coordinate system
    # v_usd_si = np.array([0, -1, 0])
    # v_usd_lr = np.array([-1, 0, 0])
    # v_usd_ap = np.array([0, 0, -1])
    
    # # Define basis vectors for mesh coordinate system
    # v_mesh_si = np.array([0, 0, 1])
    # v_mesh_lr = np.array([1, 0, 0])
    # v_mesh_ap = np.array([0, 1, 0])
    
    # Create rotation matrices from basis vectors
    R_mesh = np.column_stack((v_mesh_lr, v_mesh_ap, v_mesh_si))
    R_usd = np.column_stack((v_usd_lr, v_usd_ap, v_usd_si))
    
    # Calculate rotation matrix from USD to mesh
    # R_usd * R = R_mesh
    # R = R_usd^-1 * R_mesh
    # R = np.linalg.inv(R_usd) @ R_mesh
    R = np.linalg.inv(R_mesh) @ R_usd
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    
    return transform, R

import omni.isaac.lab.utils.math as math_utils  # noqa: F401, E402
import torch
def quat_from_euler_xyz_deg(roll, pitch, yaw, device="cuda"):
    euler_angles = np.array([roll, pitch, yaw])
    euler_angles_rad = np.radians(euler_angles)
    euler_angles_rad = torch.tensor(euler_angles_rad, device=device).double()
    quat_sim_to_nifti = math_utils.quat_from_euler_xyz(
        roll=euler_angles_rad[0], pitch=euler_angles_rad[1], yaw=euler_angles_rad[2]
    )
    quat_sim_to_nifti = quat_sim_to_nifti.unsqueeze(0)
    return quat_sim_to_nifti


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (in degrees).
    Uses the convention of rotations around X, Y, Z axes (roll, pitch, yaw).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        tuple: (roll, pitch, yaw) in degrees
    """
    # Check if we're in the singularity known as "Gimbal lock"
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = atan2(R[2,1], R[2,2])  # Roll
        y = atan2(-R[2,0], sy)     # Pitch
        z = atan2(R[1,0], R[0,0])  # Yaw
    else:
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
    
    # Convert to degrees
    return degrees(x), degrees(y), degrees(z)

# Example usage
if __name__ == "__main__":
    # Get USD to mesh transform
    usd_to_mesh_transform, usd_to_mesh_rotation = get_usd_to_mesh_transform()
    print("Transformation matrix from USD to organ mesh coordinates:")
    print(usd_to_mesh_transform)
    
    # Extract Euler angles
    roll, pitch, yaw = rotation_matrix_to_euler_angles(usd_to_mesh_rotation)
    print("\nEuler angles for USD to mesh transformation (in degrees):")
    print(f"Roll (X-axis): {roll:.2f}")
    print(f"Pitch (Y-axis): {pitch:.2f}")
    print(f"Yaw (Z-axis): {yaw:.2f}")
    
    # Test the transformation with a sample point
    test_point_mesh = np.array([v_usd_lr[0], v_usd_lr[1], v_usd_lr[2], 1])  # Homogeneous coordinates
    test_point_usd = usd_to_mesh_transform @ test_point_mesh
    print("\nTest point in mesh coordinates:", test_point_mesh[:3])
    print("Test point in USD coordinates:", test_point_usd[:3])
    
    # # For comparison, also calculate the mesh to USD transform
    # mesh_to_usd_transform = np.linalg.inv(usd_to_mesh_transform)
    # print("\nTransformation matrix from organ mesh to USD coordinates (inverse):")
    # print(mesh_to_usd_transform)
    usd_to_mesh_transform = torch.tensor(usd_to_mesh_transform, device="cuda").double()
    print(f"quat1: ", math_utils.quat_from_matrix(usd_to_mesh_transform[:3, :3]))
    print(f"quat2: ", quat_from_euler_xyz_deg(90.0, 180.0, 0.0))