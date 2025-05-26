import numpy as np
from scipy.spatial import cKDTree


def compute_trajectory_overlap_and_distance(
    ground_truth_traj: np.ndarray, predicted_traj: np.ndarray, radius: float = 0.01
) -> tuple[float, float]:
    """Computes the percentage of ground truth points within a given radius of predicted points
    and the average minimum distance from ground truth points to predicted points.

    Args:
        ground_truth_traj: The ground truth trajectory.
        predicted_traj: The predicted trajectory.
        radius: The radius to compute the overlap and distance.
    """
    if ground_truth_traj.ndim != 2 or ground_truth_traj.shape[1] != 3:  # Check for N x 3 shape
        raise ValueError(f"Ground truth trajectory must be of shape [t, 3], got {ground_truth_traj.shape}")
    if predicted_traj.ndim != 2 or predicted_traj.shape[1] != 3:  # Check for N x 3 shape
        raise ValueError(f"Predicted trajectory must be of shape [t, 3], got {predicted_traj.shape}")

    if predicted_traj.shape[0] == 0:
        print("Warning: Predicted trajectory is empty. Returning 0 overlap and inf distance.")
        return 0.0, np.inf
    if ground_truth_traj.shape[0] == 0:
        print("Warning: Ground truth trajectory is empty. Returning 0 overlap and 0 distance.")
        return 0.0, 0.0

    tree = cKDTree(predicted_traj)
    distances, _ = tree.query(ground_truth_traj)

    percentage = 100.0 * np.sum(distances < radius) / len(ground_truth_traj)
    avg_distance = np.mean(distances)

    return percentage, avg_distance
