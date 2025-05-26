import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_ground_truth_data(
    data_path: str,
    episode_idx: int,
    gt_actions_key: str = "data/demo_0/observations/robot_obs",
    gt_states_key: str = "data/demo_0/state",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Loads ground truth actions and states for a given episode from an HDF5 file.

    Args:
        data_path: The path to the HDF5 file.
        episode_idx: The index of the episode.
        gt_actions_key: The key to the ground truth actions.
        gt_states_key: The key to the ground truth states.
    """
    try:
        with h5py.File(data_path, "r") as f:
            gt_actions = f[gt_actions_key][:, 0, :]
            gt_states = f[gt_states_key][:]
        return gt_actions, gt_states
    except Exception as err:
        print(f"Error loading ground truth for episode {episode_idx}: {err}. Returning None.")
        return None, None


def filter_scanning_points(
    gt_actions: np.ndarray, gt_states: np.ndarray, min_distance_threshold: float = 1e-3
) -> np.ndarray | None:
    """Filters ground truth actions to get scanning points and removes points too close to their predecessors.

    Args:
        gt_actions: The ground truth actions.
        gt_states: The ground truth states.
        min_distance_threshold: The minimum distance threshold for filtering points.
    """
    scanning_mask = gt_states == 3
    if not np.any(scanning_mask):
        return None

    scanning_points_gt = gt_actions[scanning_mask][:, :3]

    if scanning_points_gt.shape[0] == 0:
        return np.array([]).reshape(0, 3)

    # Filter points from the end if they are too close to their predecessor
    num_points_to_keep = scanning_points_gt.shape[0]
    if num_points_to_keep > 1:  # Need at least 2 points to compare
        for i in range(scanning_points_gt.shape[0] - 1, 0, -1):
            distance = np.linalg.norm(scanning_points_gt[i] - scanning_points_gt[i - 1])
            if distance < min_distance_threshold:
                num_points_to_keep = i
            else:
                break
    return scanning_points_gt[:num_points_to_keep]


def load_predicted_trajectory(pred_file_path: str, pred_data_key: str = "robot_obs") -> np.ndarray | None:
    """Loads predicted trajectory from an .npz file.

    Args:
        pred_file_path: The path to the .npz file.
        pred_data_key: The key to the predicted data.
    """
    try:
        pred_data = np.load(pred_file_path)
        pred_traj = pred_data[pred_data_key][:, 0, 0, :3]
        return pred_traj
    except Exception as e:
        print(f"Error loading predicted trajectory from {pred_file_path}: {e}")
        return None


def plot_3d_trajectories(
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    save_path: str,
    gt_label: str = "Ground Truth",
    pred_label: str = "Predicted Trajectory",
    pred_color: str = "red",
    gt_color: str = "blue",
    title: str = "3D Trajectories",
) -> None:
    """Plots ground truth and predicted 3D trajectories and saves the plot.

    Args:
        gt_trajectory: The ground truth trajectory.
        pred_trajectory: The predicted trajectory.
        save_path: The path to save the plot.
        gt_label: The label for the ground truth trajectory.
        pred_label: The label for the predicted trajectory.
        pred_color: The color for the predicted trajectory.
        gt_color: The color for the ground truth trajectory.
        title: The title of the plot.
    """
    if gt_trajectory.shape[1] != 3 or pred_trajectory.shape[1] != 3:
        raise ValueError("Both trajectories must have shape [t, 3] for 3D coordinates.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], label=gt_label, color=gt_color)
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], label=pred_label, color=pred_color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved 3D trajectory plot to {save_path}")
    plt.close(fig)  # Close figure to save memory


def plot_success_rate_vs_radius(
    radius: np.ndarray,
    all_methods_mean_rates: dict,
    all_methods_ci_lower: dict,
    all_methods_ci_upper: dict,
    method_details: dict,
    save_path: str,
    title: str = "Mean Success Rate vs. Radius (95% CI)",
) -> None:
    """Plots success rate vs. radius for multiple methods and saves the plot.

    Args:
        radius: The radius to plot.
        all_methods_mean_rates: The mean success rates for each method.
        all_methods_ci_lower: The lower confidence intervals for each method.
        all_methods_ci_upper: The upper confidence intervals for each method.
        method_details: The details for each method.
        save_path: The path to save the plot.
        title: The title of the plot.
    """
    plt.figure(figsize=(12, 8))
    for method_name, mean_rates in all_methods_mean_rates.items():
        details = method_details[method_name]
        plt.plot(radius, mean_rates, "o-", label=details["label"], color=details["color"])
        plt.fill_between(
            radius,
            all_methods_ci_lower[method_name],
            all_methods_ci_upper[method_name],
            color=details["color"],
            alpha=0.2,
            label=f"{details['label']} 95% CI",
        )

    plt.xlabel("Radius (m)")
    plt.ylabel("Success Rate (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 101)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved success rate vs. radius plot to {save_path}")
    plt.close(plt.gcf())  # Close the current figure
