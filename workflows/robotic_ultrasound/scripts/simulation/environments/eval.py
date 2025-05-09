import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
episode = 4
holdstep=50
data_root = "../data/hdf5/heldout-test50/"


def plot_3d_trajectories(trajectory1: np.ndarray, trajectory2: np.ndarray,
                         label1: str = "Trajectory 1", label2: str = "Trajectory 2",
                         title: str = "3D Trajectories") -> None:
    """
    Plots two 3D trajectories.

    Args:
        trajectory1 (np.ndarray): Array of shape [t1, 3] for the first trajectory.
        trajectory2 (np.ndarray): Array of shape [t2, 3] for the second trajectory.
        label1 (str): Label for the first trajectory.
        label2 (str): Label for the second trajectory.
    """
    if trajectory1.shape[1] != 3 or trajectory2.shape[1] != 3:
        raise ValueError("Both trajectories must have shape [t, 3] for 3D coordinates.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2], label=label1, color='blue')
    ax.plot(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], label=label2, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()

from scipy.spatial import cKDTree


def compute_trajectory_overlap_and_distance(
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    radius: float = 0.01
) -> tuple[float, float]:
    """
    Computes:
      - the percentage of points in trajectory1 within a given radius of trajectory2, and
      - the average minimum distance from trajectory1 to trajectory2.

    Args:
        trajectory1 (np.ndarray): Array of shape [t1, 3].
        trajectory2 (np.ndarray): Array of shape [t2, 3].
        radius (float): Threshold to consider a point "on the trajectory".

    Returns:
        tuple:
            - percentage (float): % of trajectory1 points within `radius` of trajectory2.
            - avg_distance (float): Average distance from trajectory1 points to nearest point in trajectory2.
    """
    if trajectory1.shape[1] != 3 or trajectory2.shape[1] != 3:
        raise ValueError("Both trajectories must be of shape [t, 3]")

    tree = cKDTree(trajectory2)
    distances, _ = tree.query(trajectory1)

    percentage = 100.0 * np.sum(distances < radius) / len(trajectory1)
    avg_distance = np.mean(distances)

    return percentage, avg_distance

success_rates = []
avg_distances = []
for e in range(episode):
    # Load the data
    data_path = f"{data_root}/data_{e}.hdf5"
    with h5py.File(data_path, "r") as f:
        # Get the action data
        gt = f["data/demo_0"]['abs_action'][:]
        state = f["data/demo_0"]['state'][:]
    pred = np.load(f"{data_root}/pi0_600_robot_obs_{e}.npz")['robot_obs']
    scanning = gt[state == 3][:,:3][:-holdstep+1]
    pred_traj = pred[:,0,0,:3]
    success_rate, avg_distance = compute_trajectory_overlap_and_distance(scanning, pred_traj)
    plot_3d_trajectories(scanning, pred_traj, label1="Ground Truth", label2="Predicted Trajectory", title=f"sucess rate: {success_rate:.4f}%, avg distance: {avg_distance:.4f}")
    success_rates.append(success_rate)
    avg_distances.append(avg_distance)
    print(e, success_rate)
print("Average success rate: ", np.mean(success_rates))
print("Average distance: ", np.mean(avg_distances))
