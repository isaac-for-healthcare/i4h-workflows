import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
# import scipy.stats # For more precise CI, but 1.96*SEM is common

episode = 50
holdstep=50
data_root = "/mnt/hdd/cosmos/heldout-test50"
DEFAULT_RADIUS_FOR_PLOTS = 0.01 # Radius used for individual 3D trajectory plots

# Define prediction sources and their labels/colors
PREDICTION_SOURCES = {
    "WCOS": {
        "file_pattern": "pi0-800_robot_obs_{e}.npz",
        "label": "With COSMOS Prediction",
        "color": "red"
    },
    "WOCOS": {
        "file_pattern": "pi0-400_robot_obs_{e}.npz", # Adjust if your GR00T files have a different pattern
        "label": "Without COSMOS Prediction",
        "color": "green"
    }
}

def plot_3d_trajectories(gt_trajectory: np.ndarray, pred_trajectory: np.ndarray, 
                         gt_label: str = "Ground Truth", pred_label: str = "Predicted Trajectory",
                         pred_color: str = 'red',
                         title: str = "3D Trajectories",
                         episode: int = 0) -> None:
    os.makedirs(f"{data_root}/{method_name}", exist_ok=True)
    if gt_trajectory.shape[1] != 3 or pred_trajectory.shape[1] != 3:
        raise ValueError("Both trajectories must have shape [t, 3] for 3D coordinates.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], label=gt_label, color='blue')
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], label=pred_label, color=pred_color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{data_root}/{method_name}/3d_trajectories-{episode}.png")
    # plt.show() # Keep this commented if saving many plots, or save them to files
    # plt.close(fig) # Close figure to save memory

def compute_trajectory_overlap_and_distance(
    ground_truth_traj: np.ndarray,
    predicted_traj: np.ndarray,
    radius: float = 0.01
) -> tuple[float, float]:
    if ground_truth_traj.shape[1] != 3 or predicted_traj.shape[1] != 3:
        raise ValueError("Both trajectories must be of shape [t, 3]")
    print(f"ground_truth_traj.shape: {ground_truth_traj.shape}")
    print(f"predicted_traj.shape: {predicted_traj.shape}")
        
    tree = cKDTree(predicted_traj)
    distances, _ = tree.query(ground_truth_traj)

    percentage = 100.0 * np.sum(distances < radius) / len(ground_truth_traj)
    avg_distance = np.mean(distances)

    return percentage, avg_distance

def plot_success_rate_vs_radius(
    radii: np.ndarray,
    all_methods_mean_rates: dict, # {method_name: [mean_rates_for_radii]}
    all_methods_ci_lower: dict,   # {method_name: [ci_lower_for_radii]}
    all_methods_ci_upper: dict,   # {method_name: [ci_upper_for_radii]}
    method_details: dict,         # From PREDICTION_SOURCES
    title: str = "Mean Success Rate vs. Radius (95% CI)"
) -> None:
    plt.figure(figsize=(12, 8))
    for method_name, mean_rates in all_methods_mean_rates.items():
        details = method_details[method_name]
        plt.plot(radii, mean_rates, 'o-', label=details["label"], color=details["color"])
        plt.fill_between(radii, 
                         all_methods_ci_lower[method_name], 
                         all_methods_ci_upper[method_name], 
                         color=details["color"], alpha=0.2,
                         label=f"{details['label']} 95% CI")

    plt.xlabel("Radius (m)")
    plt.ylabel("Success Rate (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 101)
    plt.tight_layout()
    plt.savefig(f"{data_root}/comparison_success_rate_vs_radius-400v800.png")
    print(f"Saved success rate vs. radius plot to {data_root}/comparison_success_rate_vs_radius.png")
    # plt.show()
    # plt.close()

# --- Main script ---
radii_to_test = np.linspace(0.001, 0.05, 20)

# Store results for each method
results_per_method = {
    method: {
        "original_success_rates": [],
        "original_avg_distances": [],
        "all_episodes_success_rates_per_radius": {r: [] for r in radii_to_test}
    }
    for method in PREDICTION_SOURCES.keys()
}

for e in range(episode):
    print(f"Processing Episode {e}...")
    data_path = f"{data_root}/data_{e}.hdf5"
    try:
        with h5py.File(data_path, "r") as f:
            gt_actions = f["data/demo_0"]['observations/robot_obs'][:,0,:]
            gt_states = f["data/demo_0"]['state'][:]
    except Exception as err:
        print(f"Error loading ground truth for episode {e}: {err}. Skipping episode.")
        for method_name in PREDICTION_SOURCES.keys():
            results_per_method[method_name]["original_success_rates"].append(np.nan)
            results_per_method[method_name]["original_avg_distances"].append(np.nan)
            for r_val in radii_to_test:
                results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val].append(np.nan)
        continue

    scanning_mask = (gt_states == 3)
    if not np.any(scanning_mask):
        print(f"Warning: Episode {e} has no 'scanning' states. Skipping.")
        # Append NaNs for all methods for this episode
        for method_name in PREDICTION_SOURCES.keys():
            results_per_method[method_name]["original_success_rates"].append(np.nan)
            results_per_method[method_name]["original_avg_distances"].append(np.nan)
            for r_val in radii_to_test:
                results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val].append(np.nan)
        continue

    scanning_points_gt = gt_actions[scanning_mask][:, :3]
    
    if holdstep > 1 and scanning_points_gt.shape[0] >= holdstep:
        scanning_gt = scanning_points_gt[:-holdstep+1]
    elif holdstep == 1 and scanning_points_gt.shape[0] > 0:
        scanning_gt = scanning_points_gt
    elif holdstep == 0 and scanning_points_gt.shape[0] > 0:
        scanning_gt = scanning_points_gt


    for method_name, details in PREDICTION_SOURCES.items():
        pred_file_path = f"{data_root}/{details['file_pattern'].format(e=e)}"
        pred_data = np.load(pred_file_path)
        # Consistently use 'robot_obs' and the specified slicing
        pred_traj = pred_data['robot_obs'][:,0,0,:3] 

        
        # 1. Original 3D plots & metrics
        success_rate_orig, avg_distance_orig = compute_trajectory_overlap_and_distance(
            scanning_gt, pred_traj, radius=DEFAULT_RADIUS_FOR_PLOTS
        )
        plot_3d_trajectories(
            scanning_gt, pred_traj, 
            gt_label="Ground Truth", pred_label=details["label"], 
            pred_color=details["color"],
            title=f"Ep {e} - {method_name}: SR ({DEFAULT_RADIUS_FOR_PLOTS}m): {success_rate_orig:.2f}%, AvgDist: {avg_distance_orig:.4f}m",
            episode=e
        )
        results_per_method[method_name]["original_success_rates"].append(success_rate_orig)
        results_per_method[method_name]["original_avg_distances"].append(avg_distance_orig)
        print(f"  {method_name} - Ep {e}: SR ({DEFAULT_RADIUS_FOR_PLOTS}m) = {success_rate_orig:.2f}%, AvgMinDist = {avg_distance_orig:.4f}m")

        # 2. Success rate vs. radius data collection
        tree = cKDTree(pred_traj)
        distances_to_pred, _ = tree.query(scanning_gt)
        for r_val in radii_to_test:
            rate_for_radius = 100.0 * np.sum(distances_to_pred < r_val) / len(scanning_gt)
            results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val].append(rate_for_radius)

# --- Calculate and Print Summaries ---
all_methods_mean_rates_vs_radius = {}
all_methods_ci_lower_vs_radius = {}
all_methods_ci_upper_vs_radius = {}

print("\n--- Overall Summary (using radius=", DEFAULT_RADIUS_FOR_PLOTS, "m) ---")
for method_name in PREDICTION_SOURCES.keys():
    mean_orig_sr = np.nanmean(results_per_method[method_name]["original_success_rates"])
    mean_orig_ad = np.nanmean(results_per_method[method_name]["original_avg_distances"])
    print(f"Method: {method_name}")
    print(f"  Avg Success Rate: {mean_orig_sr:.2f}%")
    print(f"  Avg Min Distance: {mean_orig_ad:.4f}m")

    # Calculate for success_rate_vs_radius plot
    current_method_mean_rates = []
    current_method_ci_lower = []
    current_method_ci_upper = []
    for r_val in radii_to_test:
        rates_at_this_radius = np.array(results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val])
        valid_rates = rates_at_this_radius[~np.isnan(rates_at_this_radius)]
        
        if len(valid_rates) > 0:
            mean_rate = np.mean(valid_rates)
            std_dev = np.std(valid_rates)
            sem = std_dev / np.sqrt(len(valid_rates)) if len(valid_rates) > 0 else 0
            ci_margin = 1.96 * sem if len(valid_rates) > 1 else 0
            current_method_mean_rates.append(mean_rate)
            current_method_ci_lower.append(max(0, mean_rate - ci_margin))
            current_method_ci_upper.append(min(100, mean_rate + ci_margin))
        else:
            current_method_mean_rates.append(np.nan)
            current_method_ci_lower.append(np.nan)
            current_method_ci_upper.append(np.nan)
            
    all_methods_mean_rates_vs_radius[method_name] = np.array(current_method_mean_rates)
    all_methods_ci_lower_vs_radius[method_name] = np.array(current_method_ci_lower)
    all_methods_ci_upper_vs_radius[method_name] = np.array(current_method_ci_upper)

# Plot success rate vs. radius for all methods
plot_success_rate_vs_radius(
    radii_to_test,
    all_methods_mean_rates_vs_radius,
    all_methods_ci_lower_vs_radius,
    all_methods_ci_upper_vs_radius,
    PREDICTION_SOURCES
)

print("\nScript finished.")
