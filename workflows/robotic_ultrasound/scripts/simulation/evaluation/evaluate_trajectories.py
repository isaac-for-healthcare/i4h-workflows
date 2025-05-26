# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
from scipy.spatial import cKDTree
from simulation.evaluation.metrics import compute_trajectory_overlap_and_distance
from simulation.evaluation.utils import (
    filter_scanning_points,
    load_ground_truth_data,
    load_predicted_trajectory,
    plot_3d_trajectories,
    plot_success_rate_vs_radius,
)


def main(episode=None, data_root=None, DEFAULT_RADIUS_FOR_PLOTS=0.01, saved_compare_name=None, PREDICTION_SOURCES=None):
    radius_to_test = np.linspace(0.001, 0.05, 20)

    results_per_method = {
        method: {
            "original_success_rates": [],
            "original_avg_distances": [],
            "all_episodes_success_rates_per_radius": {r: [] for r in radius_to_test},
        }
        for method in PREDICTION_SOURCES.keys()
    }

    if episode is None:
        # check how many episodes are in the data root
        episode = len([f for f in os.listdir(data_root) if f.endswith(".hdf5")])
    else:
        episode = int(episode)

    for e in range(episode):
        print(f"Processing Episode {e}...")
        gt_data_path = f"{data_root}/data_{e}.hdf5"
        gt_actions, gt_states = load_ground_truth_data(gt_data_path, e)

        if gt_actions is None or gt_states is None:
            print(f"Warning: Episode {e} has no valid ground truth data. Skipping.")
            for method_name_skip in PREDICTION_SOURCES.keys():
                results_per_method[method_name_skip]["original_success_rates"].append(np.nan)
                results_per_method[method_name_skip]["original_avg_distances"].append(np.nan)
                for r_val_skip in radius_to_test:
                    results_per_method[method_name_skip]["all_episodes_success_rates_per_radius"][r_val_skip].append(
                        np.nan
                    )
            continue

        scanning_gt = filter_scanning_points(gt_actions, gt_states)

        if scanning_gt is None or scanning_gt.shape[0] == 0:
            print(f"Warning: Episode {e} has no valid 'scanning' points after filtering. Skipping.")
            for method_name_skip in PREDICTION_SOURCES.keys():
                results_per_method[method_name_skip]["original_success_rates"].append(np.nan)
                results_per_method[method_name_skip]["original_avg_distances"].append(np.nan)
                for r_val_skip in radius_to_test:
                    results_per_method[method_name_skip]["all_episodes_success_rates_per_radius"][r_val_skip].append(
                        np.nan
                    )
            continue

        for method_name, details in PREDICTION_SOURCES.items():
            pred_file_path = f"{data_root}/{details['file_pattern'].format(e=e)}"
            pred_traj = load_predicted_trajectory(pred_file_path)

            if pred_traj is None or pred_traj.shape[0] == 0:
                print(f"  {method_name} - Ep {e}: Predicted trajectory is empty or failed to load. Skipping.")
                success_rate_orig, avg_distance_orig = np.nan, np.nan
                for r_val_skip in radius_to_test:
                    results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val_skip].append(np.nan)
            else:
                success_rate_orig, avg_distance_orig = compute_trajectory_overlap_and_distance(
                    scanning_gt, pred_traj, radius=DEFAULT_RADIUS_FOR_PLOTS
                )
                plot_3d_save_path = f"{data_root}/{method_name}/3d_trajectories-{e}.png"
                plot_3d_trajectories(
                    scanning_gt,
                    pred_traj,
                    save_path=plot_3d_save_path,
                    gt_label="Ground Truth",
                    pred_label=details["label"],
                    pred_color=details["color"],
                    title=f"Ep {e} - {method_name}: SR ({DEFAULT_RADIUS_FOR_PLOTS}m): "
                    f"{success_rate_orig:.2f}%, AvgDist: {avg_distance_orig:.4f}m",
                )
                print(
                    f"  {method_name} - Ep {e}: SR ({DEFAULT_RADIUS_FOR_PLOTS}m) = "
                    f"{success_rate_orig:.2f}%, AvgMinDist = {avg_distance_orig:.4f}m"
                )

                # Success rate vs. radius data collection
                if pred_traj.shape[0] > 0:
                    tree = cKDTree(pred_traj)
                    distances_to_pred, _ = tree.query(scanning_gt)
                    for r_val in radius_to_test:
                        rate_for_radius = 100.0 * np.sum(distances_to_pred < r_val) / len(scanning_gt)
                        results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val].append(
                            rate_for_radius
                        )
                else:
                    for r_val_skip in radius_to_test:
                        results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val_skip].append(
                            np.nan
                        )

            results_per_method[method_name]["original_success_rates"].append(success_rate_orig)
            results_per_method[method_name]["original_avg_distances"].append(avg_distance_orig)

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

        current_method_mean_rates = []
        current_method_ci_lower = []
        current_method_ci_upper = []
        for r_val in radius_to_test:
            rates_at_this_radius = np.array(
                results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val]
            )
            valid_rates = rates_at_this_radius[~np.isnan(rates_at_this_radius)]

            if len(valid_rates) > 0:
                mean_rate = np.mean(valid_rates)
                std_dev = np.std(valid_rates)
                sem = std_dev / np.sqrt(len(valid_rates)) if len(valid_rates) > 0 else 0.0
                ci_margin = 1.96 * sem
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

    success_rate_plot_save_path = f"{data_root}/{saved_compare_name}"
    plot_success_rate_vs_radius(
        radius_to_test,
        all_methods_mean_rates_vs_radius,
        all_methods_ci_lower_vs_radius,
        all_methods_ci_upper_vs_radius,
        PREDICTION_SOURCES,
        save_path=success_rate_plot_save_path,
    )

    print("\nScript finished.")


if __name__ == "__main__":
    episode = 50
    data_root = "/mnt/hdd/cosmos/heldout-test50"
    DEFAULT_RADIUS_FOR_PLOTS = 0.01  # Radius used for individual 3D trajectory plots
    saved_compare_name = "comparison_success_rate_vs_radius.png"

    PREDICTION_SOURCES = {
        "WCOS": {"file_pattern": "800/pi0-800_robot_obs_{e}.npz", "label": "With COSMOS Prediction", "color": "red"},
        "WOCOS": {
            "file_pattern": "400/pi0-400_robot_obs_{e}.npz",
            "label": "Without COSMOS Prediction",
            "color": "green",
        },
    }
    main(
        episode=episode,
        data_root=data_root,
        DEFAULT_RADIUS_FOR_PLOTS=DEFAULT_RADIUS_FOR_PLOTS,
        saved_compare_name=saved_compare_name,
        PREDICTION_SOURCES=PREDICTION_SOURCES,
    )
