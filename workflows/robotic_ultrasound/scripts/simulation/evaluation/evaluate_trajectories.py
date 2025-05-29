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

import argparse
import os
from dataclasses import dataclass

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


@dataclass
class PredictionSourceConfig:
    file_pattern: str
    label: str
    color: str


def _initialize_method_results_structure(radius_values):
    """Initializes the data structure for storing results for one method."""
    return {
        "original_success_rates": [],
        "original_avg_distances": [],
        "all_episodes_success_rates_per_radius": {r: [] for r in radius_values},
    }


def main(
    episode=None,
    data_root=None,
    radius_for_plots=0.01,
    radius_to_test=(0.001, 0.05, 20),
    saved_compare_name=None,
    prediction_sources=None,
):
    """Evaluate the trajectories of the prediction sources.

    This function plots the success rate versus radius for each prediction source
    and also plots the 3D trajectories for each source.

    Args:
        episode: The episode to evaluate. If None, all episodes will be evaluated.
        data_root: The root directory of the data.
        radius_for_plots: The radius for the plots (default is 0.01m).
        radius_to_test: The radius to test. It is a tuple of (start, end, num_points).
        saved_compare_name: The name of the saved comparison plot.
        prediction_sources: A dictionary where keys are method names and values are
                            PredictionSourceConfig objects.
    """
    radius_values = np.linspace(radius_to_test[0], radius_to_test[1], radius_to_test[2])

    results_per_method = {
        method: _initialize_method_results_structure(radius_values) for method in prediction_sources.keys()
    }

    if episode is None:
        episode = len([f for f in os.listdir(data_root) if f.endswith(".hdf5")])
    else:
        episode = int(episode)

    for e in range(episode):
        print(f"Processing Episode {e}...")
        gt_data_path = f"{data_root}/data_{e}.hdf5"
        gt_actions, gt_states = load_ground_truth_data(gt_data_path, e)

        if gt_actions is None or gt_states is None:
            print(f"Warning: Episode {e} has no valid ground truth data. Skipping.")
            for method_name_skip in prediction_sources.keys():
                results_per_method[method_name_skip]["original_success_rates"].append(np.nan)
                results_per_method[method_name_skip]["original_avg_distances"].append(np.nan)
                for r_val_skip in radius_values:
                    results_per_method[method_name_skip]["all_episodes_success_rates_per_radius"][r_val_skip].append(
                        np.nan
                    )
            continue

        scanning_gt = filter_scanning_points(gt_actions, gt_states)

        if scanning_gt is None or scanning_gt.shape[0] == 0:
            print(f"Warning: Episode {e} has no valid 'scanning' points after filtering. Skipping.")
            for method_name_skip in prediction_sources.keys():
                results_per_method[method_name_skip]["original_success_rates"].append(np.nan)
                results_per_method[method_name_skip]["original_avg_distances"].append(np.nan)
                for r_val_skip in radius_values:
                    results_per_method[method_name_skip]["all_episodes_success_rates_per_radius"][r_val_skip].append(
                        np.nan
                    )
            continue

        for method_name, details in prediction_sources.items():
            pred_file_path = f"{data_root}/{details.file_pattern.format(e=e)}"
            pred_traj = load_predicted_trajectory(pred_file_path)

            if pred_traj is None or pred_traj.shape[0] == 0:
                print(f"  {method_name} - Ep {e}: Predicted trajectory is empty or failed to load. Skipping.")
                success_rate_orig, avg_distance_orig = np.nan, np.nan
                for r_val_skip in radius_values:
                    results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val_skip].append(np.nan)
            else:
                success_rate_orig, avg_distance_orig = compute_trajectory_overlap_and_distance(
                    scanning_gt, pred_traj, radius=radius_for_plots
                )
                plot_3d_save_path = f"{data_root}/{method_name}/3d_trajectories-{e}.png"
                plot_3d_trajectories(
                    scanning_gt,
                    pred_traj,
                    save_path=plot_3d_save_path,
                    gt_label="Ground Truth",
                    pred_label=details.label,
                    pred_color=details.color,
                    title=f"Ep {e} - {method_name}: SR ({radius_for_plots}m): "
                    f"{success_rate_orig:.2f}%, AvgDist: {avg_distance_orig:.4f}m",
                )
                print(
                    f"  {method_name} - Ep {e}: SR ({radius_for_plots}m) = "
                    f"{success_rate_orig:.2f}%, AvgMinDist = {avg_distance_orig:.4f}m"
                )

                # Success rate vs. radius data collection
                if pred_traj.shape[0] > 0:
                    tree = cKDTree(pred_traj)
                    distances_to_pred, _ = tree.query(scanning_gt)
                    for r_val in radius_values:
                        rate_for_radius = 100.0 * np.sum(distances_to_pred < r_val) / len(scanning_gt)
                        results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val].append(
                            rate_for_radius
                        )
                else:
                    for r_val_skip in radius_values:
                        results_per_method[method_name]["all_episodes_success_rates_per_radius"][r_val_skip].append(
                            np.nan
                        )

            results_per_method[method_name]["original_success_rates"].append(success_rate_orig)
            results_per_method[method_name]["original_avg_distances"].append(avg_distance_orig)

    # --- Calculate and Print Summaries ---
    all_methods_mean_rates_vs_radius = {}
    all_methods_ci_lower_vs_radius = {}
    all_methods_ci_upper_vs_radius = {}

    print("\n--- Overall Summary (using radius=", radius_for_plots, "m) ---")
    for method_name in prediction_sources.keys():
        mean_orig_sr = np.nanmean(results_per_method[method_name]["original_success_rates"])
        mean_orig_ad = np.nanmean(results_per_method[method_name]["original_avg_distances"])
        print(f"Method: {method_name}")
        print(f"  Avg Success Rate: {mean_orig_sr:.2f}%")
        print(f"  Avg Min Distance: {mean_orig_ad:.4f}m")

        current_method_mean_rates = []
        current_method_ci_lower = []
        current_method_ci_upper = []
        for r_val in radius_values:
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
        radius_values,
        all_methods_mean_rates_vs_radius,
        all_methods_ci_lower_vs_radius,
        all_methods_ci_upper_vs_radius,
        prediction_sources,
        save_path=success_rate_plot_save_path,
    )

    print("\nScript finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectories from prediction sources.")
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Number of initial episodes. Default: all episodes in data_root.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/hdd/cosmos/heldout-test50",
        help="The root directory of the data.",
    )
    parser.add_argument(
        "--radius_for_plots",
        type=float,
        default=0.01,
        help="The radius for the 3D trajectory plots (default is 0.01m).",
    )
    parser.add_argument(
        "--radius_to_test",
        type=lambda s: tuple(map(float, s.split(","))),
        default=(0.001, 0.05, 20),
        help="The radius range to test for success rate plot. Format: '(start,end,num_points)'",
    )
    parser.add_argument(
        "--saved_compare_name",
        type=str,
        default="comparison_success_rate_vs_radius.png",
        help="The name of the saved comparison plot.",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        action="append",
        help="Name/key for a prediction source (e.g., WCOS). Specify once per source.",
        default=[],
    )
    parser.add_argument(
        "--ps-file-pattern",
        type=str,
        action="append",
        help="File pattern for a prediction source. Must match order of --method-name.",
        default=[],
    )
    parser.add_argument(
        "--ps-label",
        type=str,
        action="append",
        help="Label for a prediction source (for plots). Must match order of --method-name.",
        default=[],
    )
    parser.add_argument(
        "--ps-color",
        type=str,
        action="append",
        help="Color for a prediction source (for plots). Must match order of --method-name.",
        default=[],
    )

    args = parser.parse_args()

    prediction_sources_dict = {}
    if args.method_name:  # If any method_name is provided
        if not (len(args.method_name) == len(args.ps_file_pattern) == len(args.ps_label) == len(args.ps_color)):
            raise ValueError(
                "Error: method-name, ps-file-pattern, ps-label, and ps-color must have the same number of entries."
            )

        for i in range(len(args.method_name)):
            name = args.method_name[i]
            pattern = args.ps_file_pattern[i]
            label = args.ps_label[i]
            color = args.ps_color[i]
            if name in prediction_sources_dict:
                print(f"Error: Duplicate method name found: {name}. Method names must be unique.")
                exit(1)
            prediction_sources_dict[name] = PredictionSourceConfig(file_pattern=pattern, label=label, color=color)
    else:
        print("No prediction sources provided via command line. Using default sources.")
        prediction_sources_dict = {
            "WCOS": PredictionSourceConfig(
                file_pattern="800/pi0_robot_obs_{e}.npz", label="With COSMOS Prediction", color="red"
            ),
            "WOCOS": PredictionSourceConfig(
                file_pattern="400/pi0_robot_obs_{e}.npz", label="Without COSMOS Prediction", color="green"
            ),
        }

    episode_to_run = args.episode
    if args.episode is None and args.data_root:
        if not os.path.exists(args.data_root) or not os.path.isdir(args.data_root):
            raise FileNotFoundError(f"Error: data_root '{args.data_root}' not found or is not a directory.")
        try:
            hdf5_files = [f for f in os.listdir(args.data_root) if f.endswith(".hdf5")]
            if not hdf5_files:
                print(f"Error: No .hdf5 files found in data_root '{args.data_root}'.")
                exit(1)
            episode_to_run = len(hdf5_files)
            print(f"Found {episode_to_run} .hdf5 files in '{args.data_root}'.")
        except Exception as e:
            print(f"Error accessing data_root '{args.data_root}' to count episodes: {e}")
            exit(1)
    elif args.episode is not None:
        episode_to_run = int(args.episode)

    main(
        episode=episode_to_run,
        data_root=args.data_root,
        radius_for_plots=args.radius_for_plots,
        radius_to_test=args.radius_to_test,
        saved_compare_name=args.saved_compare_name,
        prediction_sources=prediction_sources_dict,
    )
