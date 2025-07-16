# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs import ManagerBasedEnv


# def task_done(
#     env: ManagerBasedRLEnv,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     right_wrist_max_x: float = 0.26,
#     min_x: float = 0.30,
#     max_x: float = 0.95,
#     min_y: float = 0.25,
#     max_y: float = 0.66,
#     min_height: float = 1.13,
#     min_vel: float = 0.20,
# ) -> torch.Tensor:
#     """Determine if the object placement task is complete.

#     This function checks whether all success conditions for the task have been met:
#     1. object is within the target x/y range
#     2. object is below a minimum height
#     3. object velocity is below threshold
#     4. Right robot wrist is retracted back towards body (past a given x pos threshold)

#     Args:
#         env: The RL environment instance.
#         object_cfg: Configuration for the object entity.
#         right_wrist_max_x: Maximum x position of the right wrist for task completion.
#         min_x: Minimum x position of the object for task completion.
#         max_x: Maximum x position of the object for task completion.
#         min_y: Minimum y position of the object for task completion.
#         max_y: Maximum y position of the object for task completion.
#         min_height: Minimum height (z position) of the object for task completion.
#         min_vel: Minimum velocity magnitude of the object for task completion.

#     Returns:
#         Boolean tensor indicating which environments have completed the task.
#     """
#     # Get object entity from the scene
#     object: RigidObject = env.scene[object_cfg.name]

#     # Extract wheel position relative to environment origin
#     wheel_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
#     wheel_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
#     wheel_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
#     wheel_vel = torch.abs(object.data.root_vel_w)

#     # Get right wrist position relative to environment origin
#     robot_body_pos_w = env.scene["robot"].data.body_pos_w
#     right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
#     right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

#     # Check all success conditions and combine with logical AND
#     done = wheel_x < max_x
#     done = torch.logical_and(done, wheel_x > min_x)
#     done = torch.logical_and(done, wheel_y < max_y)
#     done = torch.logical_and(done, wheel_y > min_y)
#     done = torch.logical_and(done, wheel_height < min_height)
#     done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
#     done = torch.logical_and(done, wheel_vel[:, 0] < min_vel)
#     done = torch.logical_and(done, wheel_vel[:, 1] < min_vel)
#     done = torch.logical_and(done, wheel_vel[:, 2] < min_vel)

#     return done



def object_moved(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.07,
) -> torch.Tensor:
    """Terminate if the object has moved by a certain distance from its initial position.

    This function caches the object's initial position at the start of the episode
    and checks if the object has moved beyond the specified distance threshold.
    It uses `env.extras` to store the initial position across steps for each environment.

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        distance_threshold: The distance threshold (in meters) to trigger termination.
                           Defaults to 0.05 (5 cm).

    Returns:
        Boolean tensor indicating which environments have met the termination condition.
    """
    # Get object entity from the scene
    obj: RigidObject = env.scene[object_cfg.name]

    # Define a unique key for storing the initial position in extras
    initial_pos_key = f"initial_pos_{object_cfg.name}"

    # At the very first step of the simulation, create the buffer in extras
    if initial_pos_key not in env.extras:
        env.extras[initial_pos_key] = obj.data.root_pos_w.clone()

    # For any environment that was reset, update its initial position.
    # The first step of an episode is indicated by episode_length_buf being 0.
    is_first_step = env.episode_length_buf == 0
    if torch.any(is_first_step):
        env.extras[initial_pos_key][is_first_step] = obj.data.root_pos_w[is_first_step].clone()

    # Retrieve initial and current positions
    initial_pos = env.extras[initial_pos_key]
    current_pos = obj.data.root_pos_w

    # Compute the Euclidean distance moved from the initial position
    distance_moved = torch.norm(current_pos[:, :3] - initial_pos[:, :3], p=2, dim=1)

    # Check if the distance moved exceeds the threshold
    return distance_moved > distance_threshold


def ultrasound_scanning_task_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("black_sorting_bin"),
    organ_cfg: SceneEntityCfg = SceneEntityCfg("organs"),
    max_object_to_bin_x: float = 0.08,
    max_object_to_bin_y: float = 0.08,
    max_object_to_bin_z: float = 0.15,
    min_scan_distance: float = 0.08,
    scan_height_tolerance: float = 0.10,
    min_pickup_distance: float = 0.08,  # 物体必须移动至少这个距离才算被拿起
) -> torch.Tensor:
    """
    Success detection for ultrasound scanning task.
    
    Success criteria (all must be met simultaneously):
    1. Object (ultrasound probe) is back in the black_sorting_bin (within tolerance)
    2. The probe has completed sufficient scanning on the organ surface
    3. The probe must have been picked up and moved significantly from its initial position
    
    This approach is robust to bin movement during simulation by:
    - Tracking object's initial position relative to bin
    - Monitoring if object has moved significantly from its starting location
    - Using current relative position for final placement check
    
    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the ultrasound probe object.
        bin_cfg: Configuration for the black sorting bin.
        organ_cfg: Configuration for the organ/phantom.
        max_object_to_bin_x: Maximum x distance between object and bin for success.
        max_object_to_bin_y: Maximum y distance between object and bin for success.
        max_object_to_bin_z: Maximum z distance between object and bin for success.
        min_scan_distance: Minimum scanning distance required on organ surface.
        scan_height_tolerance: Height tolerance above organ surface for valid scanning.
        min_pickup_distance: Minimum distance object must travel from initial position to be considered "picked up".
    
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    object_asset: RigidObject = env.scene[object_cfg.name]
    bin_asset: RigidObject = env.scene[bin_cfg.name]
    organ_asset: RigidObject = env.scene[organ_cfg.name]
    
    # Get positions relative to environment origin
    object_pos = object_asset.data.root_pos_w - env.scene.env_origins
    bin_pos = bin_asset.data.root_pos_w - env.scene.env_origins
    organ_pos = organ_asset.data.root_pos_w - env.scene.env_origins
    
    # Initialize tracking variables if not exists
    if not hasattr(env, '_scan_distance_total'):
        env._scan_distance_total = torch.zeros(env.num_envs, device=env.device)
        env._last_scan_pos = torch.zeros((env.num_envs, 3), device=env.device)
        env._scan_distance_total[:] = 0.0
        
    # Initialize object initial position tracking (relative to environment, not bin)
    if not hasattr(env, '_object_initial_pos'):
        env._object_initial_pos = object_pos.clone()
        env._object_has_been_picked_up = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._max_distance_from_initial = torch.zeros(env.num_envs, device=env.device)
    
    # Track maximum distance object has moved from its initial position
    current_distance_from_initial = torch.norm(object_pos - env._object_initial_pos, dim=1)
    env._max_distance_from_initial = torch.maximum(env._max_distance_from_initial, current_distance_from_initial)
    
    # Update "has been picked up" status - once object moves far enough from initial position
    env._object_has_been_picked_up = torch.logical_or(
        env._object_has_been_picked_up, 
        env._max_distance_from_initial > min_pickup_distance
    )
    
    # Update scanning distance when near organ
    object_to_organ_distance = torch.norm(object_pos - organ_pos, dim=1)
    # Consider scanning when object is close to organ surface (within scan_height_tolerance)
    is_scanning = object_to_organ_distance < scan_height_tolerance
    # print(f"is_scanning: {is_scanning}", f"object_to_organ_distance: {object_to_organ_distance}", f"scan_height_tolerance: {scan_height_tolerance}")
    
    # Calculate movement distance for environments that are currently scanning
    for env_idx in range(env.num_envs):
        if is_scanning[env_idx]:
            if env._scan_distance_total[env_idx] == 0.0:
                # First time near organ, initialize last position
                env._last_scan_pos[env_idx] = object_pos[env_idx].clone()
                # Set a small initial value to indicate scanning has started
                env._scan_distance_total[env_idx] = 0.001
            else:
                # Calculate distance moved since last scan position
                movement = torch.norm(object_pos[env_idx] - env._last_scan_pos[env_idx])
                env._scan_distance_total[env_idx] += movement
                env._last_scan_pos[env_idx] = object_pos[env_idx].clone()
    
    # Check if object is back in bin (current relative positions)
    object_to_bin_x = torch.abs(object_pos[:, 0] - bin_pos[:, 0])
    object_to_bin_y = torch.abs(object_pos[:, 1] - bin_pos[:, 1])
    object_to_bin_z = torch.abs(object_pos[:, 2] - bin_pos[:, 2])
    
    # Check all success conditions
    object_in_bin = object_to_bin_x < max_object_to_bin_x
    object_in_bin = torch.logical_and(object_in_bin, object_to_bin_y < max_object_to_bin_y)
    object_in_bin = torch.logical_and(object_in_bin, object_to_bin_z < max_object_to_bin_z)
    
    # Check if sufficient scanning has been completed
    sufficient_scanning = env._scan_distance_total >= min_scan_distance
    
    # Task is successful when ALL conditions are met:
    # 1. Object is currently in bin (relative to current bin position)
    # 2. Object has been picked up (moved significantly from initial position)
    # 3. Sufficient scanning has been completed
    success = torch.logical_and(object_in_bin, env._object_has_been_picked_up)
    success = torch.logical_and(success, sufficient_scanning)
    
    # # Debug information
    # print(f"object_in_bin: {object_in_bin}")
    # print(f"object_has_been_picked_up: {env._object_has_been_picked_up}")
    # print(f"max_distance_from_initial: {env._max_distance_from_initial}")
    # print(f"sufficient_scanning: {sufficient_scanning}")
    # print(f"scan_distance_total: {env._scan_distance_total}")
    # print(f"min_scan_distance: {min_scan_distance}")
    # print(f"current_object_pos: {object_pos[0] if env.num_envs > 0 else 'N/A'}")
    # print(f"success: {success}")
    
    return success

def reset_task_state(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the task state for ultrasound scanning task."""
    if hasattr(env, '_scan_distance_total'):
        env._scan_distance_total[env_ids] = 0.0
        env._last_scan_pos[env_ids] = 0.0
    if hasattr(env, '_object_has_been_picked_up'):
        env._object_has_been_picked_up[env_ids] = False
    if hasattr(env, '_max_distance_from_initial'):
        env._max_distance_from_initial[env_ids] = 0.0
    # Note: _object_initial_pos will be reset automatically when the function runs after reset

def reset_specific_object_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
