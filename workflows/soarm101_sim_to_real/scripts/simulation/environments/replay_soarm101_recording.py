#!/usr/bin/env python3

"""
Replay HDF5 recordings for SoArm101 robot using Isaac Lab's dataset handling.
This script replays demonstrations recorded with teleoperation_record.py.
Based on Isaac Lab's replay_demos.py with SoArm101-specific enhancements.
"""

import argparse
import contextlib
import gymnasium as gym
import os
import time
import torch
from isaaclab.app import AppLauncher

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import task

# Add argparse arguments
parser = argparse.ArgumentParser(description="Replay SoArm101 recorded demonstrations")
parser.add_argument("dataset_file", type=str, default="", help="Path to the recorded HDF5 dataset file")
parser.add_argument("--task", type=str, default="SO101-Scrub-Nurse-v0", help="Name of the task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = original speed, 0.5 = half speed, 2.0 = double speed)")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)

parser.add_argument(
    "--action_key", 
    type=str, 
    default=None,
    help="Action key to use from dataset (default: 'actions')"
)


# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def get_action_from_episode(episode_data, action_key="actions", step_index=None):
    """Get action from episode data using specified action key.
    
    Args:
        episode_data: The episode data object
        action_key: Key to use for actions ("actions", "abs_actions", etc.)
        step_index: If provided, get action at specific step
    
    Returns:
        Action tensor or None if not available
    """
    try:
        if hasattr(episode_data, 'data') and action_key in episode_data.data:
            actions = episode_data.data[action_key]
            if step_index is not None:
                if step_index < len(actions):
                    return actions[step_index]
                else:
                    return None
            else:
                # Get next action using episode's internal index
                if hasattr(episode_data, 'next_action_index'):
                    if episode_data.next_action_index < len(actions):
                        action = actions[episode_data.next_action_index]
                        episode_data.next_action_index += 1
                        return action
                    else:
                        return None
                else:
                    # Initialize index if not present
                    episode_data.next_action_index = 0
                    if len(actions) > 0:
                        action = actions[0]
                        episode_data.next_action_index = 1
                        return action
                    else:
                        return None
        else:
            # Fallback to default method
            return episode_data.get_next_action()
    except Exception as e:
        print(f"Error getting action from episode: {e}")
        return episode_data.get_next_action()


def main():
    """Main function for replaying SoArm101 demonstrations."""
    global is_paused

    # Load dataset using Isaac Lab's dataset handler
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    
    print(f"Loading dataset: {args_cli.dataset_file}")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    print(f"Dataset has {episode_count} episodes")

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    # Determine action key to use
    action_key_to_use = args_cli.action_key if args_cli.action_key else "actions"

    # Get dataset FPS for proper timing control
    try:
        dataset_fps = dataset_file_handler.get_fps()
        target_frame_time = 1.0 / dataset_fps / args_cli.speed
    except:
        dataset_fps = 30.0  # Default FPS
        target_frame_time = 1.0 / dataset_fps / args_cli.speed

    # Determine episodes to replay
    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    # Use task from args or dataset
    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    # Force single environment for replay
    num_envs = 1

    # Parse environment configuration
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)
    env_cfg.use_teleop_device("so101leader")
    
    # Disable recorders and terminations for replay
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    # Reset environment once
    env.reset()

    # Main replay loop
    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    total_steps = 0    
    rate_limiter = RateLimiter(60)

    try:
        print(f"Starting replay of {len(episode_indices_to_replay)} episodes...")
        
        with torch.inference_mode():
            for episode_idx in episode_indices_to_replay:
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    break
                    
                replayed_episode_count += 1
                print(f"Episode {replayed_episode_count}/{len(episode_indices_to_replay)}: #{episode_idx}")
                
                # Load episode data
                episode_data = dataset_file_handler.load_episode(episode_names[episode_idx], env.device)
                
                # Reset environment to initial state
                env.reset()
                initial_state = episode_data.get_initial_state()
                print(f"initial_state: {initial_state}")
                if initial_state:
                    env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
                
                # Initialize episode data for custom action loading
                episode_data.next_action_index = 0
                episode_step = 0
                
                # Replay all actions in this episode
                while True:
                    # Get next action
                    action = get_action_from_episode(episode_data, action_key_to_use)
                    if action is None:
                        print(f"Episode {episode_idx} completed ({episode_step} steps)")
                        break
                    
                    # Ensure action has correct shape [1, 6] for single environment
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                          
                    obs, rewards, terminated, truncated, info = env.step(action)
                    rate_limiter.sleep(env)
                    
                    
                    episode_step += 1
                    total_steps += 1
                    # Check if episode terminated naturally
                    if terminated or truncated:
                        print(f"Episode {episode_idx} terminated at step {episode_step}")
                        break
    
    except KeyboardInterrupt:
        print("Replay interrupted by user")
    except Exception as e:
        print(f"Error during replay: {e}")
        import traceback
        traceback.print_exc()
    
    # Close environment
    env.close()
    print("Replay complete")

if __name__ == "__main__":
    main()
    simulation_app.close() 