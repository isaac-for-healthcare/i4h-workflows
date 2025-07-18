"""
SO-ARM 101 Gymnasium Environment Wrapper for Isaac Lab
"""

import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
GYM_AVAILABLE = True

# Only import AppLauncher at module level - other Isaac Lab imports happen later
from isaaclab.app import AppLauncher

# These will be imported after Isaac Sim is initialized
sim_utils = None
InteractiveScene = None
SOARM101_TABLE_SCENE_CFG = None


class SoArm101Env(gym.Env):
    """
    SO-ARM 101 Gymnasium Environment for Isaac Lab
    
    This environment wraps the Isaac Lab SO-ARM 101 robot simulation
    into a standard Gymnasium interface for RL training.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, num_envs: int = 1, device: str = "cuda", render_mode: Optional[str] = None, 
                 sim_steps_per_action: int = 10, action_scale: float = 1.0):
        super().__init__()
        
        self.num_envs = num_envs
        self.device = device
        self.render_mode = render_mode
        
        # Simulation parameters for better movement visibility
        self.sim_steps_per_action = sim_steps_per_action  # Multiple physics steps per action
        self.action_scale = action_scale  # Scale actions to make movements larger
        
        # Initialize Isaac Lab (this will import modules after launching Isaac Sim)
        self._init_isaac_lab()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Internal state
        self.current_step = 0
        self.max_episode_length = 1000
        
    def _init_isaac_lab(self):
        """Initialize Isaac Lab simulation."""
        global sim_utils, InteractiveScene, SOARM101_TABLE_SCENE_CFG
        
        # Initialize Isaac Sim FIRST (following teleoperation.py pattern)
        import argparse
        parser = argparse.ArgumentParser(description="SO-ARM 101 Gymnasium Environment")
        
        # Let AppLauncher add its own arguments (including device, headless, etc.)
        AppLauncher.add_app_launcher_args(parser)
        
        # Parse with empty args list, then manually set our values
        args = parser.parse_args([])
        args.device = self.device
        args.headless = (self.render_mode != "human")
        
        # Launch Isaac Sim
        app_launcher = AppLauncher(args)
        self.simulation_app = app_launcher.app
        
        # NOW import Isaac Lab modules (after Isaac Sim is running)
        import isaaclab.sim as sim_utils_module
        from isaaclab.scene import InteractiveScene as InteractiveSceneClass
        from simulation.configs.soarm101_robot_cfg import SOARM101_TABLE_SCENE_CFG as SceneCfg
        
        # Assign to global variables so other methods can use them
        sim_utils = sim_utils_module
        InteractiveScene = InteractiveSceneClass
        SOARM101_TABLE_SCENE_CFG = SceneCfg
        
        # Create simulation context
        sim_cfg = sim_utils.SimulationCfg(device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        # Set camera view
        self.sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.2])
        
        # Create scene
        scene_cfg = SOARM101_TABLE_SCENE_CFG(self.num_envs, env_spacing=4.0)
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene["soarm101"]
        
        # Play the simulator
        self.sim.reset()
        
    def _setup_spaces(self):
        """Setup Gymnasium action and observation spaces."""
        # Action space: 6 joint positions (continuous)
        self.action_space = spaces.Box(
            low=np.array([-1.92, -1.75, -1.69, -1.66, -2.74, -0.17]),  # Joint limits
            high=np.array([1.92, 1.75, 1.69, 1.66, 2.84, 1.75]),
            dtype=np.float32,
        )
        
        # Observation space: joint positions + joint velocities + end-effector pose
        obs_dim = 6 + 6 + 7  # positions + velocities + (xyz + quaternion)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Reset robot to initial position
        joint_positions = torch.zeros((self.num_envs, 6), device=self.device)
        self.robot.set_joint_position_target(joint_positions)
        
        # Step simulation to settle
        for _ in range(10):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
            
        self.current_step = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Convert action to tensor and scale for more visible movements
        action_tensor = torch.from_numpy(action).float().to(self.device) * self.action_scale
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0).repeat(self.num_envs, 1)
            
        self.robot.set_joint_position_target(action_tensor)
        
        # Step simulation multiple times for smoother, more visible movement
        for _ in range(self.sim_steps_per_action):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
        
        self.current_step += 1
        
        # Get observation, reward, and check termination
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_length
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Joint positions and velocities
        joint_pos = self.robot.data.joint_pos[:, :6]  # First 6 joints
        joint_vel = self.robot.data.joint_vel[:, :6]
        
        # End-effector pose (simplified - you might want to compute actual FK)
        # For now, just use the last joint position as a proxy
        ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        ee_quat = torch.tensor([0, 0, 0, 1], device=self.device).repeat(self.num_envs, 1)
        
        # Concatenate all observations
        obs = torch.cat([joint_pos, joint_vel, ee_pos, ee_quat], dim=1)
        
        # Return observation for the first environment (if single env)
        if self.num_envs == 1:
            return obs[0].cpu().numpy()
        else:
            return obs.cpu().numpy()
            
    def _compute_reward(self) -> float:
        """Compute reward (not needed for non-RL use cases)."""
        # Since you're not doing RL training, just return 0
        # The Gymnasium interface requires a reward, but it can be ignored
        return 0.0
        
    def _check_terminated(self) -> bool:
        """Check if episode is terminated."""
        # Simple termination: no termination for base environment
        return False
        
    def _get_info(self) -> Dict:
        """Get additional info."""
        return {
            "step": self.current_step,
            "joint_positions": self.robot.data.joint_pos[:, :6].cpu().numpy(),
            "joint_velocities": self.robot.data.joint_vel[:, :6].cpu().numpy(),
        }
        
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Isaac Lab handles rendering automatically when not headless
            pass
        elif self.render_mode == "rgb_array":
            # You would implement camera capture here
            # For now, return a placeholder
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
    def close(self):
        """Close the environment."""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.close()


class SoArm101ReachEnv(SoArm101Env):
    """
    SO-ARM 101 Reach Task Environment
    
    The robot must reach a target position with its end-effector.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_position = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset with new target position."""
        obs, info = super().reset(seed, options)
        
        # Set random target position
        self.target_position = np.random.uniform(
            low=[0.2, -0.3, 0.1],
            high=[0.6, 0.3, 0.5]
        )
        
        info["target_position"] = self.target_position
        return obs, info
        
    def _compute_reward(self) -> float:
        """Compute reach reward based on distance to target."""
        if self.target_position is None:
            return 0.0
            
        # Get end-effector position (simplified - you'd compute actual FK)
        # For demo, use a proxy based on joint positions
        joint_pos = self.robot.data.joint_pos[0, :6].cpu().numpy()
        
        # Simple FK approximation (replace with actual kinematics)
        ee_pos_approx = np.array([
            0.4 + 0.2 * np.cos(joint_pos[0]),
            0.2 * np.sin(joint_pos[0]), 
            0.3 + 0.1 * joint_pos[1]
        ])
        
        # Distance-based reward
        distance = np.linalg.norm(ee_pos_approx - self.target_position)
        reward = -distance
        
        # Bonus for being very close
        if distance < 0.05:
            reward += 10.0
            
        return float(reward)
        
    def _check_terminated(self) -> bool:
        """Terminate if target is reached."""
        if self.target_position is None:
            return False
            
        # Check if end-effector is close to target
        joint_pos = self.robot.data.joint_pos[0, :6].cpu().numpy()
        ee_pos_approx = np.array([
            0.4 + 0.2 * np.cos(joint_pos[0]),
            0.2 * np.sin(joint_pos[0]),
            0.3 + 0.1 * joint_pos[1]
        ])
        
        distance = np.linalg.norm(ee_pos_approx - self.target_position)
        return distance < 0.02  # 2cm tolerance


class SoArm101VecEnv(SoArm101Env):
    """
    Vectorized version of SO-ARM 101 environment for faster training.
    """
    
    def __init__(self, num_envs: int = 16, **kwargs):
        kwargs["num_envs"] = num_envs
        super().__init__(**kwargs)
        
    def step(self, actions: np.ndarray):
        """Step with vectorized actions."""
        # actions should be shape (num_envs, action_dim)
        action_tensor = torch.from_numpy(actions).float().to(self.device) * self.action_scale
        
        self.robot.set_joint_position_target(action_tensor)
        
        # Step simulation multiple times for smoother movement
        for _ in range(self.sim_steps_per_action):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
        
        self.current_step += 1
        
        # Get vectorized observations and rewards
        obs = self._get_observation()  # Shape: (num_envs, obs_dim)
        rewards = self._compute_vectorized_reward()  # Shape: (num_envs,)
        terminated = np.array([self._check_terminated()] * self.num_envs)
        truncated = np.array([self.current_step >= self.max_episode_length] * self.num_envs)
        infos = [self._get_info() for _ in range(self.num_envs)]
        
        return obs, rewards, terminated, truncated, infos
        
    def _compute_vectorized_reward(self) -> np.ndarray:
        """Compute vectorized rewards (not needed for non-RL use cases)."""
        # Return zeros for all environments since you're not doing RL
        return np.zeros(self.num_envs) 