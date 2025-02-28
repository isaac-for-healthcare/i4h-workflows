from enum import Enum
import torch

from typing import Dict
from dataclasses import dataclass
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.torch.rotations import quat_from_euler_xyz
from omni.isaac.lab.utils.math import compute_pose_error, axis_angle_from_quat
from simulation.policies.state_machine.modules.base_module import BaseControlModule
from simulation.policies.state_machine.utils import compute_relative_action, RobotPositions, RobotQuaternions, SMState

class UltrasoundStateMachine:
    """State machine for ultrasound procedure control."""
    
    def __init__(self, modules: Dict[str, BaseControlModule], device: str = "cuda:0"):
        """Initialize the state machine.
        
        Args:
            modules (Dict[str, BaseControlModule]): Dictionary of control modules
            device (str): Device to run computations on
        """
        self.modules = modules
        self.device = device
        self.organ_offset = torch.tensor(RobotPositions.ORGAN_OFFSET, device=device)
        self.target_offset = torch.tensor(RobotPositions.TARGET_OFFSET, device=device)
        self.down_quaternion = torch.tensor(RobotQuaternions.DOWN, device=device)
        self.state = SMState()
        self.object_view = RigidPrimView(
            prim_paths_expr="/World/envs/env.*/Robot/panda_hand", name="robot",
            track_contact_forces=True,
            prepare_contact_sensors=True,
            contact_filter_prim_paths_expr=["/World/envs/env.*/organs"],
            max_contact_count=50
        )
        self.object_view.initialize()
        self.assert_module_order()
        self.set_using_policy()
    
    def compute_action(self, env, robot_obs: torch.Tensor) -> torch.Tensor:
        """Compute combined action from all active modules.
        
        Args:
            env: The environment instance
            robot_obs: Current robot observations
            
        Returns:
            Combined action tensor
        """
        self.state.robot_obs = robot_obs
        self.state.contact_normal_force = self.get_normal_force()
        # Only use first env observation
        module_actions = {}
        
        for name, module in self.modules.items():
            action, state = module.compute_action(env, self.state)
            module_actions[name] = action
            self.state = state

        abs_commands = sum(module_actions.values())

        rel_commands = compute_relative_action(abs_commands, robot_obs.unsqueeze(0))

        # The policy already outputs relative actions, so we need to convert to absolute
        if self.using_policy:
            rel_commands[:, :3] = abs_commands[:, :3]

        return rel_commands, abs_commands
    
    def get_normal_force(self):
        """
        Get the average normal force from the contact sensor.
        """
        contact_forces = self.object_view.get_contact_force_data()[2]  # Shape: [num_contacts, 3]
        # Create mask for non-zero vectors (where all components are zero)
        non_zero_mask = torch.all(contact_forces == 0, dim=1).logical_not()
        # Sum forces only for non-zero vectors
        force_sum = contact_forces[non_zero_mask].sum(dim=0)
        # Calculate mean by dividing by number of non-zero vectors
        num_non_zero = non_zero_mask.sum()
        mean_force = force_sum / num_non_zero if num_non_zero > 0 else torch.zeros(3, device=contact_forces.device)
        return mean_force
    
    def reset(self) -> None:
        """Reset the state machine and all modules."""
        self.state.reset()
        for module in self.modules.values():
            module.reset()

    def assert_module_order(self):
        """Assert that the modules are in the correct order."""
        assert list(self.modules.keys()) == ["force", "orientation", "path_planning"], "Modules must be in the correct order"

    def set_using_policy(self):
        """Set the using_policy flag for both the state machine and force module."""
        assert "force" in self.modules, "Force module must be present"
        self.using_policy = self.modules["force"].using_policy
