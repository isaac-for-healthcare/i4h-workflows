import torch
from modules.base_module import BaseControlModule
from omni.isaac.lab.utils.math import quat_apply_yaw
from utils import PhantomScanPositions, SMState, UltrasoundState


class PathPlanningModule(BaseControlModule):
    """Module for path planning control."""

    def __init__(self, device: str = "cuda:0", use_quaternion: bool = False):
        self.start_pos = torch.tensor(PhantomScanPositions.SCAN_START, device=device).unsqueeze(0)
        self.contact_pos = torch.tensor(PhantomScanPositions.SCAN_CONTACT, device=device).unsqueeze(0)
        self.end_pos = torch.tensor(PhantomScanPositions.SCAN_END, device=device).unsqueeze(0)

        self.rel_target_offset = self.end_pos - self.contact_pos
        self.scan_steps = 100  # Adjusts the speed of the scan
        self.hold_steps = 50
        self.increment = self.rel_target_offset / self.scan_steps
        self.current_step = 0
        super().__init__(device, use_quaternion)

    def compute_action(self, env, sm_state: SMState):
        """Compute path planning action.

        Args:
            env: Environment instance
            sm_state: Current state machine state

        Returns:
            Tuple of computed action and updated state
        """
        action = self.get_base_action(env)
        object_data = env.unwrapped.scene["organs"].data
        object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
        object_orientation = object_data.root_quat_w

        if str(sm_state.state) == str(UltrasoundState.SETUP):
            # Note: quat_apply_yaw is used to map positions into the Torso's frame
            action[:, :3] = object_position + quat_apply_yaw(object_orientation, self.start_pos)
            return action, sm_state
        elif str(sm_state.state) == str(UltrasoundState.APPROACH):
            # Switch to the contact position
            action[:, :3] = object_position + quat_apply_yaw(object_orientation, self.contact_pos)

            # Check for state changes
            current_position = sm_state.robot_obs[0, :3]
            if torch.all(torch.abs(current_position - action[0, :3]) < 1e-3):
                sm_state.state = UltrasoundState.CONTACT

            return action, sm_state
        elif str(sm_state.state) == str(UltrasoundState.SCANNING):
            object_data = env.unwrapped.scene["organs"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            object_orientation = object_data.root_quat_w

            # Transform organ offset and increment to world frame using object orientation
            world_organ_start_pos = quat_apply_yaw(object_orientation, self.contact_pos)
            world_increment = quat_apply_yaw(object_orientation, self.increment)

            # Apply transformations
            target_position = object_position + world_organ_start_pos + world_increment
            action[:, :3] = target_position

            # Slowly move the robot towards the target
            if self.current_step < self.scan_steps:
                self.increment += self.rel_target_offset / self.scan_steps
                self.current_step += 1
            # Hold the robot in the same position for hold_steps
            elif self.current_step < (self.scan_steps + self.hold_steps):
                self.current_step += 1
            # Terminate the scan
            else:
                sm_state.state = UltrasoundState.DONE

            return action, sm_state
        # Else keep the robot in the same position
        else:
            return action, sm_state

    def reset(self):
        self.current_step = 0
        self.increment = self.rel_target_offset / self.scan_steps
        pass
