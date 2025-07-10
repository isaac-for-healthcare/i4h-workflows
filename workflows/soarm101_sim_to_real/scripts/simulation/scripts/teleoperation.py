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

"""
SO-ARM 101 Teleoperation Script with Table Environment

This script allows keyboard control of the SO-ARM 101 robotic arm on a table.
The robot is positioned on top of a Seattle Lab Table for realistic interaction.

Keyboard Controls:
- I/K: shoulder_pan (base rotation)
- J/L: shoulder_lift (shoulder elevation)
- U/O: elbow_flex (elbow flexion)
- Z/X: wrist_flex (wrist flexion)
- C/V: wrist_roll (wrist rotation)
- B/N: gripper (gripper open/close)
- ESC: Exit the application
- SPACE: Reset to default position

Usage:
    python teleoperation.py --num_envs 1
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Teleoperate SO-ARM 101 robot on a table using keyboard controls."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--step_size", type=float, default=0.02, help="Step size for joint movements.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
import carb
import omni.appwindow

from simulation.configs.soarm101_robot_cfg import SOARM101_TABLE_SCENE_CFG


class SoArm101TeleoperationController:
    """Teleoperation controller for SO-ARM 101 on table."""
    
    def __init__(self, scene: InteractiveScene, step_size: float = 0.02):
        """Initialize the teleoperation controller."""
        self.scene = scene
        self.step_size = step_size
        self.robot = scene["soarm101"]
        
        # Joint names in order
        self.joint_names = [
            "shoulder_pan",   # 0
            "shoulder_lift",  # 1
            "elbow_flex",     # 2
            "wrist_flex",     # 3
            "wrist_roll",     # 4
            "gripper"         # 5
        ]
        
        # Current joint positions
        self.current_joint_pos = self.robot.data.default_joint_pos.clone()
        
        # Joint limits (based on real robot calibration data with 5% safety margin)
        self.joint_limits = {
            "shoulder_pan": [-1.858, 1.918],    # Real: [-2.068, 2.128]
            "shoulder_lift": [-1.716, 1.556],   # Real: [-1.898, 1.738]
            "elbow_flex": [-1.738, 1.404],      # Real: [-1.913, 1.578]
            "wrist_flex": [-1.587, 1.662],      # Real: [-1.767, 1.842]
            "wrist_roll": [-2.725, 2.593],      # Real: [-3.020, 2.888]
            "gripper": [-0.071, 2.266]           # Real: [-0.071, 2.266]
        }
        
        # Key mapping for joint control
        self.key_mapping = {
            carb.input.KeyboardInput.I: ("shoulder_pan", 1),
            carb.input.KeyboardInput.K: ("shoulder_pan", -1),
            carb.input.KeyboardInput.J: ("shoulder_lift", 1),
            carb.input.KeyboardInput.L: ("shoulder_lift", -1),
            carb.input.KeyboardInput.U: ("elbow_flex", 1),
            carb.input.KeyboardInput.O: ("elbow_flex", -1),
            carb.input.KeyboardInput.Z: ("wrist_flex", 1),
            carb.input.KeyboardInput.X: ("wrist_flex", -1),
            carb.input.KeyboardInput.C: ("wrist_roll", 1),
            carb.input.KeyboardInput.V: ("wrist_roll", -1),
            carb.input.KeyboardInput.B: ("gripper", 1),
            carb.input.KeyboardInput.N: ("gripper", -1),
        }
        
        # Track pressed keys
        self.pressed_keys = set()
        
        # Keyboard interface will be setup later
        self.input_interface = None
        self.keyboard = None
        self.keyboard_sub = None
        
        print("Keyboard Controls:")
        print("  I/K: shoulder_pan (base rotation)")
        print("  J/L: shoulder_lift (shoulder elevation)")
        print("  U/O: elbow_flex (elbow flexion)")
        print("  Z/X: wrist_flex (wrist flexion)")
        print("  C/V: wrist_roll (wrist rotation)")
        print("  B/N: gripper (open/close)")
        print("  SPACE: Reset to default position")
        print("  ESC: Exit")
        print("Press and hold keys to move joints continuously")
        
    def setup_keyboard_interface(self):
        """Setup keyboard interface after simulation is fully initialized."""
        if self.input_interface is None:
            print("[INFO]: Setting up keyboard interface...")
            self.input_interface = carb.input.acquire_input_interface()
            self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self.keyboard_sub = self.input_interface.subscribe_to_keyboard_events(
                self.keyboard, self.on_keyboard_event
            )
            print("[INFO]: Keyboard interface ready!")
        
    def on_keyboard_event(self, event, *args):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.ESCAPE:
                simulation_app.close()
                return True
            elif event.input == carb.input.KeyboardInput.SPACE:
                self.reset_to_default()
                return True
            elif event.input in self.key_mapping:
                self.pressed_keys.add(event.input)
                return True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in self.pressed_keys:
                self.pressed_keys.remove(event.input)
                return True
        return False
        
    def reset_to_default(self):
        """Reset robot to default position."""
        self.current_joint_pos = self.robot.data.default_joint_pos.clone()
        self.robot.set_joint_position_target(self.current_joint_pos)
        print("[INFO]: Reset to default position")
        self.print_joint_status()
    
    def update_joint_positions(self):
        """Update joint positions based on pressed keys."""
        # Ensure keyboard interface is setup
        if self.input_interface is None:
            self.setup_keyboard_interface()
            
        # Process all currently pressed keys
        for key in self.pressed_keys:
            if key in self.key_mapping:
                joint_name, direction = self.key_mapping[key]
                joint_idx = self.joint_names.index(joint_name)
                
                # Calculate new position
                delta = direction * self.step_size
                new_pos = self.current_joint_pos[:, joint_idx] + delta
                
                # Apply joint limits
                min_limit, max_limit = self.joint_limits[joint_name]
                new_pos = torch.clamp(new_pos, min_limit, max_limit)
                
                # Update current position
                self.current_joint_pos[:, joint_idx] = new_pos
        
        # Apply new joint positions
        self.robot.set_joint_position_target(self.current_joint_pos)
    
    def print_joint_status(self):
        """Print current joint positions."""
        print("\nCurrent Joint Positions:")
        for i, joint_name in enumerate(self.joint_names):
            pos = self.current_joint_pos[0, i].item()
            min_limit, max_limit = self.joint_limits[joint_name]
            print(f"  {joint_name:15}: {pos:6.3f} rad (limits: [{min_limit:5.1f}, {max_limit:5.1f}])")
    
    def get_joint_status_string(self):
        """Get joint status as formatted string."""
        status = []
        for i, joint_name in enumerate(self.joint_names):
            pos = self.current_joint_pos[0, i].item()
            status.append(f"{joint_name}: {pos:6.3f}")
        return " | ".join(status)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view optimized for table environment
    # Position camera to show both robot and table clearly
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.2])
    
    # Create scene with table environment
    scene_cfg = SOARM101_TABLE_SCENE_CFG(args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Initialize teleoperation controller
    teleop_controller = SoArm101TeleoperationController(scene, step_size=args_cli.step_size)
    
    # Print initial status
    teleop_controller.print_joint_status()
    
    print("[INFO]: SO-ARM 101 table teleoperation ready!")
    print("[INFO]: Robot is positioned on the table surface")
    print("[INFO]: Press ESC to exit")
    
    # Main simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0
    
    while simulation_app.is_running():
        # Update joint positions based on keyboard input
        teleop_controller.update_joint_positions()
        
        # Write data to simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Print status every 60 frames (approximately 1 second at 60 FPS)
        if count % 60 == 0:
            status = teleop_controller.get_joint_status_string()
            print(f"\r[{count:6d}] {status}", end="", flush=True)
    
    # Clean up keyboard subscription
    if teleop_controller.keyboard_sub is not None:
        teleop_controller.keyboard_sub.unsubscribe()


if __name__ == "__main__":
    main()
    simulation_app.close() 