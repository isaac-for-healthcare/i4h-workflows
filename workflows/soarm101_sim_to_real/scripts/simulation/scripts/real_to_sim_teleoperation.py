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
import time
import math
import threading
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="SO-ARM 101 real-to-sim teleoperation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--host", type=str, default="localhost", help="TCP host")
parser.add_argument("--port", type=int, default=8888, help="TCP port")
parser.add_argument("--update_rate", type=float, default=200.0, help="Update rate Hz")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports - AFTER app launch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from simulation.configs.soarm101_robot_cfg import SOARM101_TABLE_SCENE_CFG


class SimplifiedTeleoperation:
    """Real-to-sim teleoperation with direct hardware integration."""
    
    # Joint configuration (centralized)
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    JOINT_MAPPING = {
        "shoulder_pan": "shoulder_pan",
        "shoulder_lift": "shoulder_lift",
        "elbow": "elbow_flex",
        "wrist_1": "wrist_flex", 
        "wrist_2": "wrist_roll",
        "gripper": "gripper"
    }
    
    # Simulation limits (from URDF) - what Isaac Lab expects
    JOINT_LIMITS = {
        "shoulder_pan": [-1.91986, 1.91986],    # Updated to match URDF
        "shoulder_lift": [-1.74533, 1.74533],   # Updated to match URDF
        "elbow_flex": [-1.69, 1.69],            # Updated to match URDF
        "wrist_flex": [-1.65806, 1.65806],      # Updated to match URDF
        "wrist_roll": [-2.74385, 2.84121],      # Updated to match URDF
        "gripper": [-0.174533, 1.74533]         # Updated to match URDF
    }
    
    # Real robot actual limits (from calibration) - using hardware joint names
    REAL_ROBOT_ACTUAL_LIMITS_HW = {
        "shoulder_pan": [-1.745329, 1.738843],  # Actual observed limits
        "shoulder_lift": [-1.745329, 1.739408],  # Actual observed limits
        "elbow": [-1.742182, 1.739034],  # Actual observed limits
        "wrist_1": [-1.740786, 1.745329],  # Actual observed limits
        "wrist_2": [-1.745329, 1.745329],  # Actual observed limits
        "gripper": [0.016273, 1.745329],  # Actual observed limits
    }

    # Real robot values are converted to radians then clamped to simulation limits
    
    def __init__(self, scene, args):
        self.scene = scene
        self.robot = scene["soarm101"]
        self.args = args
        self.communication = None
        self.running = False
        
        # Hardware state
        self.latest_data = None
        self.connected = False
        self.monitor_thread = None
    

    def convert_real_to_sim_range(self, hw_joint_name: str, isaac_joint_name: str, real_value_rad: float) -> float:
        """
        Convert real robot joint value (in radians) to simulation range using actual limits.
        
        Args:
            hw_joint_name: Hardware joint name (from calibration)
            isaac_joint_name: Isaac Lab joint name (for simulation limits)
            real_value_rad: Joint position from real robot (in radians)
            
        Returns:
            Joint position mapped to simulation range (in radians)
        """
        if hw_joint_name in self.REAL_ROBOT_ACTUAL_LIMITS_HW and isaac_joint_name in self.JOINT_LIMITS:
            # Get limits for this joint
            real_min, real_max = self.REAL_ROBOT_ACTUAL_LIMITS_HW[hw_joint_name]
            sim_min, sim_max = self.JOINT_LIMITS[isaac_joint_name]
            
            # Clamp real value to actual real robot limits
            real_value_rad = max(real_min, min(real_max, real_value_rad))
            
            # Normalize to [0, 1] using actual real robot range
            if real_max != real_min:
                normalized = (real_value_rad - real_min) / (real_max - real_min)
            else:
                normalized = 0.5
            
            # Map to simulation range
            sim_value = sim_min + normalized * (sim_max - sim_min)
            
            return sim_value
        else:
            # Pass through if no conversion available
            return real_value_rad

    def setup_hardware(self):
        """Setup hardware communication."""
        # Import ONLY when needed to avoid circular deps
        from simulation.communication import TCPCommunication, CommunicationData
        
        # Create TCP communication
        self.communication = TCPCommunication(host=self.args.host, port=self.args.port)
        
        # Connect
        if not self.communication.connect():
            print(f"Failed to connect to {self.args.host}:{self.args.port}")
            print("Make sure hardware driver is running:")
            print(f"  python communication/host_soarm_driver.py --port {self.args.port}")
            return False
        
        self.connected = True
        return True
    
    def start_monitoring_thread(self):
        """Start monitoring thread AFTER everything is initialized."""
        if self.connected and self.monitor_thread is None:
            self.monitor_thread = threading.Thread(target=self._monitor_hardware)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def _monitor_hardware(self):
        """Monitor hardware data in background."""
        while self.connected:
            try:
                # Get latest data using the get_latest_data method
                data = self.communication.get_latest_data()
                
                if data:
                    self.latest_data = data
                else:
                    # Try the alternative method if get_latest_data returns None
                    data = self.communication.receive_data(timeout=0.01)
                    if data:
                        self.latest_data = data
                    
                time.sleep(0.005)  # 200Hz
                
            except Exception as e:
                print(f"Hardware monitoring error: {e}")
                time.sleep(0.1)
    
    def update_robot(self):
        """Update robot from hardware data."""
        if not self.latest_data or not self.latest_data.leader_arm:
            return
        
        # Convert hardware data to Isaac Lab format
        num_envs = self.robot.num_instances
        positions = torch.zeros((num_envs, len(self.JOINT_NAMES)))
        

        
        for hw_joint, isaac_joint in self.JOINT_MAPPING.items():
            if hw_joint in self.latest_data.leader_arm:
                raw_value = self.latest_data.leader_arm[hw_joint]
                
                # Convert from degrees to radians (no range conversion)
                if hw_joint == "gripper":
                    # Gripper uses 0-100 range, convert to radians
                    # Map 0-100 range to actual robot gripper limits
                    real_min, real_max = self.REAL_ROBOT_ACTUAL_LIMITS_HW["gripper"]
                    normalized = raw_value / 100.0
                    angle_rad = real_min + normalized * (real_max - real_min)
                else:
                    # Other joints use degrees, convert to radians
                    angle_rad = math.radians(raw_value)
                
                # Convert from real robot range to simulation range
                angle_rad_converted = self.convert_real_to_sim_range(hw_joint, isaac_joint, angle_rad)
                
                joint_idx = self.JOINT_NAMES.index(isaac_joint)
                positions[:, joint_idx] = angle_rad_converted
        
        # Apply joint limits for safety (should be redundant after conversion)
        for i, joint_name in enumerate(self.JOINT_NAMES):
            if joint_name in self.JOINT_LIMITS:
                min_limit, max_limit = self.JOINT_LIMITS[joint_name]
                positions[:, i] = torch.clamp(positions[:, i], min_limit, max_limit)
        
        # Update robot using Isaac Lab API
        self.robot.set_joint_position_target(positions)
    
    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        self.connected = False
        
        if self.communication:
            self.communication.disconnect()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.2])
    
    # Create scene
    scene_cfg = SOARM101_TABLE_SCENE_CFG(args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Initialize teleoperation controller
    teleop = SimplifiedTeleoperation(scene, args_cli)
    
    # Setup hardware
    if not teleop.setup_hardware():
        return
    
    print("Ready! Move the real robot to see the virtual robot follow.")
    print("Press Ctrl+C to stop.")
    
    # Main simulation loop
    teleop.running = True
    sim_dt = sim.get_physics_dt()
    loop_dt = 1.0 / args_cli.update_rate
    count = 0
    
    # Start monitoring thread
    teleop.start_monitoring_thread()
    
    while simulation_app.is_running() and teleop.running:
        start_time = time.time()
        
        # Update robot from hardware
        teleop.update_robot()
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Status update
        if count % 60 == 0:
            data_status = "✅ Data OK" if teleop.latest_data else "❌ No Data"
            conn_status = "✅ Connected" if teleop.connected else "❌ Disconnected"
            
            if teleop.latest_data and teleop.latest_data.leader_arm:
                pan_angle = teleop.latest_data.leader_arm.get('shoulder_pan', 0)
                print(f"\r[{count:6d}] {conn_status} | {data_status} | Pan:{pan_angle:.1f}°", end="", flush=True)
            else:
                print(f"\r[{count:6d}] {conn_status} | {data_status}", end="", flush=True)
        
        # Maintain update rate
        elapsed = time.time() - start_time
        sleep_time = max(0, loop_dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Clean up
    if teleop:
        teleop.shutdown()


if __name__ == "__main__":
    main() 