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


import time
import threading
import argparse
from typing import Dict, Optional

# LeRobot imports
from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

from simulation.communication.interface import CommunicationData
from simulation.communication.tcp_server import TCPServer


class SOArmHardwareDriver:
    """SO-ARM hardware driver with pluggable communication protocols."""
    
    def __init__(self, communication_server):
        """
        Initialize hardware driver.
        
        Args:
            communication_server: Communication server instance
        """
        self.communication_server = communication_server
        self.running = False
        
        # Hardware state
        self.leader_arm = None
        self.follower_arm = None
        self.leader_connected = False
        self.follower_connected = False
        
        # Data streaming
        self.data_thread: Optional[threading.Thread] = None
        
        self.setup_hardware()
    
    def setup_hardware(self):
        """Initialize SO-ARM hardware."""
        # Setup Leader arm
        print("Setting up SO101 Leader arm...")
        try:
            leader_config = SO101LeaderConfig(
                port="/dev/ttyACM0",
                id="my_awesome_leader_arm"
            )
            
            self.leader_arm = SO101Leader(leader_config)
            self.leader_arm.connect(calibrate=False)
            self.leader_connected = True
            print(f"✅ SO101 Leader arm connected")
            print(f"Leader calibration: {self.leader_arm.calibration_fpath}")
        except Exception as e:
            print(f"❌ Failed to connect Leader arm: {e}")
            self.leader_connected = False
            print("⚠️ Running without real Leader arm (dummy data mode)")
        
        # Setup Follower arm (optional)
        print("Setting up SO101 Follower arm (optional)...")
        try:
            follower_config = SO101FollowerConfig(
                port="/dev/ttyACM1",
                id="my_awesome_follower_arm"
            )
            
            self.follower_arm = SO101Follower(follower_config)
            self.follower_arm.connect(calibrate=False)
            self.follower_connected = True
            print(f"✅ SO101 Follower arm connected")
            print(f"Follower calibration: {self.follower_arm.calibration_fpath}")
        except Exception as e:
            print(f"⚠️ Follower arm not available: {e}")
            self.follower_connected = False
            print("💡 Isaac Sim will work without real follower arm")
        
        # Summary
        if self.leader_connected and self.follower_connected:
            print("✅ Full teleoperation setup: Leader → Follower + Isaac Sim")
        elif self.leader_connected:
            print("⚡ Isaac Sim only setup: Leader → Isaac Sim")
        else:
            print("📊 Demo mode: Dummy data → Isaac Sim")
    
    def read_leader_data(self) -> Dict[str, float]:
        """Read leader arm data."""
        if self.leader_connected and self.leader_arm:
            try:
                action_data = self.leader_arm.get_action()
                return {
                    'shoulder_pan': action_data.get('shoulder_pan.pos', 0.0),
                    'shoulder_lift': action_data.get('shoulder_lift.pos', 0.0),
                    'elbow': action_data.get('elbow_flex.pos', 0.0),
                    'wrist_1': action_data.get('wrist_flex.pos', 0.0),
                    'wrist_2': action_data.get('wrist_roll.pos', 0.0),
                    'gripper': action_data.get('gripper.pos', 0.0)
                }
            except Exception as e:
                print(f"⚠️ Error reading leader arm: {e}")
                self.leader_connected = False
                
        # Dummy data for testing
        t = time.time()
        return {
            'shoulder_pan': 15.0 * (1 + 0.3 * (t % 10) / 10),
            'shoulder_lift': -50.0 * (1 + 0.2 * ((t + 3) % 8) / 8),
            'elbow': 60.0 * (1 + 0.4 * ((t + 6) % 12) / 12),
            'wrist_1': 30.0 * (1 + 0.6 * ((t + 9) % 6) / 6),
            'wrist_2': -10.0 * (1 + 0.3 * ((t + 12) % 15) / 15),
            'gripper': 5.0 * (1 + 0.2 * ((t + 2) % 7) / 7)
        }
    
    def read_follower_data(self) -> Optional[Dict[str, float]]:
        """Read follower arm data (optional)."""
        if self.follower_connected and self.follower_arm:
            try:
                obs_data = self.follower_arm.get_observation()
                return {
                    'shoulder_pan': obs_data.get('shoulder_pan.pos', 0.0),
                    'shoulder_lift': obs_data.get('shoulder_lift.pos', 0.0),
                    'elbow': obs_data.get('elbow_flex.pos', 0.0),
                    'wrist_1': obs_data.get('wrist_flex.pos', 0.0),
                    'wrist_2': obs_data.get('wrist_roll.pos', 0.0),
                    'gripper': obs_data.get('gripper.pos', 0.0)
                }
            except Exception as e:
                print(f"⚠️ Error reading follower arm: {e}")
                self.follower_connected = False
                
        # Return None if follower not connected
        return None
    
    def send_command_to_follower(self, leader_data: Dict[str, float]) -> None:
        """Send leader positions as commands to follower arm (if connected)."""
        if self.follower_connected and self.follower_arm and leader_data:
            try:
                # Convert format back to LeRobot format for follower commands
                action_dict = {
                    'shoulder_pan.pos': leader_data.get('shoulder_pan', 0.0),
                    'shoulder_lift.pos': leader_data.get('shoulder_lift', 0.0),
                    'elbow_flex.pos': leader_data.get('elbow', 0.0),
                    'wrist_flex.pos': leader_data.get('wrist_1', 0.0),
                    'wrist_roll.pos': leader_data.get('wrist_2', 0.0),
                    'gripper.pos': leader_data.get('gripper', 0.0)
                }
                
                # Send action to follower arm
                self.follower_arm.send_action(action_dict)
                
            except Exception as e:
                print(f"⚠️ Error sending command to follower arm: {e}")
                self.follower_connected = False
    
    def start_driver(self):
        """Start the hardware driver."""
        # Start communication server
        if not self.communication_server.start_server():
            print("Failed to start communication server")
            return False
        
        # Start data streaming thread
        self.running = True
        self.data_thread = threading.Thread(target=self._data_streaming_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        print(f"SO-ARM driver started using {self.communication_server.get_protocol_name()}")
        print(f"Server info: {self.communication_server.get_server_info()}")
        
        return True
    
    def stop_driver(self):
        """Stop the hardware driver."""
        self.running = False
        
        # Stop data streaming
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)
        
        # Stop communication server
        self.communication_server.stop_server()
        
        # Disconnect hardware
        if self.leader_arm and self.leader_connected:
            self.leader_arm.disconnect()
            print("Leader arm disconnected")
        
        if self.follower_arm and self.follower_connected:
            self.follower_arm.disconnect()
            print("Follower arm disconnected")
    
    def _data_streaming_loop(self):
        """Data streaming loop (runs in background thread)."""
        loop_count = 0
        
        while self.running:
            try:
                # Read hardware data
                leader_data = self.read_leader_data()
                follower_data = self.read_follower_data()  # May return None
                
                # Send leader commands to follower arm (if available)
                self.send_command_to_follower(leader_data)
                
                # Create communication data
                comm_data = CommunicationData(
                    leader_arm=leader_data,
                    follower_arm=follower_data,  # Include follower data
                    hardware_status={
                        "leader_connected": self.leader_connected,
                        "follower_connected": self.follower_connected
                    },
                    timestamp=time.time()
                )
                
                # Broadcast to all connected clients
                self.communication_server.broadcast_data(comm_data)
                
                loop_count += 1
                
                # Status update every 200 loops (1 second at 200Hz)
                if loop_count % 200 == 0:
                    server_info = self.communication_server.get_server_info()
                    print(f"📡 Data streaming: {loop_count} loops, {server_info['connected_clients']} clients")
                
                # 200Hz update rate
                time.sleep(0.005)
                
            except Exception as e:
                if self.running:
                    print(f"Data streaming error: {e}")
                    time.sleep(0.1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SO-ARM Hardware Driver for Isaac Lab")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=8888, help="Port number")
    
    args = parser.parse_args()
    
    print("SO-ARM Hardware Driver for Isaac Lab")
    print(f"Address: {args.host}:{args.port}")
    print("💡 Supports both Leader and Follower arms")
    print("=" * 50)
    
    # Create TCP communication server
    communication_server = TCPServer(host=args.host, port=args.port)
    
    # Create hardware driver
    driver = SOArmHardwareDriver(communication_server)
    
    try:
        if driver.start_driver():
            print("Driver started successfully. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
        else:
            print("Failed to start driver")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        driver.stop_driver()


if __name__ == "__main__":
    main() 