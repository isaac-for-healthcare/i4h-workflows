#!/usr/bin/env python3
"""
SO-ARM101 Hardware Driver - ULTRA HIGH-FREQUENCY Host Machine
Communicates with real SO-ARM hardware using LeRobot and sends data via TCP socket.
No ROS2 required on host machine.

PERFORMANCE OPTIMIZATIONS:
- Normal mode: 100Hz hardware sampling and data transmission  
- Real-time mode: 140Hz for minimal latency (7ms intervals)
- Optimized TCP data streaming with JSON compression
- Concurrent hardware control and network serving
"""

import time
import json
import socket
import threading
import argparse
from pathlib import Path
import sys

def setup_lerobot_path():
    """Add LeRobot to Python path"""
    # lerobot and isaacsim_soarm_real_sim are in the same directory
    script_dir = Path(__file__).parent  # isaacsim_soarm_real_sim directory
    lerobot_path = script_dir.parent / "lerobot"  # lerobot in parent directory
    
    if lerobot_path.exists():
        sys.path.insert(0, str(lerobot_path))
        print(f"Added LeRobot path: {lerobot_path}")
    else:
        print(f"Warning: LeRobot path not found: {lerobot_path}")
        print(f"Expected path: {lerobot_path}")
        print(f"Script directory: {script_dir}")

class SOArmTCPServer:
    def __init__(self, host='localhost', port=8888, realtime=False):
        self.host = host
        self.port = port
        self.realtime = realtime
        self.server_socket = None
        self.clients = []
        self.running = False
        
        # Initialize SO-ARM hardware with status tracking
        self.leader_arm = None
        self.follower_arm = None
        self.leader_connected = False
        self.follower_connected = False
        self.follower_optional = True  # Make follower optional by default
        
        self.setup_soarm()
        
    def setup_soarm(self):
        """Initialize SO-ARM hardware using LeRobot - with optional follower"""
        try:
            setup_lerobot_path()
            
            # Import LeRobot classes for SO101
            from lerobot.common.teleoperators.so101_leader import (
                SO101Leader, 
                SO101LeaderConfig
            )
            from lerobot.common.robots.so101_follower import (
                SO101Follower,
                SO101FollowerConfig
            )
            
            # Setup Leader arm (required)
            print("🔧 Setting up SO101 Leader arm...")
            try:
                leader_config = SO101LeaderConfig(
                    port="/dev/ttyACM1",  # Leader arm port
                    id="my_awesome_leader_arm"  # Use your calibrated ID
                )
                
                self.leader_arm = SO101Leader(leader_config)
                self.leader_arm.connect(calibrate=False)  # Use existing calibration data
                self.leader_connected = True
                print("✅ SO101 Leader arm connected and calibrated")
                print(f"📁 Leader calibration: {self.leader_arm.calibration_fpath}")
                
            except Exception as e:
                print(f"❌ Failed to connect Leader arm: {e}")
                self.leader_connected = False
                print("⚠️ Running without real Leader arm (dummy data mode)")
            
            # Setup Follower arm (optional)
            print("🔧 Setting up SO101 Follower arm (optional)...")
            try:
                follower_config = SO101FollowerConfig(
                    port="/dev/ttyACM0",  # Follower arm port
                    id="my_awesome_follower_arm"  # Use your calibrated ID
                )
                
                self.follower_arm = SO101Follower(follower_config)
                self.follower_arm.connect(calibrate=False)  # Use existing calibration data
                self.follower_connected = True
                print("✅ SO101 Follower arm connected and calibrated")
                print(f"📁 Follower calibration: {self.follower_arm.calibration_fpath}")
                
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
            
            if self.realtime:
                print("🚀 Real-time mode enabled for minimal latency")
            
        except ImportError as e:
            print(f"❌ LeRobot dependencies missing: {e}")
            print("📋 Please install LeRobot dependencies:")
            print("  cd /home/venn/Desktop/code/lerobot")
            print("  pip install -e .")
            print("📊 Using dummy data mode...")
            self.leader_connected = False
            self.follower_connected = False
        except Exception as e:
            print(f"❌ Error initializing SO-ARM hardware: {e}")
            print("📊 Using dummy data mode...")
            self.leader_connected = False
            self.follower_connected = False
    
    def read_leader_arm(self):
        """Read leader arm joint positions"""
        if self.leader_connected and self.leader_arm:
            try:
                # LeRobot SO101 Leader returns dict with "{motor}.pos" keys
                action_data = self.leader_arm.get_action()
                
                # Convert LeRobot format to our format
                return {
                    'shoulder_pan': action_data.get('shoulder_pan.pos', 0.0),
                    'shoulder_lift': action_data.get('shoulder_lift.pos', 0.0), 
                    'elbow': action_data.get('elbow_flex.pos', 0.0),  # elbow_flex -> elbow
                    'wrist_1': action_data.get('wrist_flex.pos', 0.0),  # wrist_flex -> wrist_1
                    'wrist_2': action_data.get('wrist_roll.pos', 0.0),  # wrist_roll -> wrist_2
                    'gripper': action_data.get('gripper.pos', 0.0)
                }
            except Exception as e:
                print(f"⚠️ Error reading leader arm: {e}")
                self.leader_connected = False  # Mark as disconnected
                
        # Dummy data with smooth motion (for demo or when disconnected)
        t = time.time()
        return {
            'shoulder_pan': 15.0 * (1 + 0.3 * (t % 10) / 10),  # More realistic angles
            'shoulder_lift': -50.0 * (1 + 0.2 * ((t + 3) % 8) / 8),
            'elbow': 60.0 * (1 + 0.4 * ((t + 6) % 12) / 12),
            'wrist_1': 30.0 * (1 + 0.6 * ((t + 9) % 6) / 6),
            'wrist_2': -10.0 * (1 + 0.3 * ((t + 12) % 15) / 15),
            'gripper': 5.0 * (1 + 0.2 * ((t + 2) % 7) / 7)
        }
    
    def read_follower_arm(self):
        """Read follower arm joint positions (optional)"""  
        if self.follower_connected and self.follower_arm:
            try:
                # LeRobot SO101 Follower returns dict with "{motor}.pos" keys from observation
                obs_data = self.follower_arm.get_observation()
                
                # Convert LeRobot format to our format
                return {
                    'shoulder_pan': obs_data.get('shoulder_pan.pos', 0.0),
                    'shoulder_lift': obs_data.get('shoulder_lift.pos', 0.0),
                    'elbow': obs_data.get('elbow_flex.pos', 0.0),  # elbow_flex -> elbow
                    'wrist_1': obs_data.get('wrist_flex.pos', 0.0),  # wrist_flex -> wrist_1
                    'wrist_2': obs_data.get('wrist_roll.pos', 0.0),  # wrist_roll -> wrist_2
                    'gripper': obs_data.get('gripper.pos', 0.0)
                }
            except Exception as e:
                print(f"⚠️ Error reading follower arm: {e}")
                self.follower_connected = False  # Mark as disconnected
                
        # Return None if follower not connected (Isaac Sim will still work)
        return None
    
    def start_server(self):
        """Start TCP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"SO-ARM TCP Server started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    print(f"Client connected from {addr}")
                    
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")
                        
        except Exception as e:
            print(f"Error starting server: {e}")
            
    def send_command_to_follower(self, leader_data):
        """Send leader positions as commands to follower arm (if connected)"""
        if self.follower_connected and self.follower_arm and leader_data:
            try:
                # Convert our format back to LeRobot format for follower commands
                action_dict = {
                    'shoulder_pan.pos': leader_data.get('shoulder_pan', 0.0),
                    'shoulder_lift.pos': leader_data.get('shoulder_lift', 0.0),
                    'elbow_flex.pos': leader_data.get('elbow', 0.0),  # elbow -> elbow_flex
                    'wrist_flex.pos': leader_data.get('wrist_1', 0.0),  # wrist_1 -> wrist_flex
                    'wrist_roll.pos': leader_data.get('wrist_2', 0.0),  # wrist_2 -> wrist_roll
                    'gripper.pos': leader_data.get('gripper', 0.0)
                }
                
                # Send action to follower arm
                self.follower_arm.send_action(action_dict)
                
            except Exception as e:
                print(f"⚠️ Error sending command to follower arm: {e}")
                self.follower_connected = False  # Mark as disconnected
    
    def handle_client(self, client_socket, addr):
        """Handle individual client connection with optimized real-time performance"""
        self.clients.append(client_socket)
        
        # Print connection status
        status = []
        if self.leader_connected:
            status.append("Leader✅")
        if self.follower_connected:
            status.append("Follower✅")
        if not status:
            status.append("Demo📊")
        print(f"📱 Client {addr} connected ({' + '.join(status)})")
        
        try:
            while self.running:
                # Read SO-ARM data
                leader_data = self.read_leader_arm() 
                follower_data = self.read_follower_arm()  # May return None
                
                # Send leader commands to follower arm (if available)
                self.send_command_to_follower(leader_data)
                
                # Create message (follower_data may be None)
                message = {
                    'timestamp': time.time(),
                    'leader_arm': leader_data,
                    'follower_arm': follower_data,  # Isaac Sim only needs leader_arm
                    'status': {
                        'leader_connected': self.leader_connected,
                        'follower_connected': self.follower_connected
                    }
                }
                
                # Send data
                json_str = json.dumps(message) + '\n'
                client_socket.send(json_str.encode('utf-8'))
                
                # Ultra high-frequency update rate for minimal latency
                if self.realtime:
                    time.sleep(0.007)  # 140Hz for real-time (7ms interval)
                else:
                    time.sleep(0.01)   # 100Hz for normal mode (10ms interval)
                
        except Exception as e:
            print(f"❌ Error handling client {addr}: {e}")
        finally:
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()
            print(f"👋 Client {addr} disconnected")
    
    def stop_server(self):
        """Stop TCP server"""
        self.running = False
        
        # Disconnect SO-ARM hardware
        try:
            if self.leader_arm:
                self.leader_arm.disconnect()
                print("Leader arm disconnected")
        except Exception as e:
            print(f"Error disconnecting leader arm: {e}")
            
        try:
            if self.follower_arm:
                self.follower_arm.disconnect()
                print("Follower arm disconnected")
        except Exception as e:
            print(f"Error disconnecting follower arm: {e}")
        
        # Close network connections
        if self.server_socket:
            self.server_socket.close()
        for client in self.clients:
            client.close()
        self.clients.clear()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SO-ARM101 Hardware Driver - TCP Server with Optional Follower')
    parser.add_argument('--port', '-p', type=int, default=8888, 
                        help='TCP server port (default: 8888)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='TCP server host (default: 0.0.0.0)')
    parser.add_argument('--realtime', '-r', action='store_true',
                        help='Enable real-time mode (50Hz) for minimal latency')
    args = parser.parse_args()
    
    print("🦾 SO-ARM101 Hardware Driver - Host Machine")
    print("=" * 50)
    print(f"🌐 Starting TCP server on {args.host}:{args.port}")
    if args.realtime:
        print("🚀 Ultra real-time mode enabled (140Hz updates) - Minimal latency!")
    else:
        print("📊 High-frequency mode (100Hz updates)")
    print("💡 Follower arm is optional - Isaac Sim works without it")
    print("=" * 50)
    
    # Create and start server
    server = SOArmTCPServer(host=args.host, port=args.port, realtime=args.realtime)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.stop_server()
        print("✅ Server stopped.")

if __name__ == "__main__":
    main() 