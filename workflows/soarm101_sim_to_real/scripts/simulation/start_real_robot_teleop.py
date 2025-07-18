#!/usr/bin/env python3
"""
Complete Real Robot Teleoperation Setup for SO-ARM 101
This script handles both hardware driver and simulation setup
"""

import sys
import os
import subprocess
import time
import signal
import argparse
from typing import Optional
import threading

# Add scripts to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.insert(0, scripts_dir)


class RealRobotTeleopSystem:
    """Complete real robot teleoperation system manager."""
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.hardware_process: Optional[subprocess.Popen] = None
        self.simulation_process: Optional[subprocess.Popen] = None
        self.running = False
        
    def check_hardware_driver_running(self) -> bool:
        """Check if hardware driver is already running."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except:
            return False
    
    def start_hardware_driver(self) -> bool:
        """Start the hardware driver process."""
        if self.check_hardware_driver_running():
            print(f"✅ Hardware driver already running on {self.host}:{self.port}")
            return True
            
        print(f"🚀 Starting hardware driver on {self.host}:{self.port}...")
        
        try:
            # Start hardware driver
            cmd = [
                sys.executable, 
                "communication/host_soarm_driver.py",
                "--host", self.host,
                "--port", str(self.port)
            ]
            
            self.hardware_process = subprocess.Popen(
                cmd,
                cwd=scripts_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait for driver to start
            print("⏳ Waiting for hardware driver to initialize...")
            for i in range(30):  # 30 second timeout
                if self.check_hardware_driver_running():
                    print("✅ Hardware driver started successfully!")
                    return True
                time.sleep(1)
                print(f"   Waiting... ({i+1}/30)")
                
            print("❌ Hardware driver failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start hardware driver: {e}")
            return False
    
    def start_simulation_with_gym(self):
        """Start simulation using Gymnasium interface."""
        print("🎮 Starting Isaac Lab simulation with Gymnasium interface...")
        
        try:
            import gymnasium as gym
            import simulation.envs
            
            print("Creating teleoperation environment...")
            
            # Create teleoperation environment with RESPONSIVE settings
            env = gym.make("SoArm101Teleop-v0",
                          host=self.host,
                          port=self.port,
                          auto_connect=True,
                          sim_steps_per_action=25)     # 🚀 More responsive (4x faster)
            
            print("✅ Teleoperation environment created!")
            print(f"🔗 Connected to robot at {self.host}:{self.port}")
            print("💡 Move your real robot to see it mirrored in Isaac Sim!")
            print("🎹 Keyboard controls:")
            print("   'R' - Reset environment")
            print("   'ESC' - Exit application") 
            print("🛑 Press Ctrl+C to stop")
            
            # Main loop
            obs, info = env.reset()
            step_count = 0
            
            try:
                while True:
                    # Step with robot data (automatic mode)
                    obs, reward, terminated, truncated, info = env.auto_step()
                    step_count += 1
                    
                    # Print status every 60 steps (~1 second at 60Hz)
                    if step_count % 60 == 0:
                        status = env.get_connection_status()
                        if status["connected"] and status["has_recent_data"]:
                            age = status["data_age"]
                            print(f"\r[{step_count:6d}] ✅ Connected | Data age: {age:.3f}s", end="", flush=True)
                        elif status["connected"]:
                            print(f"\r[{step_count:6d}] ⚠️  Connected | No recent data", end="", flush=True)
                        else:
                            print(f"\r[{step_count:6d}] ❌ Disconnected", end="", flush=True)
                    
                    if terminated or truncated:
                        obs, info = env.reset()
                        
            except KeyboardInterrupt:
                print("\n🛑 Stopping simulation...")
                
            finally:
                env.close()
                print("✅ Simulation closed")
                
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def start_simulation_direct(self):
        """Start simulation using direct script (alternative method)."""
        print("🎮 Starting Isaac Lab simulation (direct)...")
        
        try:
            cmd = [
                sys.executable,
                "scripts/real_to_sim_teleoperation.py", 
                "--host", self.host,
                "--port", str(self.port),
                "--update_rate", "200"
            ]
            
            self.simulation_process = subprocess.Popen(
                cmd,
                cwd=scripts_dir
            )
            
            print("✅ Simulation started!")
            print("💡 Move your real robot to see it mirrored in Isaac Sim!")
            
            # Wait for simulation to finish
            self.simulation_process.wait()
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
    
    def stop_hardware_driver(self):
        """Stop the hardware driver."""
        if self.hardware_process:
            print("🛑 Stopping hardware driver...")
            self.hardware_process.terminate()
            try:
                self.hardware_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.hardware_process.kill()
            self.hardware_process = None
            print("✅ Hardware driver stopped")
    
    def stop_simulation(self):
        """Stop the simulation."""
        if self.simulation_process:
            print("🛑 Stopping simulation...")
            self.simulation_process.terminate()
            try:
                self.simulation_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.simulation_process.kill()
            self.simulation_process = None
            print("✅ Simulation stopped")
    
    def cleanup(self):
        """Clean up all processes."""
        self.running = False
        self.stop_simulation()
        self.stop_hardware_driver()
    
    def run_complete_system(self, use_gym: bool = True):
        """Run the complete teleoperation system."""
        self.running = True
        
        def signal_handler(signum, frame):
            print("\n🛑 Received interrupt signal...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Step 1: Start hardware driver
            if not self.start_hardware_driver():
                print("❌ Failed to start hardware driver. Exiting.")
                return False
                
            # Step 2: Start simulation
            if use_gym:
                self.start_simulation_with_gym()
            else:
                self.start_simulation_direct()
                
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            self.cleanup()
        
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SO-ARM 101 Real Robot Teleoperation System")
    parser.add_argument("--host", default="localhost", help="Host address (default: localhost)")
    parser.add_argument("--port", type=int, default=8888, help="Port number (default: 8888)")
    parser.add_argument("--gym", action="store_true", help="Use Gymnasium interface (default)")
    parser.add_argument("--direct", action="store_true", help="Use direct script interface")
    parser.add_argument("--check", action="store_true", help="Only check if hardware driver is running")
    
    args = parser.parse_args()
    
    print("SO-ARM 101 Real Robot Teleoperation System")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print()
    
    system = RealRobotTeleopSystem(host=args.host, port=args.port)
    
    if args.check:
        if system.check_hardware_driver_running():
            print(f"✅ Hardware driver is running on {args.host}:{args.port}")
        else:
            print(f"❌ Hardware driver is NOT running on {args.host}:{args.port}")
        return
    
    # Determine which interface to use
    use_gym = not args.direct  # Default to Gymnasium unless --direct is specified
    
    if use_gym:
        print("🎮 Using Gymnasium interface (recommended)")
    else:
        print("🎮 Using direct script interface")
    
    print()
    print("📋 System will:")
    print("  1. Start hardware driver (if not running)")
    print("  2. Connect to real SO-ARM robot")
    print("  3. Start Isaac Lab simulation")
    print("  4. Mirror real robot movements in simulation")
    print()
    print("💡 Make sure your real robot is connected and accessible!")
    print("🛑 Press Ctrl+C to stop the system")
    print()
    
    input("Press Enter to start the system...")
    
    success = system.run_complete_system(use_gym=use_gym)
    
    if success:
        print("\n🎉 System completed successfully!")
    else:
        print("\n❌ System encountered errors")
        sys.exit(1)


if __name__ == "__main__":
    main() 