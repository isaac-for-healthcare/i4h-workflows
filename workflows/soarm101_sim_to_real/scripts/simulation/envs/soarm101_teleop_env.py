"""
SO-ARM 101 Teleoperation Gymnasium Environment
Integrates real robot control with Isaac Lab simulation
"""

import numpy as np
import torch
import threading
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

# Import the base environment
from .soarm101_env import SoArm101Env

# Communication imports will be done after Isaac Sim launch
TCPCommunication = None
CommunicationData = None

# Isaac Lab keyboard imports will be done after Isaac Sim launch
carb = None
omni = None


class SoArm101TeleopEnv(SoArm101Env):
    """
    SO-ARM 101 Teleoperation Environment
    
    This environment receives real robot joint positions via TCP and mirrors
    them in the Isaac Lab simulation with enhanced movement parameters.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8888, 
                 auto_connect: bool = True, **kwargs):
        """
        Initialize teleoperation environment.
        
        Args:
            host: TCP server host address
            port: TCP server port  
            auto_connect: Whether to automatically connect to real robot
            **kwargs: Additional arguments passed to base environment
        """
        # Set enhanced defaults for teleoperation
        if "sim_steps_per_action" not in kwargs:
            kwargs["sim_steps_per_action"] = 25  # Very smooth for real robot
        if "action_scale" not in kwargs:
            kwargs["action_scale"] = 1.0  # Direct mapping for real robot
        if "render_mode" not in kwargs:
            kwargs["render_mode"] = "human"  # Show GUI by default
            
        # Initialize base environment
        super().__init__(**kwargs)
        
        # Teleoperation parameters
        self.host = host
        self.port = port
        self.auto_connect = auto_connect
        
        # Real robot communication
        self.communication = None
        self.connected = False
        self.monitor_thread = None
        self.running = False
        
        # Joint mapping (hardware name -> Isaac Lab name)
        self.joint_mapping = {
            "shoulder_pan": "shoulder_pan",
            "shoulder_lift": "shoulder_lift", 
            "elbow": "elbow_flex",
            "wrist_1": "wrist_flex",
            "wrist_2": "wrist_roll",
            "gripper": "gripper"
        }
        
        # Real robot calibration limits (from hardware)
        self.real_robot_limits = {
            "shoulder_pan": [-1.735329, 1.738843],
            "shoulder_lift": [-1.745329, 1.739408], 
            "elbow": [-1.742182, 1.739034],
            "wrist_1": [-1.740786, 1.745329],
            "wrist_2": [-1.595329, 1.905329],
            "gripper": [-0.016273, 1.745329],
        }
        
        # Isaac Lab joint limits
        self.isaac_limits = {
            "shoulder_pan": [-1.91986, 1.91986],
            "shoulder_lift": [-1.74533, 1.74533],
            "elbow_flex": [-1.69, 1.69],
            "wrist_flex": [-1.65806, 1.65806],
            "wrist_roll": [-2.74385, 2.84121],
            "gripper": [-0.174533, 1.74533]
        }
        
        # Latest robot data
        self.latest_robot_data = None
        self.last_update_time = 0
        
        # Keyboard interface for manual reset
        self.input_interface = None
        self.keyboard = None
        self.keyboard_sub = None
        self.reset_requested = False
        
        # Auto-connect if enabled
        if self.auto_connect:
            self.connect_to_robot()
    
    def _init_isaac_lab(self):
        """Initialize Isaac Lab and import communication modules."""
        # Call parent initialization
        super()._init_isaac_lab()
        
        # Now import communication modules (after Isaac Sim is running)
        global TCPCommunication, CommunicationData, carb, omni
        from simulation.communication import TCPCommunication, CommunicationData
        
        # Import keyboard modules for reset functionality
        import carb.input
        import omni.appwindow
        carb = carb.input
        omni = omni.appwindow
        
        # Setup keyboard interface for manual reset
        self._setup_keyboard_interface()
    
    def _setup_keyboard_interface(self):
        """Setup keyboard interface for manual controls."""
        try:
            print("🎹 Setting up keyboard interface...")
            self.input_interface = carb.acquire_input_interface()
            self.keyboard = omni.get_default_app_window().get_keyboard()
            self.keyboard_sub = self.input_interface.subscribe_to_keyboard_events(
                self.keyboard, self._on_keyboard_event
            )
            print("✅ Keyboard interface ready!")
            print("💡 Press 'R' to reset the environment")
            print("💡 Press 'ESC' to close the application")
        except Exception as e:
            print(f"⚠️ Could not setup keyboard interface: {e}")
            print("💡 Manual reset via 'R' key will not be available")
    
    def _on_keyboard_event(self, event, *args):
        """Handle keyboard events."""
        if event.type == carb.KeyboardEventType.KEY_PRESS:
            if event.input == carb.KeyboardInput.R:
                print("🔄 Reset requested via 'R' key!")
                self.reset_requested = True
                return True
            elif event.input == carb.KeyboardInput.ESCAPE:
                print("🛑 Exit requested via 'ESC' key!")
                self.close()
                return True
        return False
    
    def connect_to_robot(self) -> bool:
        """Connect to real robot via TCP."""
        print(f"🤖 Connecting to real robot at {self.host}:{self.port}...")
        
        try:
            # Import after Isaac Sim launch
            from simulation.communication import TCPCommunication
            
            self.communication = TCPCommunication(host=self.host, port=self.port)
            
            if self.communication.connect():
                self.connected = True
                self.running = True
                
                # Start monitoring thread
                self.monitor_thread = threading.Thread(target=self._monitor_robot)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
                
                print("✅ Connected to real robot!")
                print("💡 Move your real robot to see it mirrored in simulation")
                return True
            else:
                print("❌ Failed to connect to real robot")
                print(f"💡 Make sure hardware driver is running:")
                print(f"   python communication/host_soarm_driver.py --port {self.port}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def disconnect_from_robot(self):
        """Disconnect from real robot."""
        if self.connected:
            self.running = False
            self.connected = False
            
            if self.communication:
                self.communication.disconnect()
                
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)
                
            print("🔌 Disconnected from real robot")
    
    def _monitor_robot(self):
        """Monitor real robot data in background thread."""
        while self.running and self.connected:
            try:
                data = self.communication.get_latest_data()
                if data and data.leader_arm:
                    self.latest_robot_data = data.leader_arm
                    self.last_update_time = time.time()
                    
                time.sleep(0.005)  # 200Hz monitoring
                
            except Exception as e:
                if self.running:
                    print(f"⚠️ Robot monitoring error: {e}")
                    time.sleep(0.1)
    
    def _convert_real_to_sim(self, hw_joint: str, isaac_joint: str, real_value_rad: float) -> float:
        """Convert real robot joint value to simulation range."""
        if hw_joint in self.real_robot_limits and isaac_joint in self.isaac_limits:
            # Get limits
            real_min, real_max = self.real_robot_limits[hw_joint]
            sim_min, sim_max = self.isaac_limits[isaac_joint]
            
            # Clamp to real robot limits
            real_value_rad = max(real_min, min(real_max, real_value_rad))
            
            # Normalize to [0, 1]
            if real_max != real_min:
                normalized = (real_value_rad - real_min) / (real_max - real_min)
            else:
                normalized = 0.5
                
            # Map to simulation range
            sim_value = sim_min + normalized * (sim_max - sim_min)
            return sim_value
        else:
            return real_value_rad
    
    def _get_robot_action(self) -> Optional[np.ndarray]:
        """Get current action from real robot data."""
        if not self.latest_robot_data:
            return None
            
        # Check if data is recent (within last 0.1 seconds)
        if time.time() - self.last_update_time > 0.1:
            return None
            
        # Convert robot data to action
        action = np.zeros(6)  # 6 joints
        
        for hw_joint, isaac_joint in self.joint_mapping.items():
            if hw_joint in self.latest_robot_data:
                raw_value = self.latest_robot_data[hw_joint]
                
                # Convert from degrees to radians (except gripper)
                if hw_joint == "gripper":
                    # Gripper uses 0-100 range, convert to radians
                    real_min, real_max = self.real_robot_limits["gripper"]
                    normalized = raw_value / 100.0
                    angle_rad = real_min + normalized * (real_max - real_min)
                else:
                    # Other joints use degrees
                    angle_rad = np.radians(raw_value)
                
                # Convert to simulation range
                sim_value = self._convert_real_to_sim(hw_joint, isaac_joint, angle_rad)
                
                # Set in action array
                joint_idx = self.JOINT_NAMES.index(isaac_joint)
                action[joint_idx] = sim_value
        
        return action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action array. Use np.array([]) for auto mode (uses real robot data).
        """
        # Check if this is auto mode (empty action array means use robot data)
        if len(action) == 0 and self.connected:
            robot_action = self._get_robot_action()
            if robot_action is not None:
                action = robot_action
            else:
                # No robot data, use last known position
                current_obs = self._get_observation()
                action = current_obs[:6]  # Use current joint positions
        elif len(action) == 0:
            # No robot connected and auto mode, stay in place
            current_obs = self._get_observation()
            action = current_obs[:6]
            
        # Execute step using parent method
        return super().step(action)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment completely (robot + all objects)."""
        print("🔄 Performing FULL environment reset (all objects)...")
        
        # Step 1: Reset the entire simulation context
        print("🔄 Resetting simulation context...")
        self.sim.reset()
        
        # Step 2: Reset the scene (this resets ALL objects to initial positions)
        print("🔄 Resetting all scene objects...")
        self.scene.reset()
        
        # Step 3: Now call parent reset for robot-specific setup
        obs, info = super().reset(seed, options)
        
        # Step 4: Add teleoperation status to info
        info["robot_connected"] = self.connected
        info["last_robot_update"] = self.last_update_time
        
        print("✅ Full environment reset complete (robot + all objects)")
        return obs, info

    def auto_step(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step using real robot data automatically."""
        # Check for reset request
        if hasattr(self, 'reset_requested') and self.reset_requested:
            print("🔄 Performing FULL environment reset (all objects)...")
            self.reset_requested = False
            obs, info = self.reset()
            return obs, 0.0, False, False, info
            
        return self.step(np.array([]))  # Empty array triggers auto mode
    
    def _get_info(self) -> Dict:
        """Get additional info including teleoperation status."""
        info = super()._get_info()
        
        # Add teleoperation info
        info["robot_connected"] = self.connected
        info["robot_host"] = f"{self.host}:{self.port}"
        
        if self.latest_robot_data:
            info["robot_data_age"] = time.time() - self.last_update_time
            info["robot_joints"] = dict(self.latest_robot_data)
        
        return info
    
    def close(self):
        """Close the environment."""
        # Clean up keyboard interface
        if self.keyboard_sub is not None and self.input_interface is not None:
            try:
                self.input_interface.unsubscribe_to_keyboard_events(
                    self.keyboard, self.keyboard_sub
                )
                print("🎹 Keyboard interface cleaned up")
            except:
                pass
        
        # Disconnect from robot
        self.disconnect_from_robot()
        
        # Then close parent environment
        super().close()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "last_update": self.last_update_time,
            "data_age": time.time() - self.last_update_time if self.last_update_time > 0 else float('inf'),
            "has_recent_data": (time.time() - self.last_update_time) < 0.1 if self.last_update_time > 0 else False
        }


# Define joint names for compatibility
SoArm101TeleopEnv.JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"] 