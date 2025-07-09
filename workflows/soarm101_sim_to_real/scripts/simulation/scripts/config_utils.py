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


import yaml
import os
import re
from typing import Dict, List, Tuple, Optional

# Default paths
DEFAULT_YAML_CONFIG = "../config/so101_real_robot_params.yaml"
DEFAULT_ENV_CONFIG = "../config/soarm101_robot_cfg.py"

def load_robot_config(config_path: str = DEFAULT_YAML_CONFIG) -> Dict:
    """Load robot configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        print("📋 Loading defaults from env config file...")
        return load_defaults_from_env_config()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML config: {e}")
        print("📋 Loading defaults from env config file...")
        return load_defaults_from_env_config()

def load_defaults_from_env_config(env_config_path: str = DEFAULT_ENV_CONFIG) -> Dict:
    """Extract default values from robot config file."""
    
    if not os.path.exists(env_config_path):
        print(f"⚠️  Robot config file not found: {env_config_path}")
        return get_fallback_config()
    
    try:
        with open(env_config_path, 'r') as f:
            content = f.read()
        
        # Note: Robot config file only contains ArticulationCfg, no termination configs
        # We extract motor parameters from actuator configurations
        
        # Extract motor parameters
        effort_limit_match = re.search(r'effort_limit=([0-9.]+)', content)
        velocity_limit_match = re.search(r'velocity_limit=([0-9.]+)', content)
        stiffness_match = re.search(r'stiffness=([0-9.]+)', content)
        damping_match = re.search(r'damping=([0-9.]+)', content)
        
        effort_limit = float(effort_limit_match.group(1)) if effort_limit_match else 5.2
        velocity_limit = float(velocity_limit_match.group(1)) if velocity_limit_match else 6.28
        stiffness = float(stiffness_match.group(1)) if stiffness_match else 80.0
        damping = float(damping_match.group(1)) if damping_match else 8.0
        
        # Build configuration
        config = {
            'joints': {},
            'control_params': {
                'effort_limit_nm': effort_limit,
                'velocity_limit_rad_s': velocity_limit,
                'isaac_sim_stiffness': stiffness,
                'isaac_sim_damping': damping,
                'p_coefficient': 16,
                'i_coefficient': 0,
                'd_coefficient': 32
            },
            'data_collection': {
                'observation_features': [
                    f"{joint}.pos" for joint in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
                ],
                'action_features': [
                    f"{joint}.pos" for joint in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
                ],
                'default_fps': 30,
                'max_fps': 120
            }
        }
        
        # Add joint configurations with fallback limits
        # Note: Robot config doesn't contain joint limits, use fallback values
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        for i, joint_name in enumerate(joint_names):
            # Use fallback limits since robot config doesn't have them
            limits = [-3.14, 3.14]  # Default to full rotation, will be updated by calibration
            config['joints'][joint_name] = {
                'id': i + 1,
                'position_limits_rad': limits,
                'position_limits_deg': [limits[0] * 180 / 3.14159, limits[1] * 180 / 3.14159],
                'velocity_limit_rad_s': velocity_limit,
                'effort_limit_nm': effort_limit,
                'homing_offset': 0,
                'encoder_range': [0, 4096]
            }
        
        print(f"✅ Loaded defaults from robot config: {len(joint_names)} joints")
        print(f"📝 Note: Robot config doesn't contain joint limits, using fallback values")
        return config
        
    except Exception as e:
        print(f"❌ Error reading robot config: {e}")
        return get_fallback_config()

def get_fallback_config() -> Dict:
    """Get hardcoded fallback configuration when all else fails."""
    return {
        'joints': {
            'shoulder_pan': {
                'id': 1,
                'position_limits_rad': [-2.098, 2.098],
                'position_limits_deg': [-120.25, 120.25],
                'velocity_limit_rad_s': 6.28,
                'effort_limit_nm': 5.2,
                'homing_offset': -57,
                'encoder_range': [700, 3435]
            },
            'shoulder_lift': {
                'id': 2,
                'position_limits_rad': [-1.818, 1.818],
                'position_limits_deg': [-104.2, 104.2],
                'velocity_limit_rad_s': 6.28,
                'effort_limit_nm': 5.2,
                'homing_offset': 1267,
                'encoder_range': [811, 3181]
            },
            'elbow_flex': {
                'id': 3,
                'position_limits_rad': [-1.746, 1.746],
                'position_limits_deg': [-100.05, 100.05],
                'velocity_limit_rad_s': 6.28,
                'effort_limit_nm': 5.2,
                'homing_offset': -761,
                'encoder_range': [801, 3077]
            },
            'wrist_flex': {
                'id': 4,
                'position_limits_rad': [-1.805, 1.805],
                'position_limits_deg': [-103.4, 103.4],
                'velocity_limit_rad_s': 6.28,
                'effort_limit_nm': 5.2,
                'homing_offset': 336,
                'encoder_range': [896, 3249]
            },
            'wrist_roll': {
                'id': 5,
                'position_limits_rad': [-2.955, 2.955],
                'position_limits_deg': [-169.25, 169.25],
                'velocity_limit_rad_s': 6.28,
                'effort_limit_nm': 5.2,
                'homing_offset': -229,
                'encoder_range': [79, 3931]
            },
            'gripper': {
                'id': 6,
                'position_limits_rad': [-1.168, 1.168],
                'position_limits_deg': [-66.9, 66.9],
                'velocity_limit_rad_s': 6.28,
                'effort_limit_nm': 5.2,
                'homing_offset': -443,
                'encoder_range': [2002, 3525]
            }
        },
        'control_params': {
            'effort_limit_nm': 5.2,
            'velocity_limit_rad_s': 6.28,
            'isaac_sim_stiffness': 80.0,
            'isaac_sim_damping': 8.0,
            'p_coefficient': 16,
            'i_coefficient': 0,
            'd_coefficient': 32
        },
        'data_collection': {
            'observation_features': [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", 
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
            ],
            'action_features': [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", 
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
            ],
            'default_fps': 30,
            'max_fps': 120
        }
    }

def update_env_config_file(calibration_data: Dict, 
                          env_config_path: str = DEFAULT_ENV_CONFIG) -> bool:
    """
    Update robot config file with user calibration data.
    
    Args:
        calibration_data: Processed calibration data with joint limits
        env_config_path: Path to robot config file
    
    Returns:
        True if successful, False otherwise
    """
    
    if not os.path.exists(env_config_path):
        print(f"❌ Robot config file not found: {env_config_path}")
        return False
    
    try:
        # Read current file
        with open(env_config_path, 'r') as f:
            content = f.read()
        
        # Note: Robot config file only contains ArticulationCfg, no termination configs
        # We only update initial joint positions to safe middle positions
        
        # Update initial joint positions to middle of range
        joint_pos_section = re.search(r'joint_pos=\{([^}]+)\}', content)
        if joint_pos_section:
            # Create new joint positions (middle of range for safety)
            new_positions = []
            for joint_name, limits in calibration_data.items():
                if isinstance(limits, (list, tuple)) and len(limits) == 2:
                    middle_pos = (limits[0] + limits[1]) / 2.0
                    new_positions.append(f'            "{joint_name}": {middle_pos:.3f}')
                    print(f"  🎯 {joint_name}: safe position = {middle_pos:.3f} rad (range: {limits[0]:.3f} to {limits[1]:.3f})")
                else:
                    new_positions.append(f'            "{joint_name}": 0.0')
                    print(f"  ⚠️  {joint_name}: using default position 0.0 rad")
            
            new_joint_pos = "joint_pos={\n" + ",\n".join(new_positions) + ",\n        }"
            content = re.sub(r'joint_pos=\{[^}]+\}', new_joint_pos, content, flags=re.DOTALL)
            print(f"  🏠 Updated initial joint positions to safe middle positions")
        else:
            print(f"  ⚠️  Could not find joint_pos section in robot config")
        
        # Write updated content
        with open(env_config_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated robot config file: {env_config_path}")
        print(f"📝 Note: Robot config only contains initial positions, not joint limits")
        print(f"📝 Joint limits should be enforced in your environment/MDP configuration")
        return True
        
    except Exception as e:
        print(f"❌ Error updating robot config: {e}")
        return False

def get_joint_limits(config: Dict) -> Dict[str, Tuple[float, float]]:
    """Extract joint limits from configuration."""
    joint_limits = {}
    
    if 'joints' in config:
        for joint_name, joint_config in config['joints'].items():
            if 'position_limits_rad' in joint_config:
                limits = joint_config['position_limits_rad']
                joint_limits[joint_name] = (limits[0], limits[1])
    
    return joint_limits

def get_joint_names(config: Dict) -> List[str]:
    """Get ordered list of joint names from configuration."""
    if 'joints' in config:
        return list(config['joints'].keys())
    return ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

def get_motor_params(config: Dict) -> Dict[str, Dict]:
    """Extract motor parameters from configuration."""
    motor_params = {}
    
    if 'joints' in config:
        for joint_name, joint_config in config['joints'].items():
            motor_params[joint_name] = {
                'id': joint_config.get('id', 0),
                'velocity_limit': joint_config.get('velocity_limit_rad_s', 6.28),
                'effort_limit': joint_config.get('effort_limit_nm', 5.2),
                'homing_offset': joint_config.get('homing_offset', 0),
                'encoder_range': joint_config.get('encoder_range', [0, 4096])
            }
    
    return motor_params

def get_control_params(config: Dict) -> Dict:
    """Extract control parameters from configuration."""
    if 'control_params' in config:
        control_config = config['control_params']
        return {
            'effort_limit_nm': control_config.get('effort_limit_nm', 5.2),
            'velocity_limit_rad_s': control_config.get('velocity_limit_rad_s', 6.28),
            'isaac_sim_stiffness': control_config.get('isaac_sim_stiffness', 80.0),
            'isaac_sim_damping': control_config.get('isaac_sim_damping', 8.0),
            'p_coefficient': control_config.get('p_coefficient', 16),
            'i_coefficient': control_config.get('i_coefficient', 0),
            'd_coefficient': control_config.get('d_coefficient', 32)
        }
    
    # Default control parameters
    return {
        'effort_limit_nm': 5.2,
        'velocity_limit_rad_s': 6.28,
        'isaac_sim_stiffness': 80.0,
        'isaac_sim_damping': 8.0,
        'p_coefficient': 16,
        'i_coefficient': 0,
        'd_coefficient': 32
    }

def get_data_collection_params(config: Dict) -> Dict:
    """Extract data collection parameters from configuration."""
    if 'data_collection' in config:
        return config['data_collection']
    
    # Default data collection parameters
    return {
        'observation_features': [
            f"{joint}.pos" for joint in get_joint_names(config)
        ],
        'action_features': [
            f"{joint}.pos" for joint in get_joint_names(config)
        ],
        'default_fps': 30,
        'max_fps': 120
    }

def print_config_summary(config: Dict) -> None:
    """Print a summary of the loaded configuration."""
    print("\n📋 Configuration Summary:")
    print("-" * 50)
    
    joint_limits = get_joint_limits(config)
    for joint_name, limits in joint_limits.items():
        print(f"  {joint_name}: [{limits[0]:.3f}, {limits[1]:.3f}] rad")
    
    control_params = get_control_params(config)
    print(f"\n🎛️  Control Parameters:")
    print(f"  Effort: {control_params['effort_limit_nm']} N⋅m")
    print(f"  Velocity: {control_params['velocity_limit_rad_s']} rad/s")
    print(f"  Stiffness: {control_params['isaac_sim_stiffness']}")
    print(f"  Damping: {control_params['isaac_sim_damping']}")
    
    data_params = get_data_collection_params(config)
    print(f"\n📊 Data Collection:")
    print(f"  FPS: {data_params['default_fps']}")
    print(f"  Features: {len(data_params['observation_features'])} obs, {len(data_params['action_features'])} actions")

if __name__ == "__main__":
    # Test configuration loading
    print("🧪 Testing Configuration System")
    print("=" * 50)
    
    config = load_robot_config()
    print_config_summary(config)
    
    print(f"\n📁 Configuration loaded from:")
    if os.path.exists(DEFAULT_YAML_CONFIG):
        print(f"  ✅ YAML: {DEFAULT_YAML_CONFIG}")
    else:
        print(f"  📋 Env Config: {DEFAULT_ENV_CONFIG}")
        print(f"  ⚠️  YAML not found: {DEFAULT_YAML_CONFIG}") 