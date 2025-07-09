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
import json
import math
import os
from typing import Dict, Tuple
from config_utils import update_env_config_file, DEFAULT_ENV_CONFIG

def encoder_counts_to_radians(min_count: int, max_count: int, 
                            resolution: int = 4096, full_rotation: float = 2 * math.pi) -> Tuple[float, float]:
    """
    Convert encoder counts to radians for SO-ARM 101 joints.
    
    Args:
        min_count: Minimum encoder count
        max_count: Maximum encoder count  
        resolution: Encoder resolution (default 4096 for 12-bit)
        full_rotation: Full rotation in radians (default 2π)
    
    Returns:
        Tuple of (min_radians, max_radians)
    """
    counts_per_radian = resolution / full_rotation
    
    # Convert to radians from center (typically 2048 for 12-bit encoder)
    center_count = resolution // 2
    
    min_radians = (min_count - center_count) / counts_per_radian
    max_radians = (max_count - center_count) / counts_per_radian
    
    return min_radians, max_radians

def analyze_calibration_file(calibration_file: str) -> Dict[str, Tuple[float, float]]:
    """
    Analyze calibration JSON file and extract joint limits.
    
    Args:
        calibration_file: Path to calibration JSON file
        
    Returns:
        Dictionary mapping joint names to (min_rad, max_rad) tuples
    """
    
    print(f"📁 Loading calibration file: {calibration_file}")
    
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)
    
    print(f"✅ Found calibration data for {len(calibration_data)} joints")
    
    # Analyze each joint
    joint_limits = {}
    
    print("\n🔍 Analyzing joint ranges:")
    print("-" * 60)
    
    for joint_name, joint_config in calibration_data.items():
        # Extract range information
        range_min = joint_config.get('range_min', 0)
        range_max = joint_config.get('range_max', 4096)
        id_num = joint_config.get('id', 0)
        homing_offset = joint_config.get('homing_offset', 0)
        
        # Convert to radians
        min_rad, max_rad = encoder_counts_to_radians(range_min, range_max)
        
        # Store limits
        joint_limits[joint_name] = (min_rad, max_rad)
        
        # Calculate range information
        range_counts = range_max - range_min
        range_radians = max_rad - min_rad
        range_degrees = range_radians * 180 / math.pi
        
        print(f"🤖 {joint_name}:")
        print(f"   ID: {id_num}, Homing: {homing_offset}")
        print(f"   Encoder: {range_min} → {range_max} ({range_counts} counts)")
        print(f"   Radians: {min_rad:.3f} → {max_rad:.3f} ({range_radians:.3f} rad)")
        print(f"   Degrees: {min_rad*180/math.pi:.1f}° → {max_rad*180/math.pi:.1f}° ({range_degrees:.1f}°)")
        print()
    
    return joint_limits

def print_comparison_table(joint_limits: Dict[str, Tuple[float, float]]):
    """Print a formatted comparison table of joint limits."""
    
    print("📊 Joint Limits Summary:")
    print("=" * 80)
    print(f"{'Joint':<15} {'Range (rad)':<20} {'Range (deg)':<20} {'Total Range':<15}")
    print("-" * 80)
    
    for joint_name, (min_rad, max_rad) in joint_limits.items():
        range_rad = max_rad - min_rad
        range_deg = range_rad * 180 / math.pi
        
        rad_str = f"±{max(abs(min_rad), abs(max_rad)):.3f}"
        deg_str = f"±{max(abs(min_rad), abs(max_rad))*180/math.pi:.1f}°"
        total_str = f"{range_deg:.1f}°"
        
        print(f"{joint_name:<15} {rad_str:<20} {deg_str:<20} {total_str:<15}")

def main():
    parser = argparse.ArgumentParser(description="Analyze real robot calibration data and update robot configuration")
    parser.add_argument(
        "--calibration_file", 
        type=str, 
        default="../config/my_awesome_follower_arm.json",
        help="Path to the calibration JSON file"
    )
    parser.add_argument(
        "--env_config", 
        type=str, 
        default=DEFAULT_ENV_CONFIG,
        help="Path to the robot configuration file"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    
    args = parser.parse_args()
    
    # Check if calibration file exists
    if not os.path.exists(args.calibration_file):
        print(f"❌ Calibration file not found: {args.calibration_file}")
        print("Please make sure the file exists or provide the correct path.")
        return 1
    
    print("🤖 SO-ARM 101 Real Calibration Data Analyzer")
    print("=" * 60)
    
    # Analyze calibration data
    try:
        joint_limits = analyze_calibration_file(args.calibration_file)
    except Exception as e:
        print(f"❌ Error analyzing calibration file: {e}")
        return 1
    
    # Print summary
    print_comparison_table(joint_limits)
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified")
        print("-" * 60)
        print("The following changes would be made:")
        print(f"📁 Robot config: {args.env_config}")
        
        for joint_name, (min_rad, max_rad) in joint_limits.items():
            middle = (min_rad + max_rad) / 2.0
            print(f"  🎯 {joint_name}: safe initial position = {middle:.3f} rad")
            print(f"     (calibrated range: {min_rad:.3f} to {max_rad:.3f})")
        
        print("\n📝 Note: Robot config only contains initial joint positions")
        print("📝 Joint limits should be enforced in your environment/MDP configuration")
        print("\nTo apply these changes, run without --dry_run flag.")
        return 0
    
    # Update robot configuration
    print(f"\n🔄 Updating robot configuration...")
    print("-" * 60)
    
    success = update_env_config_file(joint_limits, args.env_config)
    
    if success:
        print(f"\n✅ Robot configuration updated successfully!")
        print(f"📁 Updated file: {args.env_config}")
        
        print("\n🎯 Next steps:")
        print("1. Create your environment/MDP configuration with proper joint limits")
        print("2. Run validation: python validate_sim_environment.py --save_report")
        print("3. Collect data: python collect_sim_data.py --output_dir ./data")
        print("4. Train policies with your RL framework")
        
        return 0
    else:
        print(f"\n❌ Failed to update robot configuration")
        print("Please check the error messages above and try again.")
        return 1

if __name__ == "__main__":
    exit(main()) 