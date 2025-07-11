#!/bin/bash

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

# Script to kill all robotic ultrasound workflow processes and their children

echo "Killing robotic ultrasound workflow processes..."

# Function to kill process and its children
kill_process_tree() {
    local pattern="$1"
    local name="$2"

    echo "Checking for processes matching: $pattern"

    # Get PIDs of matching processes
    pids=$(pgrep -f "$pattern")

    if [ ! -z "$pids" ]; then
        echo "Found PIDs: $pids for $name"

        for pid in $pids; do
            echo "Killing process tree for PID $pid ($name)"

            # Get all child processes
            children=$(pgrep -P $pid 2>/dev/null)
            if [ ! -z "$children" ]; then
                echo "  Found child processes: $children"
                # Kill children first
                for child in $children; do
                    echo "  Killing child process $child"
                    kill -TERM $child 2>/dev/null
                done
            fi

            # Kill the parent process
            echo "  Killing parent process $pid"
            kill -TERM $pid 2>/dev/null
        done

        # Wait for graceful shutdown
        sleep 2

        # Force kill any remaining processes and children
        for pid in $pids; do
            if kill -0 $pid 2>/dev/null; then
                echo "Force killing process tree for PID $pid"
                children=$(pgrep -P $pid 2>/dev/null)
                if [ ! -z "$children" ]; then
                    for child in $children; do
                        kill -9 $child 2>/dev/null
                    done
                fi
                kill -9 $pid 2>/dev/null
            fi
        done

        echo "Killed $name processes"
    else
        echo "No processes found for $name"
    fi
}

# Kill specific processes and their children
kill_process_tree "policy_runner.run_policy" "policy runner"
kill_process_tree "simulation.environments.sim_with_dds" "simulation with DDS"
kill_process_tree "simulation.examples.ultrasound_raytracing" "ultrasound raytracing"
kill_process_tree "utils.visualization" "visualization"
kill_process_tree "isaac-sim" "Isaac Sim"

# Double-check for any remaining processes
echo ""
echo "Double-checking for any remaining processes..."
remaining=$(ps aux | grep -E "(policy_runner.run_policy|simulation.environments.sim_with_dds|simulation.examples.ultrasound_raytracing|utils.visualization|isaac-sim)" | grep -v grep | grep -v "kill_all_processes.sh")

if [ ! -z "$remaining" ]; then
    echo "Found remaining processes:"
    echo "$remaining"
    echo "Force killing any stragglers..."
    pkill -9 -f "policy_runner.run_policy" 2>/dev/null
    pkill -9 -f "simulation.environments.sim_with_dds" 2>/dev/null
    pkill -9 -f "simulation.examples.ultrasound_raytracing" 2>/dev/null
    pkill -9 -f "utils.visualization" 2>/dev/null
    pkill -9 -f "isaac-sim" 2>/dev/null
else
    echo "No remaining processes found."
fi

echo ""
echo "All processes and their children should be terminated now."
