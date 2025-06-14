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

# Initialize verbose mode flag
VERBOSE=false

# Parse command line arguments
while getopts "v" opt; do
    case $opt in
        v) VERBOSE=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Function to run commands with optional verbose output
run_cmd() {
    if [ "$VERBOSE" = true ]; then
        echo "Running command: $1" >&2
    fi
    eval "$1"
}

# Function to get numeric value from sysctl output
get_sysctl_value() {
    run_cmd "sysctl -n $1"
}

# Print header
echo "UDP Socket Buffer Size Information"
echo "=================================="

# Print UDP send buffer sizes
echo -e "\nUDP SEND BUFFER SIZES:"
echo "-------------------------"
echo "Default send buffer size (bytes):"
run_cmd "sysctl net.core.wmem_default"

echo -e "\nMaximum send buffer size (bytes):"
run_cmd "sysctl net.core.wmem_max"

# Print UDP receive buffer sizes
echo -e "\nUDP RECEIVE BUFFER SIZES:"
echo "---------------------------"
echo "Default receive buffer size (bytes):"
run_cmd "sysctl net.core.rmem_default"

echo -e "\nMaximum receive buffer size (bytes):"
run_cmd "sysctl net.core.rmem_max"
echo "=================================="

# Store current values
current_rmem_default=$(get_sysctl_value net.core.rmem_default)
current_wmem_default=$(get_sysctl_value net.core.wmem_default)
current_rmem_max=$(get_sysctl_value net.core.rmem_max)
current_wmem_max=$(get_sysctl_value net.core.wmem_max)

# Desired values
new_rmem_default=8500000
new_wmem_default=8500000
new_rmem_max=8500000
new_wmem_max=8500000

if [[ $current_rmem_default -lt $new_rmem_default || $current_wmem_default -lt $new_wmem_default || $current_rmem_max -lt $new_rmem_max || $current_wmem_max -lt $new_wmem_max ]]; then

    # Prompt for buffer size increase
    echo -e "\nConnext recommends larger values for these settings to improve performance.\n
    Would you like to increase your send/receive socket buffer sizes? Default will be increased to 8500000, and Maximum to 8500000. (y/n)"
    read answer

    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        echo "Checking current values and increasing buffer sizes if needed..."

        # Function to update sysctl if new value is higher
        update_if_higher() {
            local param=$1
            local current=$2
            local new=$3
            if [ "$current" -lt "$new" ]; then
                if [ "$VERBOSE" = true ]; then
                    echo "Increasing $param from $current to $new"
                fi
                run_cmd "sysctl -w $param=$new"
            else
                echo "Keeping $param at current value ($current) as it's already higher than suggested value ($new)"
            fi
        }

        # Update each parameter only if new value is higher
        update_if_higher "net.core.rmem_default" "$current_rmem_default" "$new_rmem_default"
        update_if_higher "net.core.wmem_default" "$current_wmem_default" "$new_wmem_default"
        update_if_higher "net.core.rmem_max" "$current_rmem_max" "$new_rmem_max"
        update_if_higher "net.core.wmem_max" "$current_wmem_max" "$new_wmem_max"

        echo -e "\nTo make these changes permanent, update /etc/sysctl.conf:"
    else
        echo "Buffer sizes left unchanged."
    fi
else
    echo "All current buffer sizes already meet or exceed the recommended values."
fi
