#!/bin/bash

# Array to store PIDs that need cleanup
declare -a PIDS_TO_CLEANUP=()

# Function to add a PID to the cleanup list
add_pid_to_cleanup() {
    PIDS_TO_CLEANUP+=("$1")
}

# Function to setup cleanup trap
setup_cleanup_trap() {
    trap 'cleanup "${PIDS_TO_CLEANUP[@]}"' SIGINT
}

# Function to cleanup processes on exit
cleanup() {
    echo "Cleaning up processes..."
    
    # Function to kill a process and its children
    kill_process() {
        local pid=$1
        if [ -n "$pid" ]; then
            # First try graceful termination
            kill -TERM $pid 2>/dev/null
            
            # Wait for 2 seconds
            sleep 2
            
            # Check if process is still running
            if ps -p $pid > /dev/null; then
                echo "Process $pid did not terminate gracefully, forcing kill..."
                # Kill the entire process group
                pkill -P $pid 2>/dev/null
                kill -9 $pid 2>/dev/null
            fi
        fi
    }

    # Kill all provided PIDs
    for pid in "$@"; do
        kill_process $pid
        
        # Minus 1 from the PID to get the parent PID
        parent_pid=$((pid - 1))
        
        # Kill the parent process if it exists
        if ps -p $parent_pid > /dev/null; then
            kill -9 $parent_pid 2>/dev/null
        fi
    done
    
    exit
}

# Function to increase buffer sizes if needed
increase_buffer_sizes() {
    # Print header
    echo "UDP Socket Buffer Size Information"
    echo "=================================="

    # Print UDP send buffer sizes
    echo -e "\nUDP SEND BUFFER SIZES:"
    echo "-------------------------"
    echo "Default send buffer size (bytes):"
    sysctl net.core.wmem_default

    echo -e "\nMaximum send buffer size (bytes):"
    sysctl net.core.wmem_max

    # Print UDP receive buffer sizes
    echo -e "\nUDP RECEIVE BUFFER SIZES:"
    echo "---------------------------"
    echo "Default receive buffer size (bytes):"
    sysctl net.core.rmem_default

    echo -e "\nMaximum receive buffer size (bytes):"
    sysctl net.core.rmem_max
    echo "=================================="

    # Store current values
    current_rmem_default=$(sysctl -n net.core.rmem_default)
    current_wmem_default=$(sysctl -n net.core.wmem_default)
    current_rmem_max=$(sysctl -n net.core.rmem_max)
    current_wmem_max=$(sysctl -n net.core.wmem_max)

    # Desired values
    new_rmem_default=212992
    new_wmem_default=212992
    new_rmem_max=4194304
    new_wmem_max=4194304

    if [[ $current_rmem_default -lt $new_rmem_default || $current_wmem_default -lt $new_wmem_default || $current_rmem_max -lt $new_rmem_max || $current_wmem_max -lt $new_wmem_max ]]; then
        # Prompt for buffer size increase
        echo -e "\nConnext recommends larger values for these settings to improve performance.\nWould you like to increase your send/receive socket buffer sizes? Default will be increased to 212992, and Maximum to 4194304. (y/n)"
        read answer

        if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
            echo "Checking current values and increasing buffer sizes if needed..."

            # Update each parameter only if new value is higher
            if [ "$current_rmem_default" -lt "$new_rmem_default" ]; then
                sudo sysctl -w net.core.rmem_default=$new_rmem_default
            fi
            if [ "$current_wmem_default" -lt "$new_wmem_default" ]; then
                sudo sysctl -w net.core.wmem_default=$new_wmem_default
            fi
            if [ "$current_rmem_max" -lt "$new_rmem_max" ]; then
                sudo sysctl -w net.core.rmem_max=$new_rmem_max
            fi
            if [ "$current_wmem_max" -lt "$new_wmem_max" ]; then
                sudo sysctl -w net.core.wmem_max=$new_wmem_max
            fi

            echo -e "\nTo make these changes permanent, update /etc/sysctl.conf:"
        else
            echo "Buffer sizes left unchanged."
        fi
    else
        echo "All current buffer sizes already meet or exceed the recommended values."
    fi
}

# Function to check required environment variables
check_env_vars() {
    echo "Checking required environment variables..."
    echo "=================================="

    # Check RTI_LICENSE_FILE
    if [ -z "$RTI_LICENSE_FILE" ]; then
        echo "ERROR: RTI_LICENSE_FILE environment variable is not set!"
        echo "Please set RTI_LICENSE_FILE to point to your RTI license file location."
        return 1
    else
        echo "RTI_LICENSE_FILE is set to: $RTI_LICENSE_FILE"
    fi

    # Check PYTHONPATH
    if [ -z "$PYTHONPATH" ]; then
        echo "ERROR: PYTHONPATH environment variable is not set!"
        echo "Please set PYTHONPATH to include the workflows/robotic_ultrasound/scripts path."
        return 1
    else
        echo "PYTHONPATH is set to: $PYTHONPATH"
    fi

    echo "=================================="
    return 0
}
