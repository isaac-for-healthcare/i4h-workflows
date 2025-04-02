#!/bin/bash

# Source the utility functions
source "$(dirname "$0")/utils.sh"

# Setup the cleanup trap
setup_cleanup_trap

# Increase buffer sizes
increase_buffer_sizes

# Check required environment variables
check_env_vars || exit 1

echo "Starting all processes..."
echo "=================================="

# Start the policy runner in the background and capture its output
echo "Starting policy runner..."
python -m policy_runner.run_policy 2>&1 | tee policy_output.log &
policy_pid=$!
add_pid_to_cleanup $policy_pid
echo "Policy runner started with PID $policy_pid"

# Wait for 10 seconds to allow initial process to start
echo "Waiting 10 seconds for initial process to start..."
sleep 10

# Run the simulation with cameras and capture its output
echo "Starting simulation..."
python -m simulation.examples.sim_with_dds --enable_cameras 2>&1 | tee simulation_output.log &
sim_pid=$!
add_pid_to_cleanup $sim_pid
echo "Simulation started with PID $sim_pid"

echo "=================================="
echo "All processes started successfully!"
echo "Press Ctrl+C to stop all processes"

# Wait for keyboard interrupt
wait
