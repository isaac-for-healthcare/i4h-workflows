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

# Start the raysim in the background and capture its output
echo "Starting raysim..."
# note: too many logs, redirect to file instead of stdout
python -m simulation.examples.ultrasound-raytracing > raysim_output.log 2>&1 &
raysim_pid=$!
add_pid_to_cleanup $raysim_pid
echo "Raysim started with PID $raysim_pid"

# Wait for 10 seconds to allow initial processes to start
echo "Waiting 10 seconds for initial processes to start..."
sleep 10

# Run the simulation with cameras and capture its output
echo "Starting simulation..."
python -m simulation.examples.sim_with_dds --enable_cameras 2>&1 | tee simulation_output.log &
sim_pid=$!
add_pid_to_cleanup $sim_pid
echo "Simulation started with PID $sim_pid"

# Wait for 10 seconds to allow initial processes to start
echo "Waiting 20 seconds for the UI to load ..."
sleep 20

# Run the visualization application
echo "Starting visualization..."
python -m utils.visualization 2>&1 | tee visualization_output.log &
visualization_pid=$!
add_pid_to_cleanup $visualization_pid
echo "Visualization started with PID $visualization_pid"


echo "=================================="
echo "All processes started successfully!"
echo "Press Ctrl+C to stop all processes"

# Wait for keyboard interrupt
wait
