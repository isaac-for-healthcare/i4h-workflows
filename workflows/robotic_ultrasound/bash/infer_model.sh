# Function to cleanup processes on exit
cleanup() {
    echo "Cleaning up processes..."
    # Kill the entire process group for both processes
    kill -9 $(ps -o pgid= -p $policy_pid) 2>/dev/null
    kill -9 $(ps -o pgid= -p $sim_pid) 2>/dev/null
    exit
}

# Set up trap for SIGINT (Ctrl+C)
trap cleanup SIGINT

# Start the policy runner in the background and capture its output
python -m policy_runner.run_policy 2>&1 | tee policy_output.log &
policy_pid=$!

# Wait for 5 seconds
sleep 5

# Run the simulation with cameras and capture its output
python -m simulation.examples.sim_with_dds --enable_cameras 2>&1 | tee simulation_output.log &
sim_pid=$!

# Wait for keyboard interrupt
echo "Press Ctrl+C to stop all processes"
wait