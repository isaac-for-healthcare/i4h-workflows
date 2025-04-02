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

    # Kill both processes
    kill_process $policy_pid
    kill_process $sim_pid
    
    exit
}

# Set up trap for SIGINT (Ctrl+C)
trap cleanup SIGINT


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

        # Function to update sysctl if new value is higher
        update_if_higher() {
            local param=$1
            local current=$2
            local new=$3
            if [ "$current" -lt "$new" ]; then
                if [ "$VERBOSE" = true ]; then
                    echo "Increasing $param from $current to $new"
                fi
                run_cmd "sudo sysctl -w $param=$new"
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
