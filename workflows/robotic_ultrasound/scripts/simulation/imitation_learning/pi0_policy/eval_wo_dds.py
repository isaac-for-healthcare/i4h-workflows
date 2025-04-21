import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Run in headless mode
import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require GUI
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from policy_runner.runners import PI0PolicyRunner

def calc_mse_video_policy(
    policy_runner: PI0PolicyRunner,
    room_video_path: str,
    wrist_video_path: str,
    parquet_path: str = "/home/yunliu/.cache/huggingface/lerobot/i4h/robotic_ultrasound-cosmos-pi-tr/data/chunk-000/episode_000000.parquet",
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    step_stride: int = 1,
    img_size: Tuple[int, int] = (224, 224),
    plot: bool = True,
    save_plot_path: Optional[str] = "mse_comparison.png",
    display_plot: bool = False,
    verbose: bool = False
):
    """
    Calculate MSE for PI0 policy on video data and plot results
    
    Args:
        policy_runner: PI0PolicyRunner instance
        room_video_path: Path to room camera video
        wrist_video_path: Path to wrist camera video
        parquet_path: Path to parquet file containing ground truth actions and states
        frame_start: Starting frame index
        frame_end: Ending frame index, if None process until the end of video
        step_stride: Stride for frame processing
        img_size: Input image size
        plot: Whether to plot results
        save_plot_path: Path to save plot (if needed)
        display_plot: Whether to display the plot
        verbose: Whether to display detailed logs
    
    Returns:
        Dict: Dictionary containing MSE and various statistics
    """
    # Load videos with explicit codec preference
    def create_capture(path):
        # Try different backend options
        cap = cv2.VideoCapture(path)
        return cap

    room_cap = cv2.VideoCapture(room_video_path)
    print(room_cap.isOpened())
    print(room_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wrist_cap = cv2.VideoCapture(wrist_video_path)
    print(wrist_cap.isOpened())
    print(wrist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Check if videos were opened successfully
    if not room_cap.isOpened() or not wrist_cap.isOpened():
        raise ValueError("Cannot open video files")
    
    # Get frame count for videos
    room_frame_count = int(room_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wrist_frame_count = int(wrist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure frame counts match between videos
    if room_frame_count != wrist_frame_count:
        print(f"Warning: Video frame counts do not match - Room: {room_frame_count}, Wrist: {wrist_frame_count}")
        frame_count = min(room_frame_count, wrist_frame_count)
    else:
        frame_count = room_frame_count
    
    if verbose:
        print(f"Total video frames: {frame_count}")
    
    # Load ground truth actions and states from Parquet file
    if verbose:
        print(f"Loading ground truth data from {parquet_path}...")
    
    df = pd.read_parquet(parquet_path)
    
    # Check dataframe structure
    if verbose:
        print("Parquet file columns: ", df.columns.tolist())
    
    # Extract state and action data
    gt_states = df['observation.state'].values
    gt_actions = df['action'].values
    
    # Check if data length matches
    data_length = len(gt_states)
    if verbose:
        print(f"Data length: {data_length}")
    
    # If end frame not specified, use minimum length
    if frame_end is None:
        frame_end = min(frame_count, data_length)
    else:
        frame_end = min(frame_end, frame_count, data_length)
    
    # Lists to store results
    frame_indices = []
    pred_actions = []
    gt_actions_used = []
    
    # Process frames in specified range
    for frame_idx in range(frame_start, frame_end, step_stride):
        if verbose and frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{frame_end}")
        
        # Set video position and read frames
        room_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        wrist_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, room_frame = room_cap.read()
        ret2, wrist_frame = wrist_cap.read()
        
        if not ret1 or not ret2:
            print(f"Could not read frame {frame_idx}, skipping")
            continue
        
        # Convert to RGB
        room_frame = cv2.cvtColor(room_frame, cv2.COLOR_BGR2RGB)
        wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
        
        # Resize frames
        room_frame = cv2.resize(room_frame, img_size)
        wrist_frame = cv2.resize(wrist_frame, img_size)
        
        # Get current joint position
        current_state = gt_states[frame_idx]
        if hasattr(current_state, 'shape') and len(current_state.shape) > 0:
            if current_state.shape[0] > 7:
                current_state = current_state[:7]  # Assume using first 7 joints
        else:
            # If state is not an array, might need different handling
            print(f"Warning: State format unexpected: {type(current_state)}")
            continue
        
        # Run policy inference
        actions = policy_runner.infer(
            room_img=room_frame,
            wrist_img=wrist_frame,
            current_state=current_state
        )
        # Get ground truth action
        ground_truth = gt_actions[frame_idx]
        # Add actions to result lists
        frame_indices.append(frame_idx)
        pred_actions.append(actions[0])
        gt_actions_used.append(ground_truth)
    
    # Release video resources
    room_cap.release()
    wrist_cap.release()
    
    # Convert to numpy arrays
    frame_indices = np.array(frame_indices)
    pred_actions = np.array(pred_actions)
    gt_actions_used = np.array(gt_actions_used)
    
    # Calculate MSE
    action_mse = np.mean((pred_actions - gt_actions_used) ** 2)
    
    # Calculate per-joint MSE
    joint_mse = np.mean((pred_actions - gt_actions_used) ** 2, axis=0)
    
    if verbose:
        print(f"Overall action MSE: {action_mse}")
        print(f"Per-joint MSE: {joint_mse}")
    
    # Plot results
    if plot:
        # Determine number of joints/DOFs
        num_dofs = pred_actions.shape[1]
        
        # Create figure large enough
        fig, axes = plt.subplots(nrows=num_dofs, ncols=1, figsize=(12, 3*num_dofs))
        fig.suptitle(f"Action Trajectory Comparison - Total MSE: {action_mse:.6f}", fontsize=14)
        
        # Handle single DOF case
        if num_dofs == 1:
            axes = [axes]
        
        # Plot curves for each DOF
        for i, ax in enumerate(axes):
            ax.plot(frame_indices, gt_actions_used[:, i], 'b-', label='Ground Truth')
            ax.plot(frame_indices, pred_actions[:, i], 'r-', label='Predicted')
            
            # Mark points every 10 frames
            marker_stride = 10
            marker_indices = list(range(0, len(frame_indices), marker_stride))
            if marker_indices:
                ax.plot(frame_indices[marker_indices], gt_actions_used[marker_indices, i], 'bo', label='GT Markers')
                ax.plot(frame_indices[marker_indices], pred_actions[marker_indices, i], 'ro', label='Pred Markers')
            
            ax.set_title(f"DOF {i} - MSE: {joint_mse[i]:.6f}")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Action Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for title
        
        if save_plot_path:
            plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Plot saved to {save_plot_path}")
        
        if display_plot:
            plt.show()
        else:
            plt.close(fig)
        gt_position_x = np.cumsum(gt_actions_used[:, 0])
        pred_position_x = np.cumsum(pred_actions[:, 0])
        gt_position_y = np.cumsum(gt_actions_used[:, 1])
        pred_position_y = np.cumsum(pred_actions[:, 1])
        gt_position_z = np.cumsum(gt_actions_used[:, 2])
        pred_position_z = np.cumsum(pred_actions[:, 2])
        
        # plot the position in 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt_position_x, gt_position_y, gt_position_z, label='gt position')
        ax.plot(pred_position_x, pred_position_y, pred_position_z, label='pred position')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.savefig("./mse-pi0-te-3d.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    # Return results
    results = {
        "total_mse": action_mse,
        "joint_mse": joint_mse,
        "pred_actions": pred_actions,
        "gt_actions": gt_actions_used,
        "frame_indices": frame_indices
    }
    
    return results


# Usage example
if __name__ == "__main__":
    # Set matplotlib to non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets
    # ckpt_path = "/home/yunliu/Workspace/Code/i4h-workflows/workflows/robotic_ultrasound/scripts/training/pi_zero/checkpoints/robotic_ultrasound_lora/liver_ultrasound/5000"
    ckpt_path = robot_us_assets.policy_ckpt
    # repo_id = "i4h/robotic_ultrasound-cosmos-pi-te"
    repo_id = "i4h/sim_liver_scan"
    room_video_path = "/home/yunliu/.cache/huggingface/lerobot/i4h/robotic_ultrasound-cosmos-tr/videos/chunk-000/observation.images.room/episode_000002_new.mp4"
    wrist_video_path = "/home/yunliu/.cache/huggingface/lerobot/i4h/robotic_ultrasound-cosmos-tr/videos/chunk-000/observation.images.wrist/episode_000002_new.mp4"
    parquet_path = "/home/yunliu/.cache/huggingface/lerobot/i4h/robotic_ultrasound-cosmos-tr/data/chunk-000/episode_000002.parquet"
    
    # Initialize policy runner
    policy_runner = PI0PolicyRunner(
        ckpt_path=ckpt_path,
        repo_id=repo_id
    )
    
    # Calculate MSE and plot
    results = calc_mse_video_policy(
        policy_runner=policy_runner,
        room_video_path=room_video_path,
        wrist_video_path=wrist_video_path,
        parquet_path=parquet_path,
        frame_start=0,
        frame_end=300,
        step_stride=1,
        plot=True,
        save_plot_path="./mse_comparison_wo-cosmos-pi0.png",
        verbose=True
    )
    
    print(f"Overall MSE: {results['total_mse']}")
    print(f"Per-joint MSE: {results['joint_mse']}")