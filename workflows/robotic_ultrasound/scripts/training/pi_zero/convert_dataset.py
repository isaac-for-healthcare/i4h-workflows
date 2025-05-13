import os
import pandas as pd
import numpy as np
import io
from PIL import Image
import cv2
import shutil
from tqdm import tqdm
import argparse
import json

def create_directory_structure(output_dir):
    """Create the necessary directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos", "chunk-000", "observation.images.room"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos", "chunk-000", "observation.images.wrist"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

def create_info_json(output_dir, num_episodes):
    """Create the info.json file."""
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": num_episodes,
        "total_frames": 0,  # Will be updated later
        "total_tasks": 1,
        "total_videos": num_episodes * 2,  # 2 video types per episode (room and wrist)
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {
            "train": f"0:{num_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.room": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 224,
                    "video.width": 224,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 224,
                    "video.width": 224,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "names": ["state"]
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": ["actions"]
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            }
        }
    }
    
    with open(os.path.join(output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    
    return info

def create_episodes_jsonl(output_dir, episode_frame_counts):
    """Create episodes.jsonl file in JSONL format."""
    episodes_file_path = os.path.join(output_dir, "meta", "episodes.jsonl")
    
    with open(episodes_file_path, "w") as f:
        for episode_index, frame_count in episode_frame_counts.items():
            episode_data = {
                "episode_index": episode_index,
                "tasks": ["Perform a liver ultrasound."],
                "length": frame_count
            }
            f.write(json.dumps(episode_data) + "\n")

def create_tasks_jsonl(output_dir):
    """Create tasks.jsonl file in JSONL format."""
    tasks_file_path = os.path.join(output_dir, "meta", "tasks.jsonl")
    
    with open(tasks_file_path, "w") as f:
        task_data = {
            "task_index": 0,
            "task": "Perform a liver ultrasound."
        }
        f.write(json.dumps(task_data) + "\n")

def create_modality_json(output_dir):
    """Create modality.json file."""
    modality = {
        "state": {
            "panda_hand": {
                "start": 0,
                "end": 7,
                "rotation_type": "quaternion"
            }
        },
        "action": {
            "panda_hand": {
                "start": 0,
                "end": 6,
                "rotation_type": "axis_angle"
            }
        },
        "video": {
            "room": {
                "original_key": "observation.images.room"
            },
            "wrist": {
                "original_key": "observation.images.wrist"
            }
        },
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        }
    }
    
    with open(os.path.join(output_dir, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, indent=4)

def create_video_from_frames(frames, video_path, fps=30):
    """Create a video from a list of frames."""
    if not frames:
        print(f"Warning: No frames to create video at {video_path}")
        return
    
    height, width, channels = frames[0].shape
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Try direct MP4 creation first
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not video.isOpened():
            raise Exception("Failed to open video writer")
        
        for frame in frames:
            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        
        video.release()
        print(f"Created video: {video_path}")
        
        # Try to convert using ffmpeg for better compatibility
        try:
            import subprocess
            temp_path = video_path + ".temp.mp4"
            os.rename(video_path, temp_path)
            
            cmd = [
                'ffmpeg', '-y', '-i', temp_path, 
                '-c:v', 'libx264', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', video_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.remove(temp_path)
            # print(f"Converted video for better compatibility: {video_path}")
        except Exception as e:
            print(f"Warning: ffmpeg conversion failed: {e}. Using original mp4.")
            # If ffmpeg fails, restore the original file
            if os.path.exists(temp_path):
                os.rename(temp_path, video_path)
    
    except Exception as e:
        print(f"Error creating video with OpenCV: {e}")
        print("Trying to save individual frames as PNG files instead...")
        
        # Fall back to saving images if video creation fails
        frames_dir = os.path.join(os.path.dirname(video_path), f"{os.path.basename(video_path).split('.')[0]}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        print(f"Saved {len(frames)} frames to {frames_dir}")

def process_parquet_file(file_path, episode_index, output_dir):
    """Process a single parquet file and generate the corresponding data."""
    df = pd.read_parquet(file_path)
    
    frames = len(df)
    
    # Create lists to hold image frames for each video type
    room_frames = []
    wrist_frames = []
    
    # Create new DataFrame with the correct structure
    new_data = {
        "observation.state": [],
        "action": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": []
    }
    
    # Calculate proper timestamps based on FPS
    fps = 30.0
    frame_duration = 1.0 / fps
    
    for idx, row in df.iterrows():
        # Process room camera image
        if "image" in row:
            img_bytes = row["image"]["bytes"]
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            
            # Resize to 224x224 if needed
            if img.shape[0] != 224 or img.shape[1] != 224:
                img = cv2.resize(img, (224, 224))
            
            room_frames.append(img)
        
        # Process wrist camera image
        if "wrist_image" in row:
            img_bytes = row["wrist_image"]["bytes"]
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            
            # Resize to 224x224 if needed
            if img.shape[0] != 224 or img.shape[1] != 224:
                img = cv2.resize(img, (224, 224))
            
            wrist_frames.append(img)
        
        # Extract state and action data
        # Adjust based on your actual data structure
        state = np.zeros(7) if "state" not in row else np.array(row["state"])
        action = np.zeros(6) if "actions" not in row else np.array(row["actions"])
        
        # Store timestamp as a float scalar (not a numpy array)
        # This will be the time in seconds for this frame
        timestamp = row["timestamp"]
        
        new_data["observation.state"].append(state)
        new_data["action"].append(action)
        new_data["timestamp"].append(np.float32(timestamp))  # Use scalar float32 instead of array
        new_data["frame_index"].append(row["frame_index"])  # Use scalar int64 instead of array
        new_data["episode_index"].append(row["episode_index"])  # Use scalar int64 instead of array
        new_data["index"].append(row["index"])  # Use scalar int64 instead of array
        new_data["task_index"].append(row["task_index"])  # Use scalar int64 instead of array
    
    # Create videos with proper timestamp metadata
    video_fps = fps
    os.makedirs(os.path.join(output_dir, "videos", "chunk-000", "observation.images.room"), exist_ok=True)
    room_video_path = os.path.join(output_dir, "videos", "chunk-000", "observation.images.room", f"episode_{episode_index:06d}.mp4")
    create_video_from_frames(room_frames, room_video_path, video_fps)
    
    os.makedirs(os.path.join(output_dir, "videos", "chunk-000", "observation.images.wrist"), exist_ok=True)
    wrist_video_path = os.path.join(output_dir, "videos", "chunk-000", "observation.images.wrist", f"episode_{episode_index:06d}.mp4")
    create_video_from_frames(wrist_frames, wrist_video_path, video_fps)
    
    # Create new parquet with restructured data
    new_df = pd.DataFrame(new_data)
    os.makedirs(os.path.join(output_dir, "data", "chunk-000"), exist_ok=True)
    parquet_path = os.path.join(output_dir, "data", "chunk-000", f"episode_{episode_index:06d}.parquet")
    new_df.to_parquet(parquet_path)
    
    print(f"Generated episode {episode_index} with {frames} frames - {parquet_path}")
    
    return frames

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to robotic_ultrasound format")
    parser.add_argument("--input_dir", type=str, default="/home/yunliu/Downloads/sim_250_rel_axis_angle/data/chunk-000",
                        help="Input directory containing parquet files")
    parser.add_argument("--output_dir", type=str, default="/home/yunliu/.cache/huggingface/lerobot/i4h/robotic_ultrasound-n1-tr",
                        help="Output directory for the converted dataset")
    parser.add_argument("--num_files", type=int, default=0,
                        help="Number of files to process")
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure(args.output_dir)
    
    # Process parquet files
    total_frames = 0
    all_files = [f for f in os.listdir(args.input_dir) if f.endswith('.parquet')]
    
    # Limit the number of files processed if specified
    if args.num_files > 0 and args.num_files < len(all_files):
        print(f"Limiting to {args.num_files} files (out of {len(all_files)} available)")
        files = all_files[:args.num_files]
    else:
        print(f"Processing all {len(all_files)} files")
        files = all_files
    
    # Track frame counts per episode
    episode_frame_counts = {}
    
    for i, file in enumerate(tqdm(files)):
        file_path = os.path.join(args.input_dir, file)
        frames = process_parquet_file(file_path, i, args.output_dir)
        total_frames += frames
        episode_frame_counts[i] = frames
    
    # Create and update info.json
    info = create_info_json(args.output_dir, len(files))
    info["total_frames"] = total_frames
    
    with open(os.path.join(args.output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    
    # Create JSONL meta files
    create_episodes_jsonl(args.output_dir, episode_frame_counts)
    create_tasks_jsonl(args.output_dir)
    create_modality_json(args.output_dir)
    
    print(f"Done! Converted {len(files)} episodes with {total_frames} total frames.")

if __name__ == "__main__":
    main() 