import h5py
import numpy as np
import cv2
from PIL import Image
import glob
import pdb
import tqdm
import os
# Load HDF5 file


def process_data(file_path, output_path):
    f = h5py.File(file_path, "r")
    # Extract data
    rgb_images = f['data/demo_0']['observations']['rgb_images']  # (263, 2, 224, 224, 3)
    depth_images = f['data/demo_0']['observations']['depth_images']  # (263, 2, 224, 224, 1)
    seg_images = f['data/demo_0']['observations']['seg_images']  # (263, 2, 224, 224, 4) - Only last channel needed

    # Video parameters
    num_frames, num_videos, height, width, _ = seg_images.shape
    fps = 30  # Frames per second
    video_code = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format

    # Create video writers
    writers = {
        "rgb": [cv2.VideoWriter(f"{output_path}/rgb_video_{i}.mp4", video_code, fps, (width, height)) for i in range(num_videos)],
        "depth": [cv2.VideoWriter(f"{output_path}/depth_video_{i}.mp4", video_code, fps, (width, height), isColor=False) for i in range(num_videos)],
        "seg": [cv2.VideoWriter(f"{output_path}/seg_mask_video_{i}.mp4", video_code, fps, (width, height), isColor=True) for i in range(num_videos)]
    }

    # Process frames
    npz_savers = {}
    for i in range(num_frames):
        for vid_idx in range(num_videos):
            # RGB Video
            rgb_frame = rgb_images[i, vid_idx].astype(np.uint8)[:,:,::-1]  # (224, 224, 3)
            writers["rgb"][vid_idx].write(rgb_frame)

            # Depth Video (Normalize and Apply Colormap)
            depth_frame = depth_images[i, vid_idx, :, :, 0]  # Extract (224, 224)
            output = 1.0 / (depth_frame + 1e-6)

            depth_min = output.min()
            depth_max = output.max()
            max_val = (2**8) - 1  # Maximum value for uint16

            if depth_max - depth_min > np.finfo("float").eps:
                out_array = max_val * (output - depth_min) / (depth_max - depth_min)
            else:
                out_array = np.zeros_like(output)

            formatted = out_array.astype("uint8")
            writers["depth"][vid_idx].write(formatted)


            # max_depth = np.nanmax(depth_frame)  # Get max valid depth ignoring NaN
            # min_depth = np.nanmin(depth_frame)  # Get min valid depth ignoring NaN
            # depth_norm = (depth_frame - min_depth) / (max_depth - min_depth) * 255
            
            # depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)  # Apply colormap
            # writers["depth"][vid_idx].write(depth_colored)

            # Segmentation Mask Video (Using Last Channel)
            
            seg_mask_frame = seg_images[i, vid_idx, :, :, 0].astype(np.uint8) # Already 0 or 255

            # Define a colormap (Each label gets an (R, G, B) color)
            color_map = {
                0: (0, 0, 0),        # Black
                1: (255, 0, 0),      # Red
                2: (0, 255, 0),      # Green
                3: (0, 0, 255),      # Blue
                4: (255, 255, 0)     # Yellow
            }
            # Create an RGB image (H, W, 3)
            H, W = seg_mask_frame.shape
            seg_mask_colored = np.zeros((H,W,3), dtype=np.uint8)
            if vid_idx not in npz_savers.keys():
                npz_savers[vid_idx] = []
            npz_savers[vid_idx].append(seg_mask_frame)
            for label, color in color_map.items():
                seg_mask_colored[seg_mask_frame == label] = np.array(color, dtype=np.uint8)  # Assign color based on label
            writers["seg"][vid_idx].write(seg_mask_colored)

    for i in npz_savers.keys():
        np.savez(f"{output_path}/seg_mask_video_{i}.npz", np.array(npz_savers[i]))

    # Release all video writers
    for category in writers:
        for writer in writers[category]:
            writer.release()

    print("Videos saved successfully!")

if __name__ == '__main__':
    exps_name = "2025-04-11-18-50-Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0/"
    folder = "data/hdf5/" + exps_name
    output_folder = "output/" + exps_name
    files = glob.glob(folder + '*.hdf5')
    for f in tqdm.tqdm(files):
        basename = os.path.basename(f).split('.')[0]
        output = output_folder + basename
        if os.path.exists(output):
            continue
        os.makedirs(output, exist_ok=True)
        process_data(f, output)


    