# Shree KRISHNAya Namaha
# Warps frame using pose info with PyTorch GPU acceleration
# Author: Nagabhushan S N
# Modified for GPU support

import datetime
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import skimage.io
import torch
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R
from io import BytesIO
import imageio
from imageio import imwrite
from tqdm import tqdm
import cv2


def save_video(video, fps, H, W, video_save_path, video_save_quality=5):
    """Save video frames to file.

    Args:
        grid (np.ndarray): Video frames array [T,H,W,C]
        fps (int): Frames per second
        H (int): Frame height
        W (int): Frame width
        video_save_quality (int): Video encoding quality (0-10)
        video_save_path (str): Output video file path
    """
    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{W}x{H}"],
        "output_params": ["-f", "mp4"],
    }
    imageio.mimsave(video_save_path, video, "mp4", **kwargs)


def load_from_fileobj(filepath: str, format: str = "mp4", mode: str = "rgb", **kwargs):
    """
    Load video from a file-like object using imageio with specified format and color mode.

    Parameters:
        file (IO[bytes]): A file-like object containing video data.
        format (str): Format of the video file (default 'mp4').
        mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

    Returns:
        tuple: A tuple containing an array of video frames and metadata about the video.
    """
    with open(filepath, "rb") as f:
        value = f.read()
    with BytesIO(value) as f:
        f.seek(0)
        video_reader = imageio.get_reader(f, format, **kwargs)

        video_frames = []
        for frame in video_reader:
            if mode == "gray":
                import cv2  # Convert frame to grayscale if mode is gray

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.expand_dims(frame, axis=2)  # Keep frame dimensions consistent
            video_frames.append(frame)

    return np.array(video_frames), video_reader.get_meta_data()


def read_video_or_image_into_frames(
    input_path: str,
    input_path_format: str = "mp4",
    also_return_fps: bool = True,
):
    """Read video or image file and convert to tensor format.

    Args:
        input_path (str): Path to input video/image file
        input_path_format (str): Format of input file (default: "mp4")
        also_return_fps (bool): Whether to return fps along with frames

    Returns:
        numpy.ndarray | tuple: Video tensor in shape [T,H,W,C], optionally with fps if requested
    """
    loaded_data = load_from_fileobj(input_path, format=input_path_format)
    frames, meta_data = loaded_data
    if input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".jpeg"):
        frames = np.array(frames[0])  # HWC, [0,255]
        if frames.shape[-1] > 3:  # RGBA, set the transparent to white
            # Separate the RGB and Alpha channels
            rgb_channels = frames[..., :3]
            alpha_channel = frames[..., 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Create a white background
            white_bg = np.ones_like(rgb_channels) * 255  # White background in RGB

            # Blend the RGB channels with the white background based on the alpha channel
            frames = (rgb_channels * alpha_channel[..., None] + white_bg * (1 - alpha_channel[..., None])).astype(
                np.uint8
            )
        frames = [frames]
        fps = 0
    else:
        fps = int(meta_data.get("fps"))

    input_tensor = np.stack(frames, axis=0)
    if also_return_fps:
        return input_tensor, fps
    return input_tensor


def build_extrinsic_from_ros_pose(pos_w, quat_w_ros):
    """
    Args:
      pos_w       : (3,) array, camera position in world frame.
      quat_w_ros  : (4,) array, ROS quaternion (w, x, y, z).

    Returns:
      E : (4,4) array, world→camera extrinsic matrix.
    """
    # 1) Reorder ROS quat to SciPy's (x,y,z,w)
    qx, qy, qz, qw = quat_w_ros[1], quat_w_ros[2], quat_w_ros[3], quat_w_ros[0]
    # 2) Build camera→world rotation
    R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()  # 3×3
    
    # 3) Invert to world→camera
    R_cw = R_wc.T
    t_w = np.array(pos_w).reshape(3,1)
    t_cw = -R_cw @ t_w                               # (3×1)
    
    # 4) Assemble homogeneous matrix
    E = np.eye(4)
    E[:3, :3] = R_cw
    E[:3,  3] = t_cw.flatten()
    return E 

class Warper:
    def __init__(self, resolution: tuple = None, device: str = 'gpu0', splat_radius: float = 1.5):
        self.resolution = resolution
        self.device = self.get_device(device)
        self.splat_radius = splat_radius
        return

    def forward_warp(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                     transformation1: torch.Tensor, transformation2: torch.Tensor, intrinsic1: torch.Tensor, 
                     intrinsic2: Optional[torch.Tensor], splat_radius: Optional[float] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        :param splat_radius: Controls how far each point spreads. Larger values create denser but more blurred results. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        # Use the provided splat_radius or fall back to the class default
        _splat_radius = splat_radius if splat_radius is not None else self.splat_radius

        # Print shapes for debugging
        # print(f"frame1: {frame1.shape}, mask1: {mask1.shape}, depth1: {depth1.shape}")
        # print(f"transformation1: {transformation1.shape}, transformation2: {transformation2.shape}")
        # print(f"intrinsic1: {intrinsic1.shape}, intrinsic2: {intrinsic2.shape}")

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device)
        mask1 = mask1.to(self.device)
        depth1 = depth1.to(self.device)
        transformation1 = transformation1.to(self.device)
        transformation2 = transformation2.to(self.device)
        intrinsic1 = intrinsic1.to(self.device)
        intrinsic2 = intrinsic2.to(self.device)

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                        intrinsic2)
        
        # print(f"trans_points1: {trans_points1.shape}")
        
        # Fix the dimension issue - ensure that trans_coordinates and grid have compatible dimensions
        trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]  # (b, h*w, 2)
        trans_coordinates = trans_coordinates.reshape(b, h, w, 2).permute(0, 3, 1, 2)  # (b, 2, h, w)
        trans_depth1 = trans_points1[:, :, 2, 0].reshape(b, h, w)
        
        # print(f"trans_coordinates: {trans_coordinates.shape}, trans_depth1: {trans_depth1.shape}")

        grid = self.create_grid(b, h, w).to(trans_coordinates.device)
        # print(f"grid: {grid.shape}")
        
        flow12 = trans_coordinates - grid
        # print(f"flow12: {flow12.shape}")

        # Ensure trans_depth1 has the right shape for bilinear_splatting (b, 1, h, w)
        trans_depth1_reshaped = trans_depth1.unsqueeze(1)  # Add channel dimension -> (b, 1, h, w)
        # print(f"trans_depth1_reshaped: {trans_depth1_reshaped.shape}")
        
        # Warp the RGB frame
        # print("Warping RGB frame...")
        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1_reshaped, flow12, None, 
                                                      is_image=True, splat_radius=_splat_radius)
        # print(f"warped_frame2: {warped_frame2.shape}, mask2: {mask2.shape}")
        
        # # For depth warping, we need to restructure the depth tensor to have 3 channels like an image
        # # Create a dummy 3-channel tensor with the depth in each channel
        # dummy_depth_frame = torch.zeros(size=(b, 3, h, w), dtype=torch.float32).to(self.device)
        # for i in range(3):  # Fill all 3 channels with the same depth values
        #     dummy_depth_frame[:, i] = trans_depth1
        
        # # print(f"dummy_depth_frame: {dummy_depth_frame.shape}")
        
        # # Warp the depth frame
        # # print("Warping depth frame...")
        # try:
        #     warped_depth_full, depth_mask = self.bilinear_splatting(dummy_depth_frame, mask1, trans_depth1_reshaped, flow12, None,
        #                                              is_image=False, splat_radius=_splat_radius)
        #     # Extract just the first channel as our warped depth
        #     warped_depth2 = warped_depth_full[:, 0]
        #     # print(f"warped_depth2: {warped_depth2.shape}")
        # except Exception as e:
        #     print(f"Error warping depth: {e}")
        #     # Fallback: just return the original depth
        #     warped_depth2 = depth1.squeeze(1)
        
        # print("Forward warp complete")
        return warped_frame2, mask2 #, warped_depth2, flow12

    def compute_transformed_points(self, depth1: torch.Tensor, transformation1: torch.Tensor, transformation2: torch.Tensor,
                                   intrinsic1: torch.Tensor, intrinsic2: Optional[torch.Tensor]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)

        # Create meshgrid for pixel coordinates
        device = depth1.device
        y_range = torch.arange(0, h, device=device)
        x_range = torch.arange(0, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # Reshape grid to [b, h*w, 3, 1] for batch matrix multiplication
        x_grid = x_grid.reshape(-1).to(device)
        y_grid = y_grid.reshape(-1).to(device)
        ones = torch.ones_like(x_grid).to(device)
        pixel_coords = torch.stack((x_grid, y_grid, ones), dim=0).float()  # [3, h*w]
        pixel_coords = pixel_coords.unsqueeze(0).repeat(b, 1, 1)  # [b, 3, h*w]
        
        # Compute world coordinates
        # 1. Unproject to camera coordinates using inverse intrinsics
        unprojected = torch.bmm(torch.inverse(intrinsic1), pixel_coords)  # [b, 3, h*w]
        
        # 2. Multiply by depth to get points in camera coordinate system
        depth_reshaped = depth1.reshape(b, 1, h*w)
        cam_points = unprojected * depth_reshaped
        
        # 3. Convert to homogeneous coordinates
        ones_hw = torch.ones(b, 1, h*w, device=device)
        cam_points_homo = torch.cat([cam_points, ones_hw], dim=1)  # [b, 4, h*w]
        
        # 4. Transform points using the transformation matrix
        transformed_points = torch.bmm(transformation, cam_points_homo)  # [b, 4, h*w]
        
        # 5. Project back to image coordinates using the target intrinsics
        transformed_3d = transformed_points[:, :3, :]  # [b, 3, h*w]
        projected_points = torch.bmm(intrinsic2, transformed_3d)  # [b, 3, h*w]
        
        # Reshape to match expected format [b, h*w, 3, 1]
        projected_points = projected_points.permute(0, 2, 1).unsqueeze(-1)  # [b, h*w, 3, 1]
        
        return projected_points

    def bilinear_splatting(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                           flow12: torch.Tensor, flow12_mask: Optional[torch.Tensor], is_image: bool = False,
                           splat_radius: Optional[float] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting - Heavily optimized PyTorch GPU implementation
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :param splat_radius: Controls how far each point spreads. Larger values create denser but more blurred results. Optional
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        device = frame1.device
        dtype = frame1.dtype
        
        # Fast-path: If splat_radius <= 1, we can use grid_sample which is much faster
        if (splat_radius is None or splat_radius <= 1.0) and getattr(self, 'splat_radius', 1.0) <= 1.0:
            # For simple warping without splatting, use the faster grid_sample approach
            return self._fast_warp_with_grid_sample(frame1, mask1, depth1, flow12, flow12_mask, is_image)
        
        # Handle optional inputs
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w), device=device)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w), device=device)
            
        # Use the provided splat_radius or fall back to the class default
        _splat_radius = splat_radius if splat_radius is not None else self.splat_radius
              
        # Create grid on device directly
        grid = self.create_grid(b, h, w).to(device)
        trans_pos = flow12 + grid
        
        # Initialize output arrays with padding - use fixed padding of 2
        padding = 2
        warped_frame = torch.zeros(size=(b, h + padding, w + padding, c), dtype=dtype, device=device)
        warped_weights = torch.zeros(size=(b, h + padding, w + padding, 1), dtype=dtype, device=device)
        
        # Convert frame to channel-last format for easier splatting
        frame1_cl = frame1.permute(0, 2, 3, 1).contiguous()  # [b, h, w, c]
        
        # Prepare depth weighting - compute once
        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        depth_min = sat_depth1.min()
        depth_max = sat_depth1.max() + 1e-8
        norm_depth = (sat_depth1 - depth_min) / (depth_max - depth_min)
        depth_weights = 1.0 + norm_depth * 1.5  # Linear scaling
        
        # Prepare a mask factor that combines mask1 and flow12_mask
        mask_factor = mask1 * flow12_mask / depth_weights
        mask_factor = mask_factor.squeeze(1)  # Remove channel dim for easier operations
        
        # First, handle the original position
        pos_offset = trans_pos + 1  # +1 for padding
        
        # Process the original position (weight factor = 1.0)
        self._process_position(frame1_cl, mask_factor, pos_offset, 1.0, warped_frame, warped_weights, b, h, w, device)
        
        # Only process additional positions if splat_radius > 1
        if _splat_radius > 1.0:
            num_extra = max(1, int(_splat_radius))
            
            # Create a grid of offsets
            y_offsets = torch.arange(-num_extra, num_extra + 1, device=device)
            x_offsets = torch.arange(-num_extra, num_extra + 1, device=device)
            y_grid, x_grid = torch.meshgrid(y_offsets, x_offsets, indexing='ij')
            offsets = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)  # [N, 2]
            
            # Calculate distances and filter valid offsets
            distances = torch.sqrt((offsets**2).sum(dim=1))
            valid_mask = (distances <= _splat_radius) & (distances > 0)  # Exclude the center point (0,0)
            valid_offsets = offsets[valid_mask]  # [M, 2]
            valid_distances = distances[valid_mask]  # [M]
            
            # Calculate weight factors
            weight_factors = 1.0 - valid_distances / _splat_radius  # [M]
            
            # Process each valid offset
            for i in range(len(valid_offsets)):
                offset = valid_offsets[i]
                weight_factor = weight_factors[i].item()
                
                # Create offset for this point
                offset_x, offset_y = offset
                curr_offset = torch.zeros_like(trans_pos)
                curr_offset[:, 0] = offset_x
                curr_offset[:, 1] = offset_y
                
                # Apply the offset and add padding
                curr_pos_offset = trans_pos + curr_offset + 1
                
                # Process this position
                self._process_position(frame1_cl, mask_factor, curr_pos_offset, weight_factor, 
                                    warped_frame, warped_weights, b, h, w, device)
        
        # Convert to channel-first format
        warped_frame_cf = warped_frame.permute(0, 3, 1, 2)
        warped_weights_cf = warped_weights.permute(0, 3, 1, 2)
        
        # Crop to original size
        half_padding = padding // 2
        cropped_warped_frame = warped_frame_cf[:, :, half_padding:h+half_padding, half_padding:w+half_padding]
        cropped_weights = warped_weights_cf[:, :, half_padding:h+half_padding, half_padding:w+half_padding]

        # Create the final output
        mask = cropped_weights > 1e-6
        zero_value = -1 if is_image else 0
        zero_tensor = torch.full_like(cropped_warped_frame, zero_value)
        
        # Divide only where mask is True
        warped_frame2 = torch.where(mask, cropped_warped_frame / (cropped_weights + 1e-8), zero_tensor)
        mask2 = mask.to(frame1.dtype)

        if is_image:
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
            
        return warped_frame2, mask2
        
    def _process_position(self, frame_cl, mask_factor, pos_offset, weight_factor,
                         warped_frame, warped_weights, b, h, w, device):
        """Helper method to process a single position for splatting"""
        # Calculate floor and ceiling coordinates
        floor_x = torch.floor(pos_offset[:, 0]).long()
        floor_y = torch.floor(pos_offset[:, 1]).long()
        ceil_x = floor_x + 1
        ceil_y = floor_y + 1
        
        # Clamp coordinates to valid range
        floor_x = torch.clamp(floor_x, 0, w + 1)
        floor_y = torch.clamp(floor_y, 0, h + 1)
        ceil_x = torch.clamp(ceil_x, 0, w + 1)
        ceil_y = torch.clamp(ceil_y, 0, h + 1)
        
        # Calculate fractional parts
        frac_x = pos_offset[:, 0] - floor_x.float()
        frac_y = pos_offset[:, 1] - floor_y.float()
        
        # Calculate bilinear weights
        nw_weight = (1.0 - frac_y) * (1.0 - frac_x) * mask_factor * weight_factor
        ne_weight = (1.0 - frac_y) * frac_x * mask_factor * weight_factor
        sw_weight = frac_y * (1.0 - frac_x) * mask_factor * weight_factor
        se_weight = frac_y * frac_x * mask_factor * weight_factor
        
        # Process each batch item separately to ensure proper indexing
        for b_idx in range(b):
            frame_b = frame_cl[b_idx]  # [h, w, c]
            
            # Get coordinates for this batch
            b_floor_x = floor_x[b_idx]
            b_floor_y = floor_y[b_idx]
            b_ceil_x = ceil_x[b_idx]
            b_ceil_y = ceil_y[b_idx]
            
            # Get weights for this batch
            b_nw_weight = nw_weight[b_idx].unsqueeze(-1)  # [h, w, 1]
            b_ne_weight = ne_weight[b_idx].unsqueeze(-1)  # [h, w, 1]
            b_sw_weight = sw_weight[b_idx].unsqueeze(-1)  # [h, w, 1]
            b_se_weight = se_weight[b_idx].unsqueeze(-1)  # [h, w, 1]
            
            # Create index tensors
            y_indices, x_indices = torch.meshgrid(
                torch.arange(h, device=device), 
                torch.arange(w, device=device),
                indexing='ij'
            )
            
            # Process each corner contribution
            # NW corner
            warped_frame[b_idx, b_floor_y, b_floor_x] += frame_b * b_nw_weight
            warped_weights[b_idx, b_floor_y, b_floor_x, 0] += b_nw_weight.squeeze(-1)
            
            # NE corner
            warped_frame[b_idx, b_floor_y, b_ceil_x] += frame_b * b_ne_weight
            warped_weights[b_idx, b_floor_y, b_ceil_x, 0] += b_ne_weight.squeeze(-1)
            
            # SW corner
            warped_frame[b_idx, b_ceil_y, b_floor_x] += frame_b * b_sw_weight
            warped_weights[b_idx, b_ceil_y, b_floor_x, 0] += b_sw_weight.squeeze(-1)
            
            # SE corner
            warped_frame[b_idx, b_ceil_y, b_ceil_x] += frame_b * b_se_weight
            warped_weights[b_idx, b_ceil_y, b_ceil_x, 0] += b_se_weight.squeeze(-1)

    def _fast_warp_with_grid_sample(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                                  flow12: torch.Tensor, flow12_mask: Optional[torch.Tensor], is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast warping using grid_sample for the simple case (no splatting)
        Much faster than bilinear splatting when splat_radius <= 1
        """
        b, c, h, w = frame1.shape
        device = frame1.device
        
        # Handle optional inputs
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w), device=device)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w), device=device)
        
        # Create the sampling grid
        grid = self.create_grid(b, h, w).to(device)
        trans_pos = flow12 + grid
        
        # Convert to the format expected by grid_sample: Normalized to [-1, 1]
        norm_trans_pos = torch.empty_like(trans_pos)
        norm_trans_pos[:, 0] = 2.0 * trans_pos[:, 0] / (w - 1) - 1.0
        norm_trans_pos[:, 1] = 2.0 * trans_pos[:, 1] / (h - 1) - 1.0
        
        # Permute to [b, h, w, 2] as expected by grid_sample
        grid_for_sampling = norm_trans_pos.permute(0, 2, 3, 1)
        
        # Sample the frame and mask
        # Use 'zeros' padding mode to handle out-of-bounds coordinates
        warped_frame = F.grid_sample(frame1, grid_for_sampling, mode='bilinear', 
                                     padding_mode='zeros', align_corners=True)
        warped_mask = F.grid_sample(mask1 * flow12_mask, grid_for_sampling, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)
        
        # Threshold the mask
        mask2 = (warped_mask > 0.5).to(frame1.dtype)
        
        # Apply depth weighting if needed
        if depth1 is not None:
            # Same depth weighting as in the full function
            sat_depth1 = torch.clamp(depth1, min=0, max=1000)
            depth_min = sat_depth1.min()
            depth_max = sat_depth1.max() + 1e-8
            norm_depth = (sat_depth1 - depth_min) / (depth_max - depth_min)
            depth_weights = 1.0 + norm_depth * 1.5
            
            # Sample the depth weights
            warped_depth_weights = F.grid_sample(1.0 / depth_weights, grid_for_sampling, 
                                              mode='bilinear', padding_mode='zeros', align_corners=True)
            
            # Apply depth weights
            warped_frame = warped_frame * warped_depth_weights
        
        if is_image:
            warped_frame = torch.clamp(warped_frame, min=-1, max=1)
            
        return warped_frame, mask2

    def bilinear_interpolation(self, frame2: torch.Tensor, mask2: Optional[torch.Tensor], flow12: torch.Tensor,
                               flow12_mask: Optional[torch.Tensor], is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear interpolation
        :param frame2: (b, c, h, w)
        :param mask2: (b, 1, h, w): 1 for known, 0 for unknown. Optional
        :param flow12: (b, 2, h, w)
        :param flow12_mask: (b, 1, h, w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame1: (b, c, h, w)
                 mask1: (b, 1, h, w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame2.shape[2:4] == self.resolution
        b, c, h, w = frame2.shape
        if mask2 is None:
            mask2 = torch.ones(size=(b, 1, h, w)).to(frame2)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame2)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        weight_nw = torch.moveaxis(prox_weight_nw * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])

        frame2_offset = F.pad(frame2, [1, 1, 1, 1])
        mask2_offset = F.pad(mask2, [1, 1, 1, 1])
        bi = torch.arange(b)[:, None, None]

        f2_nw = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0]]
        f2_sw = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]]
        f2_ne = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]]
        f2_se = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]]

        m2_nw = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0]]
        m2_sw = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]]
        m2_ne = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]]
        m2_se = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]]

        nr = weight_nw * f2_nw * m2_nw + weight_sw * f2_sw * m2_sw + \
             weight_ne * f2_ne * m2_ne + weight_se * f2_se * m2_se
        dr = weight_nw * m2_nw + weight_sw * m2_sw + weight_ne * m2_ne + weight_se * m2_se

        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=nr.dtype, device=nr.device)
        warped_frame1 = torch.where(dr > 0, nr / dr, zero_tensor)
        mask1 = (dr > 0).to(frame2)

        # Convert to channel first
        warped_frame1 = torch.moveaxis(warped_frame1, [0, 1, 2, 3], [0, 2, 3, 1])
        mask1 = torch.moveaxis(mask1, [0, 1, 2, 3], [0, 2, 3, 1])

        if is_image:
            assert warped_frame1.min() >= -1.1  # Allow for rounding errors
            assert warped_frame1.max() <= 1.1
            warped_frame1 = torch.clamp(warped_frame1, min=-1, max=1)
        return warped_frame1, mask1

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def read_image(path: Path) -> torch.Tensor:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def read_depth(path: Path) -> torch.Tensor:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        elif path.suffix == '.exr':
            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.suffix}')
        return depth

    @staticmethod
    def camera_intrinsic_transform(capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(3)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        if device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device('cpu')
        return device

def rgb_to_mask(image, color_map=None):
        """
        Convert a (..., 3) image to a (...) mask using a color map.

        Parameters:
        - image: np.ndarray of shape (..., 3), dtype=np.uint8
        - color_map: dict mapping (R, G, B) tuples to integer labels

        Returns:
        - mask: np.ndarray of shape (...), dtype=int
        """
        if color_map is None:
            color_map = {
                (0, 0, 0) : 0,
                (255, 0, 0) : 1,
                (0, 255, 0) : 2,
                (0, 0, 255) : 3,
                (255, 255, 0) : 4 
            }
        # Ensure image is uint8
        image = image.astype(np.uint8)

        # Prepare output mask
        shape = image.shape
        mask = np.zeros(shape[:-1], dtype=np.uint8)

        # Create a view to compare colors efficiently
        for rgb, label in color_map.items():
            matches = (image[..., 0] == rgb[0]) & \
                    (image[..., 1] == rgb[1]) & \
                    (image[..., 2] == rgb[2])
            mask[matches] = label

        return mask

def main(room_camera_params_path, wrist_camera_params_path, transfered_video_path, wrist_img_video_path, seg_depth_images_path, output_dir, device='gpu0', return_concat_video=False, fill_missing_pixels=False):
    """Process video using GPU-accelerated warping"""

    # Get camera parameters from saved npz file
    room_camera_params = np.load(room_camera_params_path)
    room_camera_intrinsic_matrices = room_camera_params['room_camera_intrinsic_matrices']  # (n_frames, 3, 3)
    room_camera_pos = room_camera_params['room_camera_pos']  # (n_frames, 3)
    room_camera_quat = room_camera_params['room_camera_quat']  # (n_frames, 4)

    wrist_camera_params = np.load(wrist_camera_params_path)
    wrist_camera_intrinsic_matrices = wrist_camera_params['wrist_camera_intrinsic_matrices']  # (n_frames, 3, 3)
    wrist_camera_pos = wrist_camera_params['wrist_camera_pos']  # (n_frames, 3)
    wrist_camera_quat = wrist_camera_params['wrist_camera_quat']  # (n_frames, 4)
    

    wrist_images, _ = read_video_or_image_into_frames(wrist_img_video_path) # (n_frames, 224, 224, 3)
    transfered_images, fps = read_video_or_image_into_frames(transfered_video_path)
    depth_images = np.load(seg_depth_images_path)['depth_images'] # (n_frames, 2, 224, 224, 1)
    seg_images = np.load(seg_depth_images_path)['seg_images']  # (n_frames, 2, 224, 224)
    if seg_images.shape[-1] == 3:
        seg_images = rgb_to_mask(seg_images)
    
    print("seg_images.shape:", seg_images.shape)
    print("depth_images.shape:", depth_images.shape)
    print("transfered_images.shape:", transfered_images.shape)
    print("wrist_images.shape:", wrist_images.shape)

    assert transfered_images.shape[0] == wrist_images.shape[0] == depth_images.shape[0] == seg_images.shape[0]
    print("Number of frames:", transfered_images.shape[0])
    H, W, C = transfered_images.shape[1:]

    # Initialize warper with GPU support
    warper = Warper(device=device)
    
    warped_images = []
    roi_masks = []
    
    for frame_idx in tqdm(range(transfered_images.shape[0])):

        # Extract parameters for current frame
        room_camera_intrinsic_matrix = room_camera_intrinsic_matrices[frame_idx, :, :]
        wrist_camera_intrinsic_matrix = wrist_camera_intrinsic_matrices[frame_idx, :, :]
        room_camera_pos_frame = room_camera_pos[frame_idx, :]
        wrist_camera_pos_frame = wrist_camera_pos[frame_idx, :]
        room_camera_quat_frame = room_camera_quat[frame_idx, :]
        wrist_camera_quat_frame = wrist_camera_quat[frame_idx, :]

        # Print camera parameters
        if frame_idx == 0:
            print("\nCamera Parameters (first frame):")
            print("Room camera intrinsics:\n", room_camera_intrinsic_matrix)
            print("Wrist camera intrinsics:\n", wrist_camera_intrinsic_matrix)
            print("Room camera position:", room_camera_pos_frame)
            print("Wrist camera position:", wrist_camera_pos_frame)
            print("Room camera quaternion (w,x,y,z):", room_camera_quat_frame)
            print("Wrist camera quaternion (w,x,y,z):", wrist_camera_quat_frame)

        # Load images
        room_img = transfered_images[frame_idx, :, :, :].astype(np.uint8)
        wrist_img = wrist_images[frame_idx, :, :, :].astype(np.uint8)
        
        # Process segmentation for masking
        room_seg_img = seg_images[frame_idx, 0, ...].astype(np.uint8)
        frame1_mask = np.zeros_like(room_seg_img, dtype=bool)
        frame1_mask[room_seg_img == 3] = True
        frame1_mask[room_seg_img == 4] = True
        frame1_mask = ~frame1_mask  # Invert mask
        
        wrist_seg_img = seg_images[frame_idx, 1, ...].astype(np.uint8)
        frame2_mask = np.zeros_like(wrist_seg_img, dtype=np.uint8)
        frame2_mask[wrist_seg_img == 3] = 1
        frame2_mask[wrist_seg_img == 4] = 1
        frame2_mask_3ch = np.repeat(frame2_mask[:, :, None], 3, axis=2)

        frame2_table_mask = np.zeros_like(wrist_seg_img, dtype=np.uint8)
        frame2_table_mask[wrist_seg_img == 2] = 1
        frame2_table_mask_3ch = np.repeat(frame2_table_mask[:, :, None], 3, axis=2)
        # frame2_table_mask = np.repeat(frame2_table_mask[:, :, None], 3, axis=2)


        # Process depth
        room_depth = depth_images[frame_idx, 0, :, :, 0].copy()
        room_depth[room_depth > 5] = 5

        # Build camera extrinsics
        transformation1 = build_extrinsic_from_ros_pose(room_camera_pos_frame, room_camera_quat_frame)
        transformation2 = build_extrinsic_from_ros_pose(wrist_camera_pos_frame, wrist_camera_quat_frame)
        
        # Convert numpy arrays to torch tensors with batch dimension
        frame1_tensor = torch.from_numpy(room_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        mask1_tensor = torch.from_numpy(frame1_mask).unsqueeze(0).unsqueeze(0).float()
        depth1_tensor = torch.from_numpy(room_depth).unsqueeze(0).unsqueeze(0).float()
        transformation1_tensor = torch.from_numpy(transformation1).unsqueeze(0).float()
        transformation2_tensor = torch.from_numpy(transformation2).unsqueeze(0).float()
        intrinsic1_tensor = torch.from_numpy(room_camera_intrinsic_matrix).unsqueeze(0).float()
        intrinsic2_tensor = torch.from_numpy(wrist_camera_intrinsic_matrix).unsqueeze(0).float()
        
        # print(frame1_tensor.max(), frame1_tensor.min())

        # Forward warp using GPU
        result = warper.forward_warp(
            frame1_tensor, mask1_tensor, depth1_tensor, 
            transformation1_tensor, transformation2_tensor, 
            intrinsic1_tensor, intrinsic2_tensor, 
            splat_radius=6.0
        )
        
        # Extract results and convert back to numpy
        warped_frame2_tensor, warped_mask_tensor = result[0], result[1]
        # print(warped_frame2_tensor.device, warped_mask_tensor.device)

        # print(warped_frame2_tensor.shape, warped_frame2_tensor.max(), warped_frame2_tensor.min())

        warped_frame2 = (warped_frame2_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        warped_mask = warped_mask_tensor.squeeze().cpu().numpy().astype(bool)

        # print(warped_frame2.shape, warped_frame2.max(), warped_frame2.min())
        # break

        
        # Create final output by combining warped and original frames
        if wrist_img is not None and frame2_mask_3ch is not None:
            if fill_missing_pixels:
                import cv2
                masked_pixels = warped_frame2[warped_mask]
                median_per_channel = np.median(masked_pixels, axis=0)
                # print(masked_pixels.shape)
                # print(median_per_channel)
                warped_mask_erode = warped_mask.astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
                warped_mask_erode = cv2.erode(warped_mask_erode, kernel, iterations=1)
                fill_missing_pixels_mask = ((1 - warped_mask_erode).astype(np.uint8) * frame2_table_mask).astype(bool)
                # Step 1: Convert boolean mask to uint8 (0 or 255)
                mask_uint8 = (fill_missing_pixels_mask.astype(np.uint8)) * 255
                # Step 2: Blur the mask — this creates a soft boundary
                # You can adjust (21, 21) for blur strength
                blurred_mask = cv2.GaussianBlur(mask_uint8, (21, 21), sigmaX=0)
                # Normalize blurred_mask to range [0.0, 1.0]
                blurred_mask = blurred_mask.astype(np.float32) / 255.0
                # Reshape median RGB color for broadcasting
                median_rgb = median_per_channel.reshape(1, 1, 3)
                # Blend only in mask region
                # Expand blurred mask to 3 channels
                alpha = np.expand_dims(blurred_mask, axis=-1)  # shape (H, W, 1)
                # Create output image with smooth transition
                warped_frame2 = warped_frame2 * (1 - alpha) + median_rgb * alpha
                # Convert to uint8 if needed
                warped_frame2 = np.clip(warped_frame2, 0, 255).astype(np.uint8)
                # print(fill_missing_pixels_mask.shape)
                # warped_frame2[fill_missing_pixels_mask] = median_per_channel
                # print(warped_frame2.shape)
                # warped_mask_3ch_flat = warped_mask_3ch.reshape(-1, 3)
                # warped_frame2_flat = warped_frame2.reshape(-1, 3)
                # warped_frame2_flat_pixels = warped_frame2_flat[warped_mask_3ch_flat]
                # print(warped_mask_3ch_flat.shape, warped_frame2_flat.shape, warped_frame2_flat_pixels.shape)
                # raise NotImplementedError("Fill missing pixels is not implemented yet.")
                
                warped_frame2 = warped_frame2 * (1 - frame2_mask_3ch) + wrist_img * frame2_mask_3ch

            combined_frame = warped_frame2 * (1 - frame2_mask_3ch) + wrist_img * frame2_mask_3ch
            warped_images.append(combined_frame)
            
            # Create ROI mask
            roi_mask = np.zeros_like(warped_mask, dtype=np.uint8)
            roi_mask[warped_mask] = 1
            if fill_missing_pixels:
                roi_mask[fill_missing_pixels_mask] = 1
            roi_mask[frame2_mask > 0] = 2
            roi_masks.append(roi_mask)
        else:
            warped_images.append(warped_frame2)
            roi_masks.append(warped_mask.astype(np.uint8))
    
    # Save results
    if return_concat_video:
        H = H * 2
        warped_video = np.concatenate([transfered_images, np.stack(warped_images, axis=0)], axis=1)
        save_video(warped_video, fps, H, W, f"{output_dir}/warped_video.mp4")
    else:
        save_video(np.stack(warped_images, axis=0), fps, H, W, f"{output_dir}/warped_video.mp4")
    
    roi_masks_array = np.stack(roi_masks, axis=0)
    if return_concat_video:
        roi_masks_array = np.concatenate([(np.ones_like(roi_masks_array)*2).astype(np.uint8), roi_masks_array], axis=1)
    print("ROI mask unique values:", np.unique(roi_masks_array))
    save_video((roi_masks_array * 127.5).astype(np.uint8), fps, H, W, f"{output_dir}/roi_masks.mp4")
    
    # Save binary masks for each region
    roi_masks_binary = np.stack([
        (roi_masks_array == 1).astype(np.uint8), 
        (roi_masks_array == 2).astype(np.uint8)
    ])
    print("Binary ROI masks shape:", roi_masks_binary.shape)
    np.savez(f"{output_dir}/roi_masks.npz", roi_masks_binary)
    
    print(f"Results saved to {output_dir}")
    return f"{output_dir}/warped_video.mp4", f"{output_dir}/roi_masks.npz"

def concat_videos(video1_path, video2_path, output_path, direction='horizontal'):
    # Open the video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    
    if direction == 'horizontal':
        # Use the minimum height and sum of widths for the output video
        output_height = min(height1, height2)
        output_width = width1 + width2
    else:  # vertical
        # Use the minimum width and sum of heights for the output video
        output_width = min(width1, width2)
        output_height = height1 + height2
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, min(fps1, fps2), (output_width, output_height))
    
    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Break if either video is finished
        if not ret1 or not ret2:
            break
        
        if direction == 'horizontal':
            # Resize frames to match the output height
            frame1 = cv2.resize(frame1, (width1, output_height))
            frame2 = cv2.resize(frame2, (width2, output_height))
            # Concatenate frames horizontally
            combined_frame = np.hstack((frame1, frame2))
        else:  # vertical
            # Resize frames to match the output width
            frame1 = cv2.resize(frame1, (output_width, height1))
            frame2 = cv2.resize(frame2, (output_width, height2))
            # Concatenate frames vertically
            combined_frame = np.vstack((frame1, frame2))
        
        # Write the combined frame
        out.write(combined_frame)
    
    # Release everything
    cap1.release()
    cap2.release()
    out.release()

if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time))) 