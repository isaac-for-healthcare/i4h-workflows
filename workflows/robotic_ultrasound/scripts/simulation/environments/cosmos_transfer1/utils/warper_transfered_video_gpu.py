# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import skimage.io
import torch
import torch.nn.functional as F
from cosmos_transfer1.utils.io import save_video
from scipy.spatial.transform import Rotation as R
from simulation.environments.cosmos_transfer1.utils.inference_utils import read_video_or_image_into_frames, rgb_to_mask
from tqdm import tqdm


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
    R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # 3) Invert to world→camera
    R_cw = R_wc.T
    t_w = np.array(pos_w).reshape(3, 1)
    t_cw = -R_cw @ t_w

    # 4) Assemble homogeneous matrix
    E = np.eye(4)
    E[:3, :3] = R_cw
    E[:3, 3] = t_cw.flatten()
    return E


class Warper:
    """
    Warper class for warping videos using GPU-accelerated bilinear splatting.
    Ref: https://github.com/NagabhushanSN95/Pose-Warping/blob/main/src/WarperPytorch.py
    """

    def __init__(self, resolution: tuple = None, device: str = "gpu0", splat_radius: float = 1.5):
        """
        Initialize the Warper class.
        Args:
            resolution: Resolution of the images and videos.
            device: Device to run the warping on.
            splat_radius: Radius of the splatting.
        """
        self.resolution = resolution
        self.device = self.get_device(device)
        self.splat_radius = splat_radius
        return

    def forward_warp(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
        splat_radius: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        Args:
            frame1: frame to warp (b,3,h,w)
            mask1: mask of frame1 (b,1,h,w): 1 for known, 0 for unknown. Optional
            depth1: depth of frame1 (b,1,h,w)
            transformation1: transformation matrix from first view to world (b,4,4)
            transformation2: transformation matrix from second view to world (b,4,4)
            intrinsic1: camera intrinsic matrix of first view (b,3,3)
            intrinsic2: camera intrinsic matrix of second view (b,3,3). Optional
            splat_radius: Controls how far each point spreads. Larger values create denser but more blurred results.
        Returns:
            warped_frame2: warped frame (b,3,h,w)
            mask2: mask of warped frame (b,1,h,w): 1 for known and 0 for unknown
            warped_frame1: warped frame (b,3,h,w)
            mask1: mask of warped frame (b,1,h,w): 1 for known and 0 for unknown
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

        trans_points1 = self.compute_transformed_points(
            depth1, transformation1, transformation2, intrinsic1, intrinsic2
        )

        # Fix the dimension issue - ensure that trans_coordinates and grid have compatible dimensions
        trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]  # (b, h*w, 2)
        trans_coordinates = trans_coordinates.reshape(b, h, w, 2).permute(0, 3, 1, 2)  # (b, 2, h, w)
        trans_depth1 = trans_points1[:, :, 2, 0].reshape(b, h, w)

        grid = self.create_grid(b, h, w).to(trans_coordinates.device)

        flow12 = trans_coordinates - grid

        # Ensure trans_depth1 has the right shape for bilinear_splatting (b, 1, h, w)
        trans_depth1_reshaped = trans_depth1.unsqueeze(1)  # Add channel dimension -> (b, 1, h, w)

        # Warp the RGB frame
        warped_frame2, mask2 = self.bilinear_splatting(
            frame1, mask1, trans_depth1_reshaped, flow12, None, is_image=True, splat_radius=_splat_radius
        )
        return warped_frame2, mask2

    def compute_transformed_points(
        self,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
    ):
        """
        Computes transformed position for each pixel location
        Args:
            depth1: depth image (b, 1, h, w)
            transformation1: transformation matrix from first view to world (b, 4, 4)
            transformation2: transformation matrix from second view to world (b, 4, 4)
            intrinsic1: camera intrinsic matrix of first view (b, 3, 3)
            intrinsic2: camera intrinsic matrix of second view (b, 3, 3)
        Returns:
            transformed_points: (b, h*w, 3, 1)
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
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")

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
        depth_reshaped = depth1.reshape(b, 1, h * w)
        cam_points = unprojected * depth_reshaped

        # 3. Convert to homogeneous coordinates
        ones_hw = torch.ones(b, 1, h * w, device=device)
        cam_points_homo = torch.cat([cam_points, ones_hw], dim=1)  # [b, 4, h*w]

        # 4. Transform points using the transformation matrix
        transformed_points = torch.bmm(transformation, cam_points_homo)  # [b, 4, h*w]

        # 5. Project back to image coordinates using the target intrinsics
        transformed_3d = transformed_points[:, :3, :]  # [b, 3, h*w]
        projected_points = torch.bmm(intrinsic2, transformed_3d)  # [b, 3, h*w]

        # Reshape to match expected format [b, h*w, 3, 1]
        projected_points = projected_points.permute(0, 2, 1).unsqueeze(-1)  # [b, h*w, 3, 1]

        return projected_points

    def bilinear_splatting(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor],
        is_image: bool = False,
        splat_radius: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame1: frame to warp (b,c,h,w)
            mask1: mask of frame1 (b,1,h,w): 1 for known, 0 for unknown. Optional
            depth1: depth of frame1 (b,1,h,w)
            flow12: flow from frame1 to frame2 (b,2,h,w)
            flow12_mask: mask of flow12 (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
            is_image: if true, output will be clipped to (-1,1) range
            splat_radius: Controls how far each point spreads. Larger values create denser but more blurred results.

        Returns:
            warped_frame2: (b,c,h,w)
            mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        device = frame1.device
        dtype = frame1.dtype

        # Fast-path: If splat_radius <= 1, we can use grid_sample which is much faster
        if (splat_radius is None or splat_radius <= 1.0) and getattr(self, "splat_radius", 1.0) <= 1.0:
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
            y_grid, x_grid = torch.meshgrid(y_offsets, x_offsets, indexing="ij")
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
                self._process_position(
                    frame1_cl,
                    mask_factor,
                    curr_pos_offset,
                    weight_factor,
                    warped_frame,
                    warped_weights,
                    b,
                    h,
                    w,
                    device,
                )

        # Convert to channel-first format
        warped_frame_cf = warped_frame.permute(0, 3, 1, 2)
        warped_weights_cf = warped_weights.permute(0, 3, 1, 2)

        # Crop to original size
        half_padding = padding // 2
        cropped_warped_frame = warped_frame_cf[:, :, half_padding : h + half_padding, half_padding : w + half_padding]
        cropped_weights = warped_weights_cf[:, :, half_padding : h + half_padding, half_padding : w + half_padding]

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

    def _process_position(
        self, frame_cl, mask_factor, pos_offset, weight_factor, warped_frame, warped_weights, b, h, w, device
    ):
        """Helper method to process a single position for splatting
        Args:
            frame_cl: frame to warp (b,h,w,c)
            mask_factor: mask factor (b,h,w)
            pos_offset: position offset (b,2,h,w)
            weight_factor: weight factor (b,h,w)
            warped_frame: warped frame (b,h,w,c)
            warped_weights: warped weights (b,h,w,1)
            b: batch size
            h: height
            w: width
            device: device
        Returns:
            warped_frame: warped frame (b,h,w,c)
            warped_weights: warped weights (b,h,w,1)
        """
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
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
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

    def _fast_warp_with_grid_sample(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor],
        is_image: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast warping using grid_sample for the simple case (no splatting)
        Much faster than bilinear splatting when splat_radius <= 1
        Args:
            frame1: frame to warp (b,c,h,w)
            mask1: mask of frame1 (b,1,h,w): 1 for known, 0 for unknown. Optional
            depth1: depth of frame1 (b,1,h,w)
            flow12: flow from frame1 to frame2 (b,2,h,w)
            flow12_mask: mask of flow12 (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
            is_image: if true, output will be clipped to (-1,1) range
        Returns:
            warped_frame: warped frame (b,c,h,w)
            mask2: mask of warped frame (b,1,h,w): 1 for known and 0 for unknown
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
        warped_frame = F.grid_sample(
            frame1, grid_for_sampling, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        warped_mask = F.grid_sample(
            mask1 * flow12_mask, grid_for_sampling, mode="bilinear", padding_mode="zeros", align_corners=True
        )

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
            warped_depth_weights = F.grid_sample(
                1.0 / depth_weights, grid_for_sampling, mode="bilinear", padding_mode="zeros", align_corners=True
            )

            # Apply depth weights
            warped_frame = warped_frame * warped_depth_weights

        if is_image:
            warped_frame = torch.clamp(warped_frame, min=-1, max=1)

        return warped_frame, mask2

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
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        if device == "cpu":
            device = torch.device("cpu")
        elif device.startswith("gpu") and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f"cuda:{gpu_num}")
        else:
            device = torch.device("cpu")
        return device


def main(
    room_camera_params_path: str,
    wrist_camera_params_path: str,
    transferred_video_path: str,
    wrist_img_video_path: str,
    seg_depth_images_path: str,
    output_dir: str,
    device: str = "gpu0",
    return_concat_video: bool = False,
    fill_missing_pixels: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process video using GPU-accelerated warping
    Args:
        room_camera_params_path: path to room camera parameters
        wrist_camera_params_path: path to wrist camera parameters
        transferred_video_path: path to transferred video
        wrist_img_video_path: path to wrist image video
        seg_depth_images_path: path to segmentation and depth images
        output_dir: path to output directory
        device: device to use
        return_concat_video: if True, return concatenated video
        fill_missing_pixels: if True, fill missing pixels
    Returns:
        warped_images: warped video
        roi_masks: roi masks of warped video
    """

    # Get camera parameters from saved npz file
    room_camera_params = np.load(room_camera_params_path)
    room_camera_intrinsic_matrices = room_camera_params["room_camera_intrinsic_matrices"]  # (n_frames, 3, 3)
    room_camera_pos = room_camera_params["room_camera_pos"]  # (n_frames, 3)
    room_camera_quat = room_camera_params["room_camera_quat"]  # (n_frames, 4)

    wrist_camera_params = np.load(wrist_camera_params_path)
    wrist_camera_intrinsic_matrices = wrist_camera_params["wrist_camera_intrinsic_matrices"]  # (n_frames, 3, 3)
    wrist_camera_pos = wrist_camera_params["wrist_camera_pos"]  # (n_frames, 3)
    wrist_camera_quat = wrist_camera_params["wrist_camera_quat"]  # (n_frames, 4)

    wrist_images, _ = read_video_or_image_into_frames(wrist_img_video_path)  # (n_frames, 224, 224, 3)
    transferred_images, fps = read_video_or_image_into_frames(transferred_video_path)
    depth_images = np.load(seg_depth_images_path)["depth_images"]  # (n_frames, 2, 224, 224, 1)
    seg_images = np.load(seg_depth_images_path)["seg_images"]  # (n_frames, 2, 224, 224, 3)
    if seg_images.shape[-1] == 3:
        seg_images = rgb_to_mask(seg_images)

    assert transferred_images.shape[0] == wrist_images.shape[0] == depth_images.shape[0] == seg_images.shape[0]
    H, W, C = transferred_images.shape[1:]

    # Initialize warper with GPU support
    warper = Warper(device=device)

    warped_images = []
    roi_masks = []

    for frame_idx in tqdm(range(transferred_images.shape[0])):
        # Extract parameters for current frame
        room_camera_intrinsic_matrix = room_camera_intrinsic_matrices[frame_idx, :, :]
        wrist_camera_intrinsic_matrix = wrist_camera_intrinsic_matrices[frame_idx, :, :]
        room_camera_pos_frame = room_camera_pos[frame_idx, :]
        wrist_camera_pos_frame = wrist_camera_pos[frame_idx, :]
        room_camera_quat_frame = room_camera_quat[frame_idx, :]
        wrist_camera_quat_frame = wrist_camera_quat[frame_idx, :]

        # Load images
        room_img = transferred_images[frame_idx, :, :, :].astype(np.uint8)
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

        # Forward warp using GPU
        result = warper.forward_warp(
            frame1_tensor,
            mask1_tensor,
            depth1_tensor,
            transformation1_tensor,
            transformation2_tensor,
            intrinsic1_tensor,
            intrinsic2_tensor,
            splat_radius=6.0,
        )

        # Extract results and convert back to numpy
        warped_frame2_tensor, warped_mask_tensor = result[0], result[1]

        warped_frame2 = (warped_frame2_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(
            np.uint8
        )
        warped_mask = warped_mask_tensor.squeeze().cpu().numpy().astype(bool)

        # Create final output by combining warped and original frames
        if wrist_img is not None and frame2_mask_3ch is not None:
            if fill_missing_pixels:
                masked_pixels = warped_frame2[warped_mask]
                median_per_channel = np.median(masked_pixels, axis=0)
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
                # Convert to uint8 and clip to range [0, 255]
                warped_frame2 = np.clip(warped_frame2, 0, 255).astype(np.uint8)

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
        warped_video = np.concatenate([transferred_images, np.stack(warped_images, axis=0)], axis=1)
        save_video(warped_video, fps, H, W, 5, f"{output_dir}/warped_video.mp4")
    else:
        save_video(np.stack(warped_images, axis=0), fps, H, W, 5, f"{output_dir}/warped_video.mp4")

    roi_masks_array = np.stack(roi_masks, axis=0)
    if return_concat_video:
        roi_masks_array = np.concatenate(
            [(np.ones_like(roi_masks_array) * 2).astype(np.uint8), roi_masks_array], axis=1
        )
    save_video((roi_masks_array * 127.5).astype(np.uint8), fps, H, W, 5, f"{output_dir}/roi_masks.mp4")

    # Save binary masks for each region
    roi_masks_binary = np.stack([(roi_masks_array == 1).astype(np.uint8), (roi_masks_array == 2).astype(np.uint8)])
    np.savez(f"{output_dir}/roi_masks.npz", roi_masks_binary)

    return f"{output_dir}/warped_video.mp4", f"{output_dir}/roi_masks.npz"


if __name__ == "__main__":
    print("Program started at " + datetime.datetime.now().strftime("%d/%m/%Y %I:%M:%S %p"))
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    end_time = time.time()
    print("Program ended at " + datetime.datetime.now().strftime("%d/%m/%Y %I:%M:%S %p"))
    print("Execution time: " + str(datetime.timedelta(seconds=end_time - start_time)))
