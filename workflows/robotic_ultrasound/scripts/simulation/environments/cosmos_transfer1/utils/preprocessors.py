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

import json
import os
import tempfile

import numpy as np
import torch
from cosmos_transfer1.auxiliary.depth_anything.model.depth_anything import DepthAnythingModel
from cosmos_transfer1.auxiliary.human_keypoint.human_keypoint import HumanKeypointModel
from cosmos_transfer1.auxiliary.sam2.sam2_model import VideoSegmentationModel, rle_encode
from cosmos_transfer1.auxiliary.sam2.sam2_utils import (
    capture_fps,
    convert_masks_to_frames,
    generate_tensor_from_images,
    video_to_frames,
    write_video,
)
from cosmos_transfer1.utils import log
from PIL import Image

SAM1_MODEL_CHECKPOINT = os.path.join(os.getenv("CHECKPOINT_DIR"), "sam_vit_h_4b8939.pth")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class VideoSegmentationModelWithSAM1(VideoSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize SAM1 predictor with the provided checkpoint
        self.sam1_predictor = sam_model_registry["vit_h"](checkpoint=SAM1_MODEL_CHECKPOINT).to(self.device)

    def sam1_segment_everything(self, image_path, area_threshold=64 * 64):
        """Segment everything using SAM1."""
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam1_predictor,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=area_threshold,  # Requires open-cv to run post-processing
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            sam1_results = mask_generator.generate(image_np)
        sam1_results = [result for result in sam1_results if result["area"] > area_threshold]
        return sam1_results

    def sample(self, **kwargs):
        """
        Main sampling function for video segmentation.
        Returns a list of detections in which each detection contains a phrase and
        an RLE-encoded segmentation mask (matching the output of the Grounded SAM model).
        """
        video_dir = kwargs.get("video_dir", "")
        mode = kwargs.get("mode", "points")
        input_data = kwargs.get("input_data", None)
        save_dir = kwargs.get("save_dir", None)
        visualize = kwargs.get("visualize", False)

        # Get frame names (expecting frames named as numbers with .jpg/.jpeg extension).
        frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.sam2_predictor.init_state(video_path=video_dir)

            ann_frame_idx = 0
            ann_obj_id = 1
            boxes = None
            points = None
            labels = None
            box = None

            visualization_data = {"mode": mode, "points": None, "labels": None, "box": None, "boxes": None}

            if input_data is not None:
                if mode == "points":
                    points = input_data.get("points")
                    labels = input_data.get("labels")
                    frame_idx, obj_ids, masks = self.sam2_predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
                    )
                    visualization_data["points"] = points
                    visualization_data["labels"] = labels
                elif mode == "box":
                    box = input_data.get("box")
                    frame_idx, obj_ids, masks = self.sam2_predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, box=box
                    )
                    visualization_data["box"] = box
                elif mode == "sam-segment-everything":
                    first_frame_path = os.path.join(video_dir, frame_names[0])
                    sam1_results = self.sam1_segment_everything(first_frame_path)
                    boxes = []
                    obj_ids = []
                    masks = []
                    log.info(f"sam1 found {len(sam1_results)} objects")
                    for object_id, sam1_result in enumerate(sam1_results):
                        self.sam2_predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=ann_frame_idx,
                            obj_id=object_id,
                            mask=sam1_result["segmentation"],
                        )
                        obj_ids.append(object_id)
                        boxes.append(sam1_result["bbox"])
                        masks.append(sam1_result["segmentation"])
                    visualization_data["boxes"] = boxes
                elif mode == "prompt":
                    text = input_data.get("text")
                    first_frame_path = os.path.join(video_dir, frame_names[0])
                    gd_results = self.get_boxes_from_text(first_frame_path, text)
                    boxes = gd_results["boxes"]
                    labels_out = gd_results["labels"]
                    scores = gd_results["scores"]
                    log.info(f"scores: {scores}")
                    if len(boxes) > 0:
                        legacy_mask = kwargs.get("legacy_mask", False)
                        if legacy_mask:
                            # Use only the highest confidence box for legacy mask
                            log.info(f"using legacy_mask: {legacy_mask}")
                            frame_idx, obj_ids, masks = self.sam2_predictor.add_new_points_or_box(
                                inference_state=state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, box=boxes[0]
                            )
                            # Update boxes and labels after processing
                            boxes = boxes[:1]
                            if labels_out is not None:
                                labels_out = labels_out[:1]
                        else:
                            log.info(f"using new_mask: {legacy_mask}")
                            for object_id, (box, label) in enumerate(zip(boxes, labels_out)):
                                frame_idx, obj_ids, masks = self.sam2_predictor.add_new_points_or_box(
                                    inference_state=state, frame_idx=ann_frame_idx, obj_id=object_id, box=box
                                )
                        visualization_data["boxes"] = boxes
                        self.grounding_labels = [str(lbl) for lbl in labels_out] if labels_out is not None else [text]
                    else:
                        print("No boxes detected. Exiting.")
                        return []  # Return empty list if no detections

                if visualize:
                    self.visualize_frame(
                        frame_idx=ann_frame_idx,
                        obj_ids=obj_ids,
                        masks=masks,
                        video_dir=video_dir,
                        frame_names=frame_names,
                        visualization_data=visualization_data,
                        save_dir=save_dir,
                    )

            video_segments = {}  # keys: frame index, values: {obj_id: mask}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                }

                # For propagated frames, visualization_data is not used.
                if visualize:
                    propagate_visualization_data = {
                        "mode": mode,
                        "points": None,
                        "labels": None,
                        "box": None,
                        "boxes": None,
                    }
                    self.visualize_frame(
                        frame_idx=out_frame_idx,
                        obj_ids=out_obj_ids,
                        masks=video_segments[out_frame_idx],
                        video_dir=video_dir,
                        frame_names=frame_names,
                        visualization_data=propagate_visualization_data,
                        save_dir=save_dir,
                    )

        # --- Post-process video_segments to produce a list of detections ---
        if len(video_segments) == 0:
            return []

        first_frame_path = os.path.join(video_dir, frame_names[0])
        first_frame = np.array(Image.open(first_frame_path).convert("RGB"))
        original_shape = first_frame.shape[:2]  # (height, width)

        object_masks = {}  # key: obj_id, value: list of 2D boolean masks
        sorted_frame_indices = sorted(video_segments.keys())
        for frame_idx in sorted_frame_indices:
            segments = video_segments[frame_idx]
            for obj_id, mask in segments.items():
                mask = np.squeeze(mask)
                if mask.ndim != 2:
                    print(f"Warning: Unexpected mask shape {mask.shape} for object {obj_id} in frame {frame_idx}.")
                    continue

                if mask.shape != original_shape:
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize((original_shape[1], original_shape[0]), resample=Image.NEAREST)
                    mask = np.array(mask_img) > 127

                if obj_id not in object_masks:
                    object_masks[obj_id] = []
                object_masks[obj_id].append(mask)

        detections = []
        for obj_id, mask_list in object_masks.items():
            mask_stack = np.stack(mask_list, axis=0)  # shape: (T, H, W)
            # Use our new rle_encode (which now follows the eff_segmentation.RleMaskSAMv2 format)
            rle = rle_encode(mask_stack)
            if mode == "prompt" and hasattr(self, "grounding_labels"):
                phrase = self.grounding_labels[0]
            else:
                phrase = input_data.get("text", "")
            detection = {"phrase": phrase, "segmentation_mask_rle": rle}
            detections.append(detection)

        return detections

    def __call__(
        self,
        input_video,
        output_video=None,
        output_tensor=None,
        prompt=None,
        box=None,
        points=None,
        labels=None,
        weight_scaler=None,
        binarize_video=False,
        legacy_mask=False,
    ):
        log.info(
            f"Processing video: {input_video} to generate segmentation video: {output_video} "
            f"segmentation tensor: {output_tensor}"
        )
        assert os.path.exists(input_video)

        # Prepare input data based on the selected mode.
        if points is not None:
            mode = "points"
            input_data = {"points": self.parse_points(points), "labels": self.parse_labels(labels)}
        elif box is not None:
            mode = "box"
            input_data = {"box": self.parse_box(box)}
        elif prompt is not None:
            if prompt == "sam-segment-everything":
                mode = "sam-segment-everything"
            else:
                mode = "prompt"
            input_data = {"text": prompt}

        with tempfile.TemporaryDirectory() as temp_input_dir:
            fps = capture_fps(input_video)
            video_to_frames(input_video, temp_input_dir)
            with tempfile.TemporaryDirectory() as temp_output_dir:
                masks = self.sample(
                    video_dir=temp_input_dir,
                    mode=mode,
                    input_data=input_data,
                    save_dir=str(temp_output_dir),
                    visualize=True,
                    legacy_mask=legacy_mask,
                )
                if output_video:
                    os.makedirs(os.path.dirname(output_video), exist_ok=True)
                    frames = convert_masks_to_frames(masks)
                    if binarize_video:
                        frames = np.any(frames > 0, axis=-1).astype(np.uint8) * 255
                    write_video(frames, output_video, fps)
                if output_tensor:
                    generate_tensor_from_images(
                        temp_output_dir, output_tensor, fps, "mask", weight_scaler=weight_scaler
                    )


class Preprocessors:
    def __init__(self):
        self.depth_model = None
        self.seg_model = None
        self.keypoint_model = None

    def __call__(self, input_video, input_prompt, control_inputs, output_folder):
        for hint_key in control_inputs:
            if hint_key in ["depth", "seg", "keypoint"]:
                self.gen_input_control(input_video, input_prompt, hint_key, control_inputs[hint_key], output_folder)

            # for all hints we need to create weight tensor if not present
            control_input = control_inputs[hint_key]

            # For each control input modality, compute a spatiotemporal weight tensor as long as
            # the user provides "control_weight_prompt". The object specified in the
            # control_weight_prompt will be treated as foreground and have control_weight for these locations.
            # Everything else will be treated as background and have control weight 0 at those locations.
            if control_input.get("control_weight_prompt", None) is not None:
                prompt = control_input["control_weight_prompt"]
                if prompt == "sam-segment-everything":
                    continue
                log.info(f"{hint_key}: generating control weight tensor with SAM using {prompt=}")
                out_tensor = os.path.join(output_folder, f"{hint_key}_control_weight.pt")
                out_video = os.path.join(output_folder, f"{hint_key}_control_weight.mp4")
                weight_scaler = (
                    control_input["control_weight"] if isinstance(control_input["control_weight"], float) else 1.0
                )
                self.segmentation(
                    in_video=input_video,
                    out_tensor=out_tensor,
                    out_video=out_video,
                    prompt=prompt,
                    weight_scaler=weight_scaler,
                    binarize_video=True,
                )
                control_input["control_weight"] = out_tensor
        return control_inputs

    def gen_input_control(self, in_video, in_prompt, hint_key, control_input, output_folder):
        # if input control isn't provided we need to run preprocessor to create input control tensor
        # for depth no special params, for SAM we need to run with prompt
        if control_input.get("input_control", None) is None:
            out_video = os.path.join(output_folder, f"{hint_key}_input_control.mp4")
            control_input["input_control"] = out_video
            if hint_key == "seg":
                prompt = control_input.get("input_control_prompt", in_prompt)
                prompt = " ".join(prompt.split()[:128])
                log.info(
                    f"no input_control provided for {hint_key}. generating input control video with SAM using {prompt=}"
                )
                self.segmentation(
                    in_video=in_video,
                    out_video=out_video,
                    prompt=prompt,
                )
            elif hint_key == "depth":
                log.info(
                    f"no input_control provided for {hint_key}. generating input control video with DepthAnythingModel"
                )
                self.depth(
                    in_video=in_video,
                    out_video=out_video,
                )
            else:
                log.info(f"no input_control provided for {hint_key}. generating input control video with Openpose")
                self.keypoint(
                    in_video=in_video,
                    out_video=out_video,
                )

    def depth(self, in_video, out_video):
        if self.depth_model is None:
            self.depth_model = DepthAnythingModel()

        self.depth_model(in_video, out_video)

    def keypoint(self, in_video, out_video):
        if self.keypoint_model is None:
            self.keypoint_model = HumanKeypointModel()

        self.keypoint_model(in_video, out_video)

    def segmentation(
        self,
        in_video,
        prompt,
        out_video=None,
        out_tensor=None,
        weight_scaler=None,
        binarize_video=False,
    ):
        if self.seg_model is None:
            self.seg_model = VideoSegmentationModelWithSAM1()
        self.seg_model(
            input_video=in_video,
            output_video=out_video,
            output_tensor=out_tensor,
            prompt=prompt,
            weight_scaler=weight_scaler,
            binarize_video=binarize_video,
        )


if __name__ == "__main__":
    control_inputs = dict(
        {
            "depth": {
                # "input_control": "depth_control_input.mp4",  # if empty we need to run depth
                # "control_weight" : "0.1", # if empty we need to run SAM
                "control_weight_prompt": "a boy",  # SAM weights prompt
            },
            "seg": {
                # "input_control": "seg_control_input.mp4",  # if empty we need to run SAM
                "input_control_prompt": "A boy",
                "control_weight_prompt": "A boy",  # if present we need to generate weight tensor
            },
        },
    )

    preprocessor = Preprocessors()
    input_video = "cosmos_transfer1/models/sam2/assets/input_video.mp4"

    preprocessor(input_video, control_inputs)
    print(json.dumps(control_inputs, indent=4))
