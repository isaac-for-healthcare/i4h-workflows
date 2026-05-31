# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Benchmark script for GR00T N1.7 policy inference on SO-ARM.

Measures end-to-end inference latency in both PyTorch and TensorRT modes
using the same GR00TN1_7_PolicyRunner used for real deployment.

Usage:
    # PyTorch inference benchmark
    python -m policy.gr00tn1_7.benchmark \
        --ckpt_path <path_to_checkpoint>

    # TensorRT inference benchmark
    python -m policy.gr00tn1_7.benchmark \
        --ckpt_path <path_to_checkpoint> \
        --inference_mode tensorrt \
        --trt_engine_path <path_to_engine_dir>

    # Compare PyTorch vs TensorRT accuracy
    python -m policy.gr00tn1_7.benchmark \
        --ckpt_path <path_to_checkpoint> \
        --inference_mode compare \
        --trt_engine_path <path_to_engine_dir>
"""

import argparse
import time

import numpy as np
import torch
from policy.gr00tn1_7.runners import GR00TN1_7_PolicyRunner


def compare_predictions(pred_trt: torch.Tensor, pred_torch: torch.Tensor) -> None:
    """Compare TensorRT and PyTorch prediction similarity."""
    print("\n=== Prediction Comparison ===")

    trt = pred_trt.float()
    pt = pred_torch.float()

    assert trt.shape == pt.shape, f"Shape mismatch: {trt.shape} vs {pt.shape}"

    flat_trt = trt.flatten()
    flat_pt = pt.flatten()

    cos_sim = torch.dot(flat_trt, flat_pt) / (torch.norm(flat_trt) * torch.norm(flat_pt))
    l1_dist = torch.abs(flat_trt - flat_pt)

    label_width = 45
    print(f"\nAction predictions (shape {trt.shape}):")
    print(f"{'Cosine Similarity (PyTorch/TensorRT):'.ljust(label_width)} {cos_sim.item():.6f}")
    print(
        f"{'L1 Mean/Max Distance (PyTorch/TensorRT):'.ljust(label_width)} "
        f"{l1_dist.mean().item():.6f}/{l1_dist.max().item():.6f}"
    )
    print(
        f"{'Max Output Values (PyTorch/TensorRT):'.ljust(label_width)} " f"{pt.max().item():.4f}/{trt.max().item():.4f}"
    )
    print(
        f"{'Mean Output Values (PyTorch/TensorRT):'.ljust(label_width)} "
        f"{pt.mean().item():.4f}/{trt.mean().item():.4f}"
    )
    print(
        f"{'Min Output Values (PyTorch/TensorRT):'.ljust(label_width)} " f"{pt.min().item():.4f}/{trt.min().item():.4f}"
    )


def _make_random_inputs(height: int = 480, width: int = 640):
    """Generate random observation inputs for benchmarking without a dataset."""
    room_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    wrist_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    arm = np.random.randn(5).astype(np.float32)
    gripper = np.random.randn(1).astype(np.float32)
    current_state = np.concatenate([arm, gripper])
    return room_img, wrist_img, current_state


def benchmark_inference(
    policy: GR00TN1_7_PolicyRunner,
    room_img: np.ndarray,
    wrist_img: np.ndarray,
    current_state: np.ndarray,
    num_warmup: int,
    num_runs: int,
    mode_label: str,
) -> torch.Tensor:
    """Run warmup + timed inference and print statistics. Returns last prediction."""
    print(f"\nPerforming {num_warmup} warmup runs...")
    for _ in range(num_warmup):
        _ = policy.infer(room_img=room_img, wrist_img=wrist_img, current_state=current_state)

    inference_times = []
    print(f"Running {num_runs} timed inference runs...")
    for i in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predicted = policy.infer(room_img=room_img, wrist_img=wrist_img, current_state=current_state)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        inference_times.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.2f} ms")

    times = np.array(inference_times)
    print(f"\n=== {mode_label} Inference Performance ===")
    print(f"Number of runs: {num_runs}")
    print(f"Mean inference time: {times.mean():.2f} \u00b1 {times.std():.2f} ms")
    print(f"Min inference time:  {times.min():.2f} ms")
    print(f"Max inference time:  {times.max():.2f} ms")
    print(f"Throughput:          {1000 / times.mean():.2f} inferences/second")
    print(f"Predicted actions shape: {predicted.shape}")

    return predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GR00T N1.7 inference for SO-ARM")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/so-arm/checkpoint-30000",
        help="Path to the GR00T N1.7 checkpoint.",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="new_embodiment",
        help="Embodiment tag for the model.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="Grip the scissors and put it into the tray",
        help="Task description text prompt.",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["pytorch", "tensorrt", "compare"],
        default="pytorch",
        help="Inference mode: 'pytorch', 'tensorrt', or 'compare' (accuracy check).",
    )
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        default=None,
        help="Path to directory containing TRT engine files.",
    )
    parser.add_argument(
        "--trt_mode",
        type=str,
        default="n17_full_pipeline",
        choices=["n17_full_pipeline", "vit_llm_only", "action_head", "dit_only"],
        help="TRT acceleration scope.",
    )
    parser.add_argument("--num_warmup_runs", type=int, default=3, help="Number of warmup runs.")
    parser.add_argument("--num_profile_runs", type=int, default=50, help="Number of profiling runs.")
    parser.add_argument("--height", type=int, default=480, help="Input image height.")
    parser.add_argument("--width", type=int, default=640, help="Input image width.")

    args = parser.parse_args()

    room_img, wrist_img, current_state = _make_random_inputs(args.height, args.width)

    if args.inference_mode == "pytorch":
        policy = GR00TN1_7_PolicyRunner(
            ckpt_path=args.ckpt_path,
            embodiment_tag=args.embodiment_tag,
            task_description=args.task_description,
        )
        benchmark_inference(
            policy,
            room_img,
            wrist_img,
            current_state,
            args.num_warmup_runs,
            args.num_profile_runs,
            "PyTorch",
        )

    elif args.inference_mode == "tensorrt":
        if not args.trt_engine_path:
            raise ValueError("--trt_engine_path is required for tensorrt mode")
        policy = GR00TN1_7_PolicyRunner(
            ckpt_path=args.ckpt_path,
            embodiment_tag=args.embodiment_tag,
            task_description=args.task_description,
            trt_engine_path=args.trt_engine_path,
            trt_mode=args.trt_mode,
        )
        benchmark_inference(
            policy,
            room_img,
            wrist_img,
            current_state,
            args.num_warmup_runs,
            args.num_profile_runs,
            "TensorRT",
        )

    elif args.inference_mode == "compare":
        if not args.trt_engine_path:
            raise ValueError("--trt_engine_path is required for compare mode")

        policy_pt = GR00TN1_7_PolicyRunner(
            ckpt_path=args.ckpt_path,
            embodiment_tag=args.embodiment_tag,
            task_description=args.task_description,
        )
        pred_pt = policy_pt.infer(room_img=room_img, wrist_img=wrist_img, current_state=current_state)

        policy_trt = GR00TN1_7_PolicyRunner(
            ckpt_path=args.ckpt_path,
            embodiment_tag=args.embodiment_tag,
            task_description=args.task_description,
            trt_engine_path=args.trt_engine_path,
            trt_mode=args.trt_mode,
        )
        pred_trt = policy_trt.infer(room_img=room_img, wrist_img=wrist_img, current_state=current_state)

        compare_predictions(pred_trt, pred_pt)
