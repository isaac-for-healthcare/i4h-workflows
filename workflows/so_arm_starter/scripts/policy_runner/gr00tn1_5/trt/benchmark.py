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

import argparse
import time
from functools import partial

import numpy as np
import torch
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from PIL import Image
from policy_runner.gr00tn1_5.trt.action_head_utils import action_head_pytorch_forward
from policy_runner.gr00tn1_5.trt.trt_model_forward import setup_tensorrt_engines


def compare_predictions(pred_tensorrt, pred_torch):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n{key}:")
        print(f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        l1_mean = l1_dist.mean().item()
        l1_max = l1_dist.max().item()
        print(f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} ' f'{l1_mean:.4f}/{l1_max:.4f}')
        torch_max = torch_tensor.max().item()
        tensorrt_max = tensorrt_tensor.max().item()
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} ' f'{torch_max:.4f}/{tensorrt_max:.4f}'
        )
        torch_mean = torch_tensor.mean().item()
        tensorrt_mean = tensorrt_tensor.mean().item()
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} '
            f'{torch_mean:.4f}/{tensorrt_mean:.4f}'
        )
        torch_min = torch_tensor.min().item()
        tensorrt_min = tensorrt_tensor.min().item()
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} ' f'{torch_min:.4f}/{tensorrt_min:.4f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GR00T inference")
    parser.add_argument("--ckpt_path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to the GR00T model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default="",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["pytorch", "tensorrt", "compare"],
        default="pytorch",
        help="Inference mode: 'pytorch' for PyTorch inference, 'tensorrt' for TensorRT inference, "
        "'compare' for comparing PyTorch and TensorRT outputs similarity",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps",
        default=4,
    )
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        help="Path to the TensorRT engine",
        default="gr00t_engine",
    )
    parser.add_argument(
        "--num_warmup_runs",
        type=int,
        help="Number of warmup runs for TensorRT inference",
        default=3,
    )
    parser.add_argument(
        "--num_profile_runs",
        type=int,
        help="Number of profiling runs for TensorRT inference",
        default=50,
    )

    args = parser.parse_args()

    MODEL_PATH = args.ckpt_path
    EMBODIMENT_TAG = "new_embodiment"
    dataset_path = args.dataset_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = DATA_CONFIG_MAP["so100_dualcam"]
    data_config.video_keys = ["video.room", "video.wrist"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    modality_config = policy.modality_config

    if dataset_path == "":
        room_img_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        room_img = Image.fromarray(room_img_array, "RGB")
        wrist_img_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        wrist_img = Image.fromarray(wrist_img_array, "RGB")
        arm = np.random.randn(5)
        gripper = np.random.randn(1)

        step_data = {
            "video.room": np.expand_dims(room_img, axis=0),
            "video.wrist": np.expand_dims(wrist_img, axis=0),
            "state.single_arm": np.expand_dims(np.array(arm), axis=0),
            "state.gripper": np.expand_dims(np.array(gripper), axis=0),
            "annotation.human.task_description": "Grip the scissors and put it into the tray",
        }
    else:
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            video_backend="torchvision_av",
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=EMBODIMENT_TAG,
        )

        step_data = dataset[0]

    if args.inference_mode == "pytorch":
        # Warmup runs
        print(f"Performing {args.num_warmup_runs} warmup runs...")
        for i in range(args.num_warmup_runs):
            _ = policy.get_action(step_data)

        # Timed inference runs
        num_runs = args.num_profile_runs
        inference_times = []

        print(f"\nRunning {num_runs} timed inference runs...")
        for i in range(num_runs):
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            start_time = time.perf_counter()

            predicted_action = policy.get_action(step_data)

            torch.cuda.synchronize()  # Ensure GPU operations are complete
            end_time = time.perf_counter()

            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)
            print(f"Run {i+1}: {inference_time:.2f} ms")

        # Calculate statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)

        print("\n=== PyTorch Inference Performance ===")
        print(f"Number of runs: {num_runs}")
        print(f"Mean inference time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Min inference time: {min_time:.2f} ms")
        print(f"Max inference time: {max_time:.2f} ms")
        print(f"Throughput: {1000/mean_time:.2f} inferences/second")

        print("\n=== PyTorch Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)

    elif args.inference_mode == "tensorrt":
        # Setup TensorRT engines
        setup_tensorrt_engines(policy, args.trt_engine_path)

        # Warmup runs (TensorRT engines may need warmup for optimal performance)
        print(f"Performing {args.num_warmup_runs} warmup runs...")
        for i in range(args.num_warmup_runs):
            _ = policy.get_action(step_data)

        # Timed inference runs
        num_runs = args.num_profile_runs
        inference_times = []

        print(f"\nRunning {num_runs} timed inference runs...")
        for i in range(num_runs):
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            start_time = time.perf_counter()

            predicted_action = policy.get_action(step_data)

            torch.cuda.synchronize()  # Ensure GPU operations are complete
            end_time = time.perf_counter()

            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)
            print(f"Run {i+1}: {inference_time:.2f} ms")

        # Calculate statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)

        print("\n=== TensorRT Inference Performance ===")
        print(f"Number of runs: {num_runs}")
        print(f"Mean inference time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Min inference time: {min_time:.2f} ms")
        print(f"Max inference time: {max_time:.2f} ms")
        print(f"Throughput: {1000/mean_time:.2f} inferences/second")

        print("\n=== TensorRT Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)

    else:
        # ensure PyTorch and TensorRT have the same init_actions
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                dtype=torch.float16,
                device=device,
            )
        # PyTorch inference
        policy.model.action_head.get_action = partial(action_head_pytorch_forward, policy.model.action_head)
        predicted_action_torch = policy.get_action(step_data)

        # Setup TensorRT engines and run inference
        setup_tensorrt_engines(policy, args.trt_engine_path)
        predicted_action_tensorrt = policy.get_action(step_data)

        # Compare predictions
        compare_predictions(predicted_action_tensorrt, predicted_action_torch)
