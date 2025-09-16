# SO-ARM Starter Policy Runner

This script allows running policy models using DDS for communication, suitable for simulation robot control.

## Supported Policies

*   **GR00T N1.5**: NVIDIA's foundation model for humanoid robots. Refer to [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) for more information.

## Run GR00T N1.5 Policy with DDS communication

### Prepare Model Weights and Dependencies

Please refer to the [Environment Setup](../../README.md#setup-environment)to install GR00T N1.5 dependencies.
For acquiring model weights and further details, consult the official [NVIDIA Isaac GR00T Installation Guide](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#installation-guide).
Ensure the environment where you run `run_policy.py` has the GR00 TN1.5 dependencies installed.

### Ensure the PYTHONPATH Is Set

Please refer to the [Environment Setup - Set environment variables before running the scripts](../../README.md#environment-variables) instructions.

### Run Policy

The policy runner supports two inference modes, both PyTorch and TensorRT:

#### PyTorch Inference
```sh
python -m policy_runner.run_policy  --ckpt_path <path_to_your_checkpoint>
```

#### TensorRT Inference

**Requirements:** TensorRT inference requires pre-built engine

- Convert PyTorch model to TensorRT engines

Export model to ONNX format
```sh
python -m policy_runner.gr00tn1_5.trt.export_onnx --ckpt_path <path_to_your_checkpoint>
```

Build TensorRT engines
```sh
bash policy_runner/gr00tn1_5/trt/build_engine.sh
```

You can choose to use Docker to build TensorRT engines.
```sh
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:25.06-py3 \
  bash policy_runner/gr00tn1_5/trt/build_engine.sh
```

- Run inference with TensorRT engines
```sh
python -m policy_runner.run_policy \
  --ckpt_path <path_to_your_checkpoint> \
  --trt \
  --trt_engine_path <path_to_tensorrt_engine_directory>
```

## Command Line Arguments

Here's a markdown table describing the command-line arguments:

| Argument                  | Description                                                              | Default Value                      |
|---------------------------|--------------------------------------------------------------------------|------------------------------------|
| `--ckpt_path`             | Checkpoint path for the policy model.                                    | Uses downloaded assets             |
| `--task_description`      | Task description text prompt for the policy.                             | `Grip the scissors and put it into the tray`  |
| `--chunk_length`          | Length of the action chunk inferred by the policy per inference step.    | 16                                 |
| `--data_config`           | Data config name (used for GR00T N1.5).                                  | `so100_dualcam`                  |
| `--embodiment_tag`        | The embodiment tag for the model (used for GR00T N1.5).                  | `new_embodiment`                   |
| `--trt`                   | Enable TensorRT engine for accelerated inference.                         | False                              |
| `--trt_engine_path`       | Path to the TensorRT engine files directory.                             | `gr00t_engine`                     |
| `--rti_license_file`      | Path to the RTI license file.                                            | Uses env `RTI_LICENSE_FILE`        |
| `--domain_id`             | Domain ID for DDS communication.                                         | 0                                  |
| `--height`                | Input image height for cameras.                                          | 480                                |
| `--width`                 | Input image width for cameras.                                           | 640                                |
| `--topic_in_room_camera`  | Topic name to consume room camera RGB data.                              | `topic_room_camera_data_rgb`       |
| `--topic_in_wrist_camera` | Topic name to consume wrist camera RGB data.                             | `topic_wrist_camera_data_rgb`      |
| `--topic_in_soarm_pos `   | Topic name to consume SOARM101 position data.                            | `topic_soarm_info`                 |
| `--topic_out`             | Topic name to publish generated SOARM101 actions.                        | `topic_soarm_ctrl`                 |
| `--verbose`               | Whether to print DDS communication logs.                                 | False                              |

## Performance Metrics

### Benchmark
- Runtime
```sh
python -m policy_runner.gr00tn1_5.trt.benchmark \
   --ckpt_path=<path_to_checkpoint>
   --inference_mode=<tensorrt_or_pytorch>
```

- Accuracy
```sh
python -m policy_runner.gr00tn1_5.trt.benchmark \
   --ckpt_path=<path_to_checkpoint>
   --inference_mode=compare
```

### Performance Results

| Hardware            | Inference Mode | Average Latency | Actions Predicted |
|---------------------|----------------|-----------------|-------------------|
| NVIDIA RTX 6000 Ada | PyTorch        | 42.16 ± 0.81 ms |        16         |
| NVIDIA RTX 6000 Ada | TensorRT       | 26.96 ± 1.86 ms |        16         |
