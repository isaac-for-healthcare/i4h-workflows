# Cosmos-transfer1 Integration

[Cosmos-Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1) is a world-to-world transfer model designed to bridge the perceptual divide between simulated and real-world environments.
We introduce a training-free guided generation method on top of Cosmos-Transfer1 to overcome unsatisfactory results on unseen healthcare simulation assets.
Directly applying Cosmos-Transfer with various control inputs results in unsatisfactory outputs for the human phantom and robotic arm (see bottom figure). In contrast, our guided generation method preserves the appearance of the phantom and robotic arm while generating diverse backgrounds.

<img src="../../../../../../docs/source/cosmos_transfer_result.png" width="512" height="600" />

This training-free guided generation approach by encoding simulation videos into the latent space and applying spatial masking to guide the generation process. The trade-off between realism and faithfulness can be controlled by adjusting the number of guided denoising steps. In addition, our generation pipeline supports multi-view video generation. We first leverage the camera information to warp the generated room view to wrist view, then use it as the guidance of wrist-view generation.

#### Download Cosmos-transfer1 Checkpoints
The cosmos-transfer1 dependency is already installed after completing the installation steps in [Quick Start](../../../../README.md#-quick-start) section. Please navigate to the third party `cosmos-transfer1` folder and run the following command to download the checkpoints:
```sh
# Ensure the current path is the project root directory
huggingface-cli login
pushd third_party/cosmos-transfer1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
popd
```

> **Note:** You need to be logged in to Hugging Face (`huggingface-cli login`) before running the download script. Additionally, for the `meta-llama/Llama-Guard-3-8B` model, you need to request additional access on the [Hugging Face model page](https://huggingface.co/meta-llama/Llama-Guard-3-8B) before you can download it.

<details>
<summary>Folder Structure of Cosmos-transfer1 Checkpoints</summary>

```sh
$ tree third_party/cosmos-transfer1/checkpoints/ -L 3
third_party/cosmos-transfer1/checkpoints/
├── depth-anything
│   └── Depth-Anything-V2-Small-hf
│       ├── config.json
│       ├── model.safetensors
│       ├── preprocessor_config.json
│       └── README.md
├── facebook
│   └── sam2-hiera-large
│       ├── config.json
│       ├── model.safetensors
│       ├── preprocessor_config.json
│       ├── processor_config.json
│       ├── README.md
│       ├── sam2_hiera_large.pt
│       ├── sam2_hiera_l.yaml
│       └── video_preprocessor_config.json
├── google-t5
│   └── t5-11b
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── README.md
│       ├── spiece.model
│       ├── tf_model.h5
│       └── tokenizer.json
├── IDEA-Research
│   └── grounding-dino-tiny
│       ├── added_tokens.json
│       ├── config.json
│       ├── model.safetensors
│       ├── preprocessor_config.json
│       ├── pytorch_model.bin
│       ├── README.md
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.txt
├── meta-llama
│   └── Llama-Guard-3-8B
│       ├── LICENSE
│       ├── original
│       └── README.md
├── nvidia
│   ├── Cosmos-Guardrail1
│   │   ├── blocklist
│   │   └── README.md
│   ├── Cosmos-Tokenize1-CV8x8x8-720p
│   │   ├── autoencoder.jit
│   │   ├── config.json
│   │   ├── decoder.jit
│   │   ├── encoder.jit
│   │   ├── image_mean_std.pt
│   │   ├── mean_std.pt
│   │   ├── model_config.yaml
│   │   ├── model.pt
│   │   └── README.md
│   ├── Cosmos-Transfer1-7B
│   │   ├── 4kupscaler_control.pt
│   │   ├── base_model.pt
│   │   ├── config.json
│   │   ├── depth_control.pt
│   │   ├── edge_control_distilled.pt
│   │   ├── edge_control.pt
│   │   ├── guardrail
│   │   ├── keypoint_control.pt
│   │   ├── README.md
│   │   ├── seg_control.pt
│   │   └── vis_control.pt
│   ├── Cosmos-Transfer1-7B-Sample-AV
│   │   ├── guardrail
│   │   └── README.md
│   └── Cosmos-UpsamplePrompt1-12B-Transfer
│       ├── consolidated.safetensors
│       ├── params.json
│       ├── README.md
│       ├── seg_upsampler_example.png
│       └── tekken.json
└── README.md

21 directories, 57 files
```

</details>

#### Video Prompt Generation
We follow the idea in [lucidsim](https://github.com/lucidsim/lucidsim) to first generate batches of meta prompt that contains a very concise description of the potential scene, then instruct the LLM (e.g., [gemma-3-27b-it](https://build.nvidia.com/google/gemma-3-27b-it)) to upsample the meta prompt with detailed descriptions.
We provide example prompts in [`generated_prompts_two_seperate_views.json`](./config/generated_prompts_two_seperate_views.json).

#### Prepare Source Data

Please follow the [instructions](../state_machine/README.md) to prepare the source data.

By default, the source data directory is `data/hdf5/<date-time>-Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0/` under the project root directory.

#### Running Cosmos-transfer1 + Guided Generation

```sh
# Ensure the current path is the project root directory
export PROJECT_ROOT=$(pwd)
export CHECKPOINT_DIR="${PROJECT_ROOT}/third_party/cosmos-transfer1/checkpoints"
export PYTHONPATH="${PROJECT_ROOT}/third_party/cosmos-transfer1:${PROJECT_ROOT}/workflows/robotic_ultrasound/scripts"
cd workflows/robotic_ultrasound/scripts/simulation/
CUDA_HOME=$CONDA_PREFIX torchrun \
    --nproc_per_node=1 --nnodes=1 \
    -m environments.cosmos_transfer1.transfer \
    --checkpoint_dir $CHECKPOINT_DIR \
    --source_data_dir "Path to source dir of h5 files" \
    --output_data_dir "Path to output dir of h5 files" \
    --offload_text_encoder_model
```
#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | str | "" | Prompt which the sampled video condition on |
| `--negative_prompt` | str | "The video captures a game playing, ..." | Negative prompt which the sampled video condition on |
| `--input_video_path` | str | "" | Optional input RGB video path |
| `--num_input_frames` | int | 1 | Number of conditional frames for long video generation |
| `--sigma_max` | float | 80 | sigma_max for partial denoising |
| `--blur_strength` | str | "medium" | Blur strength applied to input |
| `--canny_threshold` | str | "medium" | Canny threshold applied to input. Lower means less blur or more detected edges, which means higher fidelity to input |
| `--controlnet_specs` | str | "inference_cosmos_transfer1_two_views.json" | Path to JSON file specifying multicontrolnet configurations |
| `--checkpoint_dir` | str | "checkpoints" | Base directory containing model checkpoints |
| `--tokenizer_dir` | str | "Cosmos-Tokenize1-CV8x8x8-720p" | Tokenizer weights directory relative to checkpoint_dir |
| `--video_save_folder` | str | "outputs/" | Output folder for generating a batch of videos |
| `--num_steps` | int | 35 | Number of diffusion sampling steps |
| `--guidance` | float | 5.0 | Classifier-free guidance scale value |
| `--fps` | int | 30 | FPS of the output video |
| `--height` | int | 224 | Height of video to sample |
| `--width` | int | 224 | Width of video to sample |
| `--seed` | int | 1 | Random seed |
| `--num_gpus` | int | 1 | Number of GPUs used to run context parallel inference. |
| `--offload_diffusion_transformer` | bool | False | Offload DiT after inference |
| `--offload_text_encoder_model` | bool | False | Offload text encoder model after inference |
| `--offload_guardrail_models` | bool | True | Offload guardrail models after inference |
| `--upsample_prompt` | bool | False | Upsample prompt using Pixtral upsampler model |
| `--offload_prompt_upsampler` | bool | False | Offload prompt upsampler model after inference |
| `--source_data_dir` | str | "" | Path to source data directory for batch inference. It contains h5 files generated from the state machine. |
| `--output_data_dir` | str | "" | Path to output data directory for batch inference. |
| `--save_name_offset` | int | 0 | Offset for the video save name. |
| `--foreground_label` | str | "3,4" | Comma-separated list of labels used to define the foreground mask during guided generation. The foreground corresponds to the object whose appearance we want to keep unchanged during generation. |
| `--sigma_threshold` | float | 1.2866 | This controls how many guidance steps are performed during generation. Smaller values mean more steps, larger values mean less steps. |
| `--concat_video_second_view` | bool | True | Whether to concatenate the first and second view videos during generation |
| `--fill_missing_pixels` | bool | True | Whether to fill missing pixels in the warped second view video |
| `--model_config_file` | str | "environments/cosmos_transfer1/config/transfer/config.py" | Relative path to the model config file |
