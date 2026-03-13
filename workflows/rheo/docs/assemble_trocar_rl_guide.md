# GR00T online RL Post-Training Tutorial

This tutorial explains how to run GR00T online RL post-training on IsaacLab tasks in i4h.

## Quick Start: Run Assemble-Trocar Training

### 1. Build and run the Workflow Docker container

From repo root (host):

```bash
# -g1.5: install GR00TN1.5 dependencies
# -r: rebuild the image
bash workflows/rheo/docker/run_docker.sh -g1.5 -r
```

### 2. Run training

```bash
# Basic training
bash /workspaces/workflows/rheo/scripts/simulation/rl/train_gr00t_assemble_trocar.sh train \
    --model_path /models/<your_gr00t_checkpoint>

# With custom environment count
bash /workspaces/workflows/rheo/scripts/simulation/rl/train_gr00t_assemble_trocar.sh train \
    --model_path /models/<your_gr00t_checkpoint> \
    env.train.total_num_envs=32 env.eval.total_num_envs=4

# Resume from checkpoint
bash /workspaces/workflows/rheo/scripts/simulation/rl/train_gr00t_assemble_trocar.sh train \
    --model_path /models/<your_gr00t_checkpoint> \
    runner.resume_dir=/path/to/checkpoint
```

> **Low Memory Tip:** If you encounter out-of-memory (OOM) errors, try reducing `total_num_envs` and `micro_batch_size`:
>
> ```bash
> bash /workspaces/workflows/rheo/scripts/simulation/rl/train_gr00t_assemble_trocar.sh train \
>     --model_path /models/<your_gr00t_checkpoint> \
>     env.train.total_num_envs=8 actor.micro_batch_size=2
> ```

### 3. Run evaluation

```bash
# Evaluate base GR00T model
bash /workspaces/workflows/rheo/scripts/simulation/rl/train_gr00t_assemble_trocar.sh eval \
    --model_path /models/<your_gr00t_checkpoint>

# Evaluate RL-trained checkpoint (load RL checkpoint directly via model_path)
bash /workspaces/workflows/rheo/scripts/simulation/rl/train_gr00t_assemble_trocar.sh eval \
    --model_path /path/to/rl_ckpt
```

### 4. Output location

Results are saved to:

- `workflows/rheo/scripts/simulation/rl/results/gr00t_assemble_trocar/{train,eval}_YYYYMMDD-HHMMSS/`
- Logs: `{mode}.log`
- TensorBoard: `tensorboard/`
- Videos (if enabled): `video/`

---

## Benchmarks: Assemble Trocar Task

### Task Overview

The Assemble Trocar task is a challenging multi-stage surgical tool assembly task that requires precise bimanual manipulation. The task is decomposed into four progressive stages with curriculum-based RL training applied to each stage.

![Infer Room GIF](https://developer.download.nvidia.com/assets/Clara/i4h/Rheo/infer_room_3x.gif)

### Task Stage Decomposition

| Stage | Description | Success Criteria |
| ------- | ------------- | ------------------ |
| **Stage 1: Lift the trocar** | Robot grasps and lifts both trocar components from the tray | Both trocars lifted above height threshold |
| **Stage 2: Align the tip** | Robot aligns the tips of the two trocar components | Distance between trocar tips < threshold |
| **Stage 3: Insert the trocar** | Robot inserts one trocar component into the other | Trocars nearly parallel AND center distance < threshold |
| **Stage 4: Place the trocar** | Robot places the assembled trocar into the mayo stand | Both trocars inside the mayo stand region |

### Performance Results

**Evaluation Setup:**

- 100 randomly generated scenes
- Box rotation angles: 0° to 10°
- Success measured at cumulative stage completion

| Model | Stage 1 | Stage 1+2 | Stage 1+2+3 | Stage 1+2+3+4 |
| ------- | --------- | ----------- | ------------- | --------------- |
| **Base Model (SFT)** | 83% | 72% | 32% | 29% |
| **RL Post-Training Stage 1** | **100%** *(+8%)* | - | - | - |
| **RL Post-Training Stage 2** | - | **92%** *(+9%)* | - | - |
| **RL Post-Training Stage 3** | - | - | **85%** *(+53%)* | - |
| **RL Post-Training Stage 4** | - | - | - | **82%** *(+53%)* |

### Key Findings

1. **Curriculum Learning Effectiveness**: Progressive RL training on individual stages leads to significant performance improvements across all stages.

2. **Stage 3-4 Breakthrough**: The most challenging stages (insertion and placement) show dramatic **+53% improvement** with RL post-training, demonstrating the effectiveness of RL for precision manipulation tasks.

3. **Stage 1-2 Refinement**: Early stages achieve **+8-9% improvements** with RL post-training, with Stage 1 reaching **100% success rate**.

### Training Configuration

**Environment Setup:**

- **Parallel Environments**: 512 (training), 128 (evaluation)
- **Episode Length**: 256 steps per episode
- **Rollout Steps**: 8 steps per rollout epoch
- **Observation Randomization**: Box rotation (0°-10°), initial trocar poses

**RL Algorithm:**

- **Policy**: GR00T N1.5 with value head
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Batch Size**: 1024 global batch size
- **Learning Rate**: 5e-6 (policy), 1e-4 (value)
- **Training Backend**: FSDP (Fully Sharded Data Parallel)

**Reward Shaping:**

- Stage-specific sparse rewards for sub-task completion
- Curriculum progression: train each stage independently
- Episode termination on stage success or max steps

## Adding Your Own Task

To add a new task for GR00T RL post-training, you need to create/modify files in the following locations:

```text
workflows/rheo/scripts/
├── simulation/
│   ├── tasks/<your_task>/           # 1. IsaacLab task definition
│   │   ├── __init__.py              #    gym.register() calls
│   │   └── <your_task>_env_cfg.py   #    Environment config
│   └── rl/
│       ├── rlinf_ext/
│       │   ├── __init__.py          # 2. Register task + env wrapper
│       │   └── config/
│       │       ├── env/<your_task>.yaml  # 3. Hydra env config
│       │       ├── model/<your_model>.yaml   # 4. Hydra model config
│       │       └── isaaclab_ppo_gr00t_<your_task>.yaml  # 5. Main config
│       └── train_gr00t_<your_task>.sh   # 6. Launch script
└── policy/
    └── gr00t_config.py              # 7. GR00T data config (if new embodiment)
```

### Step 1: Create IsaacLab Task

Create `simulation/tasks/<your_task>/__init__.py`:

```python
import gymnasium as gym
from . import <your_task>_env_cfg

gym.register(
    id="Isaac-<YourTask>-G129-Dex3-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": <your_task>_env_cfg.YourTaskEnvCfg,
    },
    disable_env_checker=True,
)

# Optional: separate eval config with deterministic resets
gym.register(
    id="Isaac-<YourTask>-G129-Dex3-Joint-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": <your_task>_env_cfg.YourTaskEvalEnvCfg,
    },
    disable_env_checker=True,
)
```

### Step 2: Register Task in rlinf_ext

Edit `rlinf_ext/__init__.py`:

```python
def register() -> None:
    # ... existing code ...

    # 1) Import your task to trigger gym.register()
    import simulation.tasks.<your_task>  # noqa: F401

    # ... rest of registration ...


def _register_isaaclab_envs() -> None:
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
    from rlinf_ext.g129_dex3_env import get_env_class  # or your custom wrapper

    IsaaclabG129Dx3Env = get_env_class()

    # Add your task IDs
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-<YourTask>-G129-Dex3-Joint", IsaaclabG129Dx3Env)
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-<YourTask>-G129-Dex3-Joint-Eval", IsaaclabG129Dx3Env)
```

### Step 3: Create RLinf Env Wrapper (if needed)

If your task has different observation/action spaces, add a custom env class factory in `rlinf_ext/__init__.py`.

See `_get_g129_dex3_env_class()` in `rlinf_ext/__init__.py` as reference. Key methods to override:

- `_make_env_function()`: Creates the IsaacLab environment
- `_wrap_obs(obs)`: Converts IsaacLab observations to RLinf schema
- `add_image(obs)`: Creates video frames for logging

### Step 4: Create Hydra Env Config

Create `rlinf_ext/config/env/<your_task>.yaml`:

```yaml
env_type: isaaclab
total_num_envs: null
auto_reset: False
ignore_terminations: False
use_rel_reward: True
seed: 0
group_size: 1

reward_coef: 1.0
max_steps_per_rollout_epoch: 10
max_episode_steps: 256

video_cfg:
  save_video: False
  video_base_dir: ${runner.logger.log_path}/video/train

init_params:
  id: "Isaac-<YourTask>-G129-Dex3-Joint"
  num_envs: null
  max_episode_steps: ${env.train.max_episode_steps}
  task_description: "<description of your task>"
```

### Step 5: Create Main Hydra Config

Create `rlinf_ext/config/isaaclab_ppo_gr00t_<your_task>.yaml`:

```yaml
defaults:
  - /env@env.train: <your_task>
  - /env@env.eval: <your_task>
  - /model@actor.model: gr00t_dex3
  - /model@rollout.model: gr00t_dex3
  - _self_

# Override eval task ID
env:
  eval:
    init_params:
      id: "Isaac-<YourTask>-G129-Dex3-Joint-Eval"

# ... rest of config (see isaaclab_ppo_gr00t_assemble_trocar.yaml)
```

### Step 6: Create Launch Script

Create `simulation/rl/train_gr00t_<your_task>.sh` (copy and modify `train_gr00t_assemble_trocar.sh`):

```bash
CONFIG_NAME="isaaclab_ppo_gr00t_<your_task>"
# ... rest of script
```

### Step 7: GR00T Data Config (if new embodiment)

If your robot has different observation/action spaces than the existing `UnitreeG1SimDataConfig`, create a new data config in `policy/gr00t_config.py`:

```python
@dataclass
class YourEmbodimentDataConfig(ModalityConfig):
    # Define video modalities, state modalities, action modalities
    # See UnitreeG1SimDataConfig as reference
    pass
```

And register the obs/action converters in `rlinf_ext/__init__.py`.

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          RLinf (Ray)                                    │
│                                                                         │
│  ┌───────────┐     ┌────────────┐     ┌──────────────┐                  │
│  │Env Worker │     │Actor Worker│     │Rollout Worker│                  │
│  └─────┬─────┘     └─────┬──────┘     └─────┬────────┘                  │
│        │                 │                  │                           │
│        └─────────────────┼──────────────────┘                           │
│                          │                                              │
│                          ▼  Worker._load_user_extensions()              │
│                             (if RLINF_EXT_MODULE is set)                │
│                          │                                              │
│  ┌───────────────────────┼─────────────────────────────────────────┐    │
│  │         RLinf Global Registries (modified by rlinf_ext)         │    │
│  │  • REGISTER_ISAACLAB_ENVS["Isaac-Assemble-Trocar-..."]          │    │
│  │  • OBS_CONVERSION["dex3"], ACTION_CONVERSION["dex3"]            │    │
│  │  • EMBODIMENT_TAG_MAPPING["new_embodiment"]                     │    │
│  │  • get_model() patched for new_embodiment                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          ▲                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           │ imports & registers
┌──────────────────────────┴──────────────────────────────────────────────┐
│                    i4h Extensions (sim/rl/rlinf_ext/)                   │
│                                                                         │
│  __init__.py::register()                                                │
│  ├── _register_isaaclab_envs()  ──► uses _get_g129_dex3_env_class()     │
│  ├── _register_gr00t_converters()                                       │
│  ├── _register_gr00t_data_config() ──► imports policy/gr00t_config.py   │
│  └── _patch_gr00t_get_model()                                           │
│                                                                         │
│  Dependencies:                                                          │
│  ├── simulation/tasks/assemble_trocar/  (IsaacLab task + gym.register)  │
│  └── policy/gr00t_config.py      (UnitreeG1SimDataConfig)               │
└─────────────────────────────────────────────────────────────────────────┘
```
