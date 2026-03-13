# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Direct evaluation of Gr00t policy in IsaacLab simulation
"""

import argparse
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Evaluate Gr00t Policy in IsaacLab")
parser.add_argument("--task", type=str, default="Isaac-Assemble-Trocar-G129-Dex3-Joint", help="task name")
parser.add_argument("--model_path", type=str, default=None, help="path to Gr00t model checkpoint")
parser.add_argument("--num_episodes", type=int, default=10, help="number of evaluation episodes")
parser.add_argument("--max_steps", type=int, default=256, help="max steps per episode")
parser.add_argument("--seed", type=int, default=4, help="random seed")
parser.add_argument("--save_video", action="store_true", help="save video of evaluation")
parser.add_argument("--video_dir", type=str, default="./eval_videos", help="directory to save videos")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="number of parallel simulation environments",
)
parser.add_argument(
    "--video_env_id",
    type=int,
    default=0,
    help="which env index to record when --save_video is set (records only one env even if num_envs>1)",
)
parser.add_argument(
    "--save_video_all_envs",
    action="store_true",
    help="when num_envs>1 and --save_video is set, save videos for ALL envs (creates num_envs * 4 mp4 files)",
)
parser.add_argument(
    "--action_chunk_size", type=int, default=1, help="number of actions to use from action chunk (1-16, default: 1)"
)
parser.add_argument(
    "--frequency",
    type=float,
    default=0.0,
    help="control frequency (Hz) for stepping actions; emulates real-time control loop",
)
parser.add_argument(
    "--success_stage",
    type=int,
    default=4,
    help="success stage for the task",
)
# Note: Using "install trocar from box" as task description yields better results than "assemble trocar from tray"
# because the model was trained with this specific task description during data collection and fine-tuning.
parser.add_argument("--task_description", type=str, default="install trocar from box", help="task description")
parser.add_argument(
    "--rl_ckpt",
    action="store_true",
    help="use Gr00tRLPolicy (with eagle-input padding and dropout removal) for RL-trained checkpoints",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="run a lightweight integration test with a dummy policy (still starts sim + env)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from simulation.examples.utils import _MultiViewConcatWriter, evaluate_episode, set_viewport_camera  # noqa: E402
from simulation.tasks import assemble_trocar  # noqa: F401


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("Gr00t Policy Direct Evaluation in IsaacLab")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Model: {args_cli.model_path or '<test>'}")
    print("=" * 60)

    test_mode = bool(getattr(args_cli, "test", False))
    if test_mode:
        print("Test mode enabled (dummy policy, no checkpoint needed)")
    elif not args_cli.model_path:
        print("--model_path is required unless --test is set")
        return

    # Parse environment configuration
    print("\n[1/4] Loading environment configuration...")
    num_envs = int(getattr(args_cli, "num_envs", 1) or 1)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed

    # Create environment
    print("\n[2/4] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)

    set_viewport_camera("/World/envs/env_0/Robot/d435_link/front_cam")

    # Load Gr00t policy
    print("\n[3/4] Loading Gr00t policy...")
    if test_mode:
        import numpy as _np

        class _DummyPolicy:
            def __init__(self, device: str):
                self.device = device

            def get_action(self, _obs):  # noqa: ANN001
                # One chunk of 16 actions; each action is a 43-D vector.
                return {"actions": _np.zeros((16, 43), dtype=_np.float32)}

        policy = _DummyPolicy(args_cli.device)
        print("Dummy policy ready")
    else:
        from policy.gr00t_config import DATA_CONFIG_MAP

        data_config = DATA_CONFIG_MAP["unitree_g1_sim"]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        rl_mode = bool(getattr(args_cli, "rl_ckpt", False))
        if rl_mode:
            from policy.apply_gr00t_rl_patch import apply_patch

            apply_patch()
            print("Applied GR00T RL patch (see tools/env_setup/patches/gr00t_policy_padding_dropout.patch)")

        from gr00t.model.policy import Gr00tPolicy

        policy = Gr00tPolicy(
            model_path=args_cli.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag="new_embodiment",
            device=args_cli.device,
        )
        print("Gr00t policy loaded successfully")

    # Run evaluation
    print("\n[4/4] Running evaluation...")
    print("=" * 60)

    results = []

    video_writer: _MultiViewConcatWriter | None = None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args_cli.save_video:
        model_name_for_file = "test" if test_mode else Path(args_cli.model_path).stem
        base_name = f"{timestamp}_{model_name_for_file}"
        video_writer = _MultiViewConcatWriter(args_cli.video_dir, base_name=base_name, fps=30)

    results = evaluate_episode(
        env,
        policy,
        max_steps=args_cli.max_steps,
        num_episodes=args_cli.num_episodes,
        save_video=args_cli.save_video,
        action_chunk_size=args_cli.action_chunk_size,
        frequency_hz=float(getattr(args_cli, "frequency", 0.0) or 0.0),
        task_description=args_cli.task_description,
        success_stage=args_cli.success_stage,
        video_writer=video_writer,
        video_env_id=int(getattr(args_cli, "video_env_id", 0) or 0),
        save_video_all_envs=bool(getattr(args_cli, "save_video_all_envs", False)),
    )

    # Calculate statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    successes = sum(1 for r in results if r["success"])
    success_rate = successes / len(results) * 100

    success_steps = [r["steps"] for r in results if r["success"]]
    avg_success_steps = np.mean(success_steps) if success_steps else 0

    print("\nOverall Statistics:")
    print(f"  Total Episodes: {len(results)}")
    print(f"  Success Number: {successes}")
    print(f"  Success Rate: {success_rate:.1f}%")
    if success_steps:
        print(f"  Average Steps (Success): {avg_success_steps:.1f}")

    print("\nEpisode-by-Episode Results:")
    for r in results:
        status = "SUCCESS" if r["success"] else "FAILED"
        print(f"  Ep {r['episode']+1:2d}: {status} | Steps: {r['steps']:4d} | Reward: {r['total_reward']:7.2f}")

    # Save results to file
    results_file = Path("./eval_results") / f"results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Task: {args_cli.task}\n")
        f.write(f"Model: {args_cli.model_path}\n")
        f.write(f"Episodes: {args_cli.num_episodes}\n")
        f.write(f"Action Chunk Size: {args_cli.action_chunk_size}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        f.write(f"Average Success Steps: {avg_success_steps:.1f}\n")
        f.write("\nDetailed Results:\n")
        for r in results:
            status = "SUCCESS" if r["success"] else "FAILED"
            stage = r.get("final_stage", -1)
            f.write(
                f"  Episode {r['episode'] + 1}: {status}"
                f" | Stage: {stage}/4"
                f" | Steps: {r['steps']}"
                f" | Reward: {r['total_reward']:.3f}\n"
            )

    print(f"\nResults saved to: {results_file}")

    if args_cli.save_video and video_writer is not None:
        video_writer.close()
        print(f"\nVideo saved to: {args_cli.video_dir}/{base_name}_*.mp4")

    print("\nCleaning up...")

    # Revert the GR00T RL patch if it was applied
    if not test_mode and bool(getattr(args_cli, "rl_ckpt", False)):
        from policy.apply_gr00t_rl_patch import revert_patch

        revert_patch()
        print("Reverted GR00T RL patch")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
