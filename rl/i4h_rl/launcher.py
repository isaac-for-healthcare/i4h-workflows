# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heavy RLinf launcher. Imported only after the lightweight preflight passes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from i4h_rl.composition import compose_config, configure_launch, write_runtime_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Workflow RLinf training/evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--rl-model-path", type=Path)
    parser.add_argument("--only-eval", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = compose_config(args.config, args.overrides)
    configure_launch(
        cfg,
        model_path=args.model_path,
        run_dir=args.run_dir,
        num_envs=args.num_envs,
        max_epochs=args.max_epochs,
        only_eval=args.only_eval,
        video=args.video,
        resume_dir=args.resume_dir,
        rl_model_path=args.rl_model_path,
    )
    os.environ.setdefault("RLINF_EXT_MODULE", "i4h_rl.extension")
    os.environ["RLINF_CONFIG_FILE"] = str(write_runtime_config(cfg, args.run_dir))
    # Ray workers inherit this and use it for eagerly-flushed rollout
    # monitoring before the main RLinf runner reaches its optimizer logger.
    os.environ["I4H_RL_RUN_DIR"] = str(args.run_dir.resolve())

    import torch.multiprocessing as mp
    from omegaconf import OmegaConf
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    mp.set_start_method("spawn", force=True)
    cfg = validate_cfg(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    if args.only_eval:
        runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
    else:
        actor = EmbodiedFSDPActor.create_group(cfg).launch(
            cluster,
            name=cfg.actor.group_name,
            placement_strategy=placement.get_strategy("actor"),
        )
        runner = EmbodiedRunner(cfg=cfg, actor=actor, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
