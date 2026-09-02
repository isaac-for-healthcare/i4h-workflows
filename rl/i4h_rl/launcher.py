# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heavy RLinf launcher. Imported only after the lightweight preflight passes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    os.environ.setdefault("RLINF_EXT_MODULE", "i4h_rl.extension")
    os.environ["RLINF_CONFIG_FILE"] = str(args.config.resolve())

    import torch.multiprocessing as mp
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, open_dict
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    mp.set_start_method("spawn", force=True)
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(args.config.resolve().parent), version_base="1.3")
    cfg = compose(config_name=args.config.stem, overrides=args.overrides)
    with open_dict(cfg):
        cfg.runner.logger.log_path = str(args.run_dir.resolve())
        cfg.runner.max_epochs = args.max_epochs
        cfg.runner.only_eval = args.only_eval
        cfg.env.train.total_num_envs = args.num_envs
        cfg.env.eval.total_num_envs = args.num_envs
        cfg.actor.model.model_path = str(args.model_path.resolve())
        cfg.rollout.model.model_path = str(args.model_path.resolve())
        if args.only_eval:
            # Training rollouts inherit actor.model, but an embodied-eval
            # worker reads rollout.model directly. Merge the compact rollout
            # overrides onto the complete GR00T contract before validation.
            cfg.runner.task_type = "embodied_eval"
            cfg.rollout.model = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
        if args.resume_dir:
            cfg.runner.resume_dir = str(args.resume_dir.resolve())
        if args.rl_model_path:
            cfg.rollout.model.rl_model_path = str(args.rl_model_path.resolve())
        if args.video:
            cfg.env.eval.video_cfg.save_video = True
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
