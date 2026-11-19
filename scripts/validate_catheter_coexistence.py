#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step both catheter packages in fresh processes with opposite import orders.

Invoke twice with the same initially empty WARP_CACHE_PATH:
  arena/.venv/bin/python scripts/validate_catheter_coexistence.py legacy-first
  arena/.venv/bin/python scripts/validate_catheter_coexistence.py numerical-first
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def legacy_fixture():
    import numpy as np
    import torch
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.cloner import ReplicateSession
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import NewtonManager

    from i4h_arena.medical.newton_catheter_physics import CatheterRodHandle, CatheterRodSpec, newton_physics_cfg

    spec = CatheterRodSpec(num_envs=1, device="cuda:0", num_segments=24)

    @configclass
    class HostCfg(DirectRLEnvCfg):
        seed = 0
        decimation = 2
        episode_length_s = 100.0
        action_space = 3
        observation_space = 1
        state_space = 0
        sim = SimulationCfg(dt=1 / 60, device="cuda:0", physics=newton_physics_cfg(spec))
        scene = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)

    class Host(DirectRLEnv):
        def _setup_scene(self):
            root = AssetBaseCfg(prim_path=self.scene.cloner_cfg.clone_template.format("[^/]+"), cloning_contexts=None)
            with ReplicateSession(
                [root], self.num_envs, self.cfg.scene.env_spacing, env_template=self.scene.cloner_cfg.clone_template
            ):
                pass
            self.handle = CatheterRodHandle(spec).install()

        def _pre_physics_step(self, actions):
            self.actions = actions

        def _apply_action(self):
            self.handle.rod.apply_proximal_control_gpu(self.actions[:, 0], self.actions[:, 1], self.physics_dt)

        def _get_observations(self):
            return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            return (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),) * 2

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            self.handle.reset(env_ids)

    env = Host(HostCfg())
    try:
        env.reset(seed=0)
        command = torch.zeros((1, 3), device=env.device)
        command[:, 0] = 0.1
        for _ in range(12):
            env.step(command)
            assert np.isfinite(NewtonManager.get_state_0().particle_q.numpy()).all()
            assert np.isfinite(NewtonManager.get_state_0().particle_qd.numpy()).all()
        print("PASS legacy catheter host: 12 control ticks, finite state", flush=True)
    finally:
        env.close()


def numerical_fixture():
    import numpy as np
    import torch
    from i4h_isaaclab.env import I4hEnv, I4hEnvCfg

    env = I4hEnv(I4hEnvCfg(case="aorta_static", stagger_episodes=False, seed=0))
    try:
        env.reset(seed=0)
        command = torch.zeros(env.action_space.shape, device=env.device)
        for _ in range(12):
            env.step(command)
            assert np.isfinite(env.simulator.scene.q.numpy()).all()
            assert np.isfinite(env.simulator.scene.qd.numpy()).all()
        print("PASS numerical aorta_static host: 12 control ticks, finite state", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order", choices=("legacy-first", "numerical-first"))
    args = parser.parse_args()
    os.environ.setdefault("I4H_DATA", str(ROOT / "third_party/i4h-numerical-sim-d1fc95c/numerical_sim/data"))
    names = ["catheter_vasculature_solver", "i4h_physics", "i4h_endoluminal"]
    if args.order == "numerical-first":
        names = ["i4h_physics", "i4h_endoluminal", "catheter_vasculature_solver"]
    import faulthandler

    faulthandler.dump_traceback_later(180, repeat=True)
    modules = {name: importlib.import_module(name) for name in names}
    for name, module in modules.items():
        checkout = (
            "i4h-physics-simulation-internal" if name == "catheter_vasculature_solver" else "i4h-numerical-sim-d1fc95c"
        )
        assert Path(module.__file__).resolve().is_relative_to(ROOT / "third_party" / checkout), module.__file__
    import newton
    import warp as wp
    from isaaclab.sim import SimulationContext

    print(
        json.dumps(
            {
                "order": args.order,
                "packages": {name: module.__file__ for name, module in modules.items()},
                "newton": newton.__file__,
                "warp": wp.__file__,
                "cache": os.environ.get("WARP_CACHE_PATH"),
            }
        ),
        flush=True,
    )
    fixtures = [legacy_fixture, numerical_fixture]
    if args.order == "numerical-first":
        fixtures.reverse()
    for fixture in fixtures:
        fixture()
        assert importlib.import_module("newton") is newton
        assert importlib.import_module("warp") is wp
        gc.collect()
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
    wp.synchronize()
    faulthandler.cancel_dump_traceback_later()
    print(f"COMPLETE {args.order}", flush=True)


if __name__ == "__main__":
    main()
