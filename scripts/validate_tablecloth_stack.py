#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU cloth gate. Run each workflow in a fresh process using arena/.venv/bin/python.

Example: arena/.venv/bin/python scripts/validate_tablecloth_stack.py spread_tablecloth_g1
"""

from __future__ import annotations

import argparse
import json
import math


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=("spread_tablecloth_g1", "spread_tablecloth_h2"))
    selected = parser.parse_args()

    from i4h_arena.app import launch_app
    from i4h_arena.cli import build_parser
    from i4h_arena.scenes.base import load_scene

    args = build_parser().parse_args(["--workflow", selected.workflow, "--mode", "idle", "--envs", "1"])
    scene = load_scene(
        f"{selected.workflow.removeprefix('spread_tablecloth_')}_tablecloth", args, register_assets=False
    )
    scene.configure_args(args)
    assert not args.xr, "Cloth GPU smoke must not enable XR"
    with launch_app(args) as app:
        import numpy as np
        import torch
        from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager as Manager

        env = app.make_env(scene)
        try:
            print("cloth smoke: initial reset", flush=True)
            env.reset(seed=0)
            print("cloth smoke: reset complete", flush=True)
            act = scene.make_actuation(env)
            model = Manager.get_model()
            assert model.surface_count == env.num_envs, (model.surface_count, env.num_envs, model.surface_label)
            assert np.all(np.bincount(model.surface_world.numpy(), minlength=env.num_envs) == 1)
            assert np.all(model.surface_particle_end.numpy() > model.surface_particle_start.numpy())
            ticks = math.ceil(120 / env.cfg.decimation)
            with torch.inference_mode():
                print(f"cloth smoke: stepping {ticks} control ticks", flush=True)
                for _ in range(ticks):
                    env.step(act.tensor())
                    assert np.isfinite(Manager.get_state_0().particle_q.numpy()).all()
                env.reset(seed=0)
                assert np.isfinite(Manager.get_state_0().particle_q.numpy()).all()
                env.step(scene.make_actuation(env).tensor())
                assert np.isfinite(Manager.get_state_0().particle_q.numpy()).all()
            print(
                json.dumps(
                    {
                        "workflow": selected.workflow,
                        "surfaces": model.surface_count,
                        "control_ticks": ticks,
                        "physics_steps": ticks * env.cfg.decimation,
                        "reset": "passed",
                        "finite": True,
                    }
                ),
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
