#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Masked ordinary reset and continuation restore with live acquired bindings."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--worlds", type=int, choices=(1, 8, 16), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["I4H_LIVER_OBJECTIVE"] = "grasped_retract"
    from i4h_arena.scenes.base import load_scene
    from i4h_rl.liver_host import build_parser, launch_app

    cli = build_parser().parse_args(
        [
            "--workflow",
            "liver_reach_retract",
            "--mode",
            "idle",
            "--envs",
            str(args.worlds),
            "--seed",
            "7341",
            "--headless",
        ]
    )
    scene = load_scene("liver_reach_retract", cli, register_assets=False)
    scene.configure_args(cli)
    with launch_app(cli) as app:
        import torch

        env = app.make_env(scene)
        with torch.inference_mode():
            env.reset()
            runtime = env.liver
            assert bool(torch.all(runtime.views["latched"] == 1))
            assert bool(torch.all(runtime.views["counts"][:, 1] > 0))
            assert bool(torch.all(runtime.views["mechanism_counts"][:, 3] > 0))
            assert not env.recorder_manager.active_terms
            selected = args.worlds // 2
            ids = torch.tensor([selected], device=env.device)

            def compare_peers(before, after):
                peers = np.arange(args.worlds) != selected
                for family in ("probe", "task", "rng"):
                    for name, value in before[family].items():
                        np.testing.assert_array_equal(
                            value[peers], after[family][name][peers], err_msg=f"{family}/{name}"
                        )
                for owner, key, world, *_ in env.i4h.physics.solver.continuation.entries:
                    name = f"{owner.name}:{key}"
                    peers = world.numpy() != selected
                    np.testing.assert_array_equal(
                        before["numerical"][name][peers], after["numerical"][name][peers], err_msg=name
                    )

            before = runtime.capture_state()
            env.reset(env_ids=ids)
            compare_peers(before, runtime.capture_state())
            before = runtime.capture_state()
            env.liver_ordinary_reset = True
            env.reset(env_ids=ids)
            env.liver_ordinary_reset = False
            compare_peers(before, runtime.capture_state())
            assert int(runtime.views["latched"][selected]) == 0
            runtime.restore_complete_state(before, ids)
            after = runtime.capture_state()
            for family in ("probe", "task", "rng"):
                for name, value in before[family].items():
                    np.testing.assert_array_equal(value, after[family][name])
            for name, value in before["numerical"].items():
                np.testing.assert_array_equal(value, after["numerical"][name])
            invalid = {**before, "identity": before["identity"].copy()}
            invalid["identity"][0, 0] ^= 1
            try:
                runtime.restore_complete_state(invalid, ids)
            except ValueError:
                pass
            else:
                raise AssertionError("incompatible bundle accepted")
            for name, value in before["numerical"].items():
                np.testing.assert_array_equal(value, runtime.capture_state()["numerical"][name])
            report = dict(
                worlds=args.worlds,
                selected_world=selected,
                active_grasps=True,
                masked_restore_isolation=True,
                ordinary_reset_isolation=True,
                exact_complete_round_trip=True,
                incompatible_bundle_rejected=True,
                training_recorders_disabled=True,
                original_patch_sizes=runtime.views["counts"][:, 1].tolist(),
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.close()


if __name__ == "__main__":
    main()
