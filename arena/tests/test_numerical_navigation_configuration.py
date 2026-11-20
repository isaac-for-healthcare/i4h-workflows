# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Navigation's rigid configuration without loading Kit on the CPU."""

import ast
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_numerical_arm_preserves_implicit_integration_and_two_way_coupling(monkeypatch):
    source = Path(__file__).parents[1] / "i4h_arena/envcfg/endoluminal_navigation/numerical.py"
    tree = ast.parse(source.read_text())
    embodiment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NumericalCatheterEmbodiment"
    )
    configure = next(
        node for node in embodiment.body if isinstance(node, ast.FunctionDef) and node.name == "tune_env_cfg"
    )
    physics = ModuleType("isaaclab_newton.physics")
    # Match the newer Lab default that triggered unstable arm and finger motion.
    physics.MJWarpSolverCfg = lambda integrator="euler": SimpleNamespace(integrator=integrator)
    physics.NewtonCfg = SimpleNamespace
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", physics)
    component = SimpleNamespace(cfg=SimpleNamespace(coupling_mode="one_way"))

    def bind(sim, *, rigid):
        assert rigid
        component.rigid_solver = sim.physics.solver_cfg

    component.configure = bind
    namespace = {"copy": copy, "I4hComponent": lambda cfg: component, "component_configuration": lambda spec: spec}
    exec(compile(ast.Module(body=[configure], type_ignores=[]), str(source), "exec"), namespace)
    spec = SimpleNamespace(rigid_bodies_enabled=True, drive_body_name="panda_hand", drive_mount_local=(0, 0, 0.1034))
    seeded = []
    owner = SimpleNamespace(rod_spec=spec, _seed_the_arm_pose_on_reset=seeded.append)
    cfg = SimpleNamespace(sim=SimpleNamespace(), scene=SimpleNamespace())

    namespace["tune_env_cfg"](owner, cfg)

    assert component.rigid_solver.integrator == "implicitfast"
    assert component.cfg.coupling_mode == "two_way"
    assert component.cfg.mounts["catheter"]["body"] == "panda_hand"
    assert seeded == [cfg]
    assert cfg.sim.dt == 1 / 120 and cfg.decimation == 4
