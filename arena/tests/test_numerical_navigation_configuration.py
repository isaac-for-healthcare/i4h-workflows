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


def native_configuration():
    """Load the actual authoring functions without starting a Kit application."""
    import math

    import numpy as np
    import pytest

    pytest.importorskip("i4h_physics")
    from i4h_isaaclab.component import I4hComponentCfg

    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    source = Path(__file__).parents[1] / "i4h_arena/envcfg/endoluminal_navigation/numerical.py"
    functions = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)]
    namespace = dict(np=np, math=math, I4hComponentCfg=I4hComponentCfg, rod_solver_cfg=rod_solver_cfg)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["component_configuration"]


def test_native_free_and_mounted_cases_build_with_upstream_apis():
    import numpy as np

    configure = native_configuration()
    from i4h_host.composition import load_case

    from i4h_arena.medical.newton_catheter_physics import CatheterRodSpec

    for mounted in (False, True):
        spec = CatheterRodSpec(device="cpu", num_envs=2, vessel_enabled=False, rigid_bodies_enabled=mounted)
        cfg = configure(spec)
        sim = load_case(
            dict(sim=dict(device="cpu", envs=2, **cfg.sim), assets=cfg.assets, systems=cfg.systems),
            loaders=cfg.loaders,
            systems=cfg.systems_registry,
        )
        control = next(item for item in sim.systems if item.id == "catheter_control")
        assert control.root_mode == ("track" if mounted else "external_pose")
        assert control.target_mode == "absolute"
        assert np.allclose(control.rod.rest_len.numpy(), spec.segment_length_m)
        for _ in range(3):
            sim.step()
        assert np.isfinite(sim.scene.q.numpy()).all()


def test_native_patient_case_preserves_transform_radii_and_pins(write_patient_bundle):
    import numpy as np

    configure = native_configuration()
    from i4h_host.composition import load_case

    from i4h_arena.medical.newton_catheter_physics import CatheterRodSpec

    points = np.float32([[0, 0, 0], [20, 0, 0], [40, 10, 0], [60, 10, 0]])
    edges = np.int32([[0, 1], [1, 2], [2, 3]])
    affine = np.array([[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]])
    affine[:3, 3] = [0.1, 0.2, 0.3]
    manifest = write_patient_bundle(
        np.zeros((3, 3, 3)), np.eye(4), centerline_scan=(points, edges, np.full(4, 5.0)), world_from_scan_m=affine
    )
    spec = CatheterRodSpec(device="cpu", num_envs=2, patient_twin_manifest=str(manifest))
    cfg = configure(spec)
    sim = load_case(
        dict(sim=dict(device="cpu", envs=2, **cfg.sim), assets=cfg.assets, systems=cfg.systems),
        loaders=cfg.loaders,
        systems=cfg.systems_registry,
    )
    wall = sim.scene.body("lumen")
    np.testing.assert_allclose(wall.tree.positions[0], points @ affine[:3, :3].T * 0.001 + affine[:3, 3], atol=1e-7)
    np.testing.assert_allclose(wall.tree.radii, 0.005)
    np.testing.assert_array_equal(sim.scene.field("inv_mass", wall).numpy()[:, [0, -1]], 0)
    before = sim.scene.field("q", wall).numpy().copy()
    for _ in range(3):
        sim.step()
    np.testing.assert_array_equal(sim.scene.field("q", wall).numpy()[:, [0, -1]], before[:, [0, -1]])
    assert np.isfinite(sim.scene.q.numpy()).all()
