# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for placing the catheter rod under Isaac Lab's Newton manager.

The ordering assertions are the valuable ones. Particles added after the model
is finalized are invisible to it, and a rod registered before its particles
exist has nothing to drive, so this checks that registration happens on
MODEL_INIT and in the right order against stubs rather than a live stack.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.medical.newton_catheter_physics import GRAVITY_WORLD_Z_UP, CatheterRodHandle, CatheterRodSpec


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
def test_segment_length_divides_the_catheter():
    spec = CatheterRodSpec(length_m=0.4, num_segments=40)

    assert spec.num_points == 41
    assert spec.segment_length_m == pytest.approx(0.01)


def test_track_direction_is_normalized():
    spec = CatheterRodSpec(track_direction_world=(0.0, 3.0, 4.0))

    assert np.linalg.norm(spec.track_direction_world) == pytest.approx(1.0)
    assert spec.track_direction_world[1] == pytest.approx(0.6)


def test_gravity_defaults_to_the_z_up_world():
    """Isaac is Z-up; the rod config's own default points along -Y."""
    assert CatheterRodSpec().gravity_world == GRAVITY_WORLD_Z_UP
    assert GRAVITY_WORLD_Z_UP[2] < 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_envs": 0}, "num_envs"),
        ({"num_segments": 0}, "num_segments"),
        ({"length_m": 0.0}, "length_m"),
        ({"radius_m": -1.0}, "radius_m"),
        ({"track_direction_world": (0.0, 0.0, 0.0)}, "non-zero"),
    ],
)
def test_invalid_specs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CatheterRodSpec(**kwargs)


def test_a_vessel_needs_a_twin():
    assert CatheterRodSpec(patient_twin_manifest=None).wants_vessel is False
    assert CatheterRodSpec(patient_twin_manifest="twin.yaml").wants_vessel is True
    assert CatheterRodSpec(patient_twin_manifest="twin.yaml", vessel_enabled=False).wants_vessel is False


# --------------------------------------------------------------------------- #
# Handle lifecycle
# --------------------------------------------------------------------------- #
def test_reading_the_particle_range_too_early_explains_why():
    handle = CatheterRodHandle(CatheterRodSpec())

    with pytest.raises(RuntimeError, match="MODEL_INIT"):
        _ = handle.particle_range


def test_no_vessel_when_no_twin_is_configured():
    handle = CatheterRodHandle(CatheterRodSpec(patient_twin_manifest=None))

    assert handle.vessel is None


# --------------------------------------------------------------------------- #
# Newton config
# --------------------------------------------------------------------------- #
class FakeNewtonCfg:
    def __init__(self, solver_cfg=None, use_cuda_graph=True):
        self.solver_cfg = solver_cfg
        self.use_cuda_graph = use_cuda_graph


class FakeSolverCfg:
    def __init__(self, **fields):
        self.__dict__.update(fields)
        self.tip_num_edges = fields.get("tip_num_edges", 10)


@pytest.fixture
def stub_isaac(monkeypatch):
    """Stub the Isaac Lab and solver-config modules the wiring imports."""
    newton_physics = types.ModuleType("isaaclab_newton.physics")
    newton_physics.NewtonCfg = FakeNewtonCfg
    newton_physics.NewtonManager = SimpleNamespace(_builder=None, register_callback=None)
    newton_pkg = types.ModuleType("isaaclab_newton")

    integration = types.ModuleType("catheter_vasculature_solver.isaaclab_integration")
    integration.XPBDRodSolverCfg = FakeSolverCfg

    for name, module in (
        ("isaaclab_newton", newton_pkg),
        ("isaaclab_newton.physics", newton_physics),
        ("catheter_vasculature_solver.isaaclab_integration", integration),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(newton_physics=newton_physics, integration=integration)


def test_solver_cfg_carries_the_scene_geometry(stub_isaac):
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec(length_m=0.4, num_segments=40, radius_m=0.001))

    assert cfg.num_segments == 40
    assert cfg.segment_length == pytest.approx(0.01)
    assert cfg.radius == pytest.approx(0.001)


def test_static_collision_and_track_stay_off(stub_isaac):
    """The deformable centerline supplies containment; two walls would fight."""
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec())

    assert cfg.collision_enabled is False
    assert cfg.track_enabled is False


def test_state_sync_stays_on_so_the_rod_starts_in_the_patient(stub_isaac):
    """It is the only route the centerline has into the solver.

    The rod builds itself as a straight rod along +X and cannot be constructed
    from a polyline, so the seeded Newton buffer reaching it on the first step
    is what puts the catheter in the vessel rather than out in the room.
    """
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec())

    assert getattr(cfg, "sync_from_state", True) is True


def test_solver_overrides_win(stub_isaac):
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec(solver_overrides={"bend_stiffness": 0.5}))

    assert cfg.bend_stiffness == pytest.approx(0.5)


def test_cuda_graph_is_disabled_when_a_vessel_is_present(stub_isaac):
    """Vessel containment resizes contact scratch, which a captured graph cannot express."""
    from i4h_arena.medical.newton_catheter_physics import newton_physics_cfg

    with_vessel = newton_physics_cfg(CatheterRodSpec(patient_twin_manifest="twin.yaml"))
    without_vessel = newton_physics_cfg(CatheterRodSpec(patient_twin_manifest=None))

    assert with_vessel.use_cuda_graph is False
    assert without_vessel.use_cuda_graph is True


def test_physics_cfg_does_not_set_class_type(stub_isaac):
    """NewtonCfg derives class_type from solver_cfg and rejects a manual value."""
    from i4h_arena.medical.newton_catheter_physics import newton_physics_cfg

    cfg = newton_physics_cfg(CatheterRodSpec())

    assert not hasattr(cfg, "class_type") or cfg.class_type is None
    assert cfg.solver_cfg is not None


# --------------------------------------------------------------------------- #
# MODEL_INIT ordering
# --------------------------------------------------------------------------- #
class FakeBuilder:
    def __init__(self):
        self.particle_count = 0


@pytest.fixture
def stub_model_init(monkeypatch, stub_isaac):
    """Record the order of builder population and rod registration."""
    calls: list[str] = []
    recorded: dict = {}
    builder = FakeBuilder()
    stub_isaac.newton_physics.NewtonManager._builder = builder

    def add_catheter_rod_to_builder(passed_builder, config, *, positions, start, direction, num_envs):
        assert passed_builder is builder
        calls.append("add_particles")
        recorded["positions"] = positions
        return SimpleNamespace(offset=0, count=(config.num_points) * num_envs, num_envs=num_envs)

    def rod_config_from_solver_cfg(solver_cfg, *, device):
        calls.append(f"rod_config:{device}")
        return SimpleNamespace(num_points=solver_cfg.num_segments + 1, device=device)

    registered: dict = {}

    class FakeRodManager:
        @staticmethod
        def register_rod(particle_range, *, rod=None):
            calls.append("register_rod")
            registered["particle_range"] = particle_range
            registered["rod"] = rod

    stub_isaac.integration.add_catheter_rod_to_builder = add_catheter_rod_to_builder
    stub_isaac.integration.rod_config_from_solver_cfg = rod_config_from_solver_cfg
    stub_isaac.integration.NewtonXPBDRodManager = FakeRodManager

    solver_module = types.ModuleType("catheter_vasculature_solver")

    def CathRodSolver(config, **kwargs):  # noqa: N802 - mirrors the real class name
        calls.append("build_rod")
        return SimpleNamespace(config=config, kwargs=kwargs)

    solver_module.CathRodSolver = CathRodSolver
    monkeypatch.setitem(sys.modules, "catheter_vasculature_solver", solver_module)
    return SimpleNamespace(calls=calls, registered=registered, builder=builder, recorded=recorded)


def test_particles_are_added_before_the_rod_is_registered(stub_model_init):
    """Registering a rod whose particles do not exist yet leaves it driving nothing."""
    handle = CatheterRodHandle(CatheterRodSpec(num_envs=2, num_segments=8))

    handle._on_model_init()

    calls = stub_model_init.calls
    assert calls.index("add_particles") < calls.index("register_rod")
    assert calls.index("build_rod") < calls.index("register_rod")


def test_registration_passes_the_particle_range_and_the_rod(stub_model_init):
    handle = CatheterRodHandle(CatheterRodSpec(num_envs=3, num_segments=8))

    handle._on_model_init()

    registered = stub_model_init.registered
    assert registered["particle_range"].num_envs == 3
    assert registered["particle_range"].count == 9 * 3
    assert registered["rod"] is handle.rod


def test_the_rod_config_is_built_on_the_requested_device(stub_model_init):
    handle = CatheterRodHandle(CatheterRodSpec(device="cuda:1"))

    handle._on_model_init()

    assert "rod_config:cuda:1" in stub_model_init.calls


def test_a_straight_rod_gets_no_explicit_positions(stub_model_init):
    handle = CatheterRodHandle(CatheterRodSpec(initial_path_world_m=None))

    handle._on_model_init()

    assert stub_model_init.recorded["positions"] is None


def test_the_vessel_path_seeds_the_rod_shape(stub_model_init):
    """The centerline sets the starting shape once, instead of being replayed
    into the solver every step, which would overwrite the physics result."""
    path = tuple((float(index) * 0.05, 0.0, 0.0) for index in range(8))
    spec = CatheterRodSpec(num_segments=8, length_m=0.2, initial_path_world_m=path)
    handle = CatheterRodHandle(spec)

    handle._on_model_init()

    positions = stub_model_init.recorded["positions"]
    assert positions is not None
    assert positions.shape == (spec.num_points, 3)
    # Sampled along the path, so the seeded rod spans the requested length.
    span = float(np.linalg.norm(positions[-1] - positions[0]))
    assert span == pytest.approx(spec.length_m, rel=1e-3)


def test_a_missing_builder_is_reported_not_silently_skipped(stub_model_init, stub_isaac):
    stub_isaac.newton_physics.NewtonManager._builder = None
    handle = CatheterRodHandle(CatheterRodSpec())

    with pytest.raises(RuntimeError, match="ModelBuilder"):
        handle._on_model_init()


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
class _RecordingRod:
    def __init__(self) -> None:
        self.reset_with: list[object] = []

    def reset(self, env_ids=None) -> None:
        self.reset_with.append(env_ids)


def _handle_with_rod() -> tuple[CatheterRodHandle, _RecordingRod]:
    handle = CatheterRodHandle(CatheterRodSpec())
    rod = _RecordingRod()
    handle._rod = rod
    return handle, rod


def test_reset_before_the_rod_exists_is_a_no_op():
    handle = CatheterRodHandle(CatheterRodSpec())

    handle.reset(None)  # must not raise; MODEL_INIT has not fired yet


def test_reset_forwards_every_environment_as_none():
    handle, rod = _handle_with_rod()

    handle.reset(None)

    assert rod.reset_with == [None]


def test_reset_forwards_the_index_tensor_untouched():
    """IsaacLab hands reset the device tensor it builds, and the solver takes it.

    Converting here instead would put a copy at each caller of a solver that
    already accepts device buffers at every entry point.
    """
    torch = pytest.importorskip("torch")
    handle, rod = _handle_with_rod()
    env_ids = torch.tensor([1, 0], dtype=torch.int32)

    handle.reset(env_ids)

    assert rod.reset_with[0] is env_ids
