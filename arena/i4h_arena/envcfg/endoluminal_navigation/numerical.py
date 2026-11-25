# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Numerical navigation composed into Arena using its custom environment factory."""

from __future__ import annotations

import copy
import math

import numpy as np
from i4h_isaaclab.component import I4hComponent, I4hComponentCfg
from i4h_isaaclab.manager import I4hHostMixin
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

from i4h_arena.embodiments.catheter import CatheterEmbodiment
from i4h_arena.embodiments.franka_catheter import FrankaCatheterEmbodiment
from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg
from i4h_arena.medical.numerical_catheter import NumericalCatheterAssetCfg


def add_lumen(scene, name, points, edges, radii, endpoints_locked=True):
    from i4h_endoluminal.loaders.centreline import add_centreline_tree

    points, edges = np.float32(points), np.int32(edges)
    pinned = np.flatnonzero(np.bincount(edges.ravel(), minlength=len(points)) == 1) if endpoints_locked else [0]
    return add_centreline_tree(
        scene,
        name,
        points,
        edges,
        np.float32(radii),
        0,
        pinned_nodes=pinned,
        pinned_segments=(0,),
        mass=1.0,
        inertia=1.0,
        open_root=True,
        precise_rest_frames=True,
    )


def component_configuration(spec):
    """Map the existing patient/rod description to accepted native mechanics."""
    from i4h_endoluminal.loaders import LOADERS
    from i4h_endoluminal.systems import SYSTEMS

    from i4h_arena.medical.catheter_initialization import rod_frames_along_polyline
    from i4h_arena.medical.newton_catheter_physics import rest_curvature_override, seeded_rest_curvature_scale
    from i4h_common.centerline import sample_polyline

    settings = rod_solver_cfg(spec)
    length = spec.segment_length_m
    area = math.pi * spec.radius_m**2
    inertia = math.pi * spec.radius_m**4 / 4
    start = np.asarray(spec.origin_world_m)
    points = start + np.linspace(0, spec.length_m, spec.num_segments + 1)[:, None] * spec.track_direction_world
    if spec.initial_path_world_m is not None:
        points = sample_polyline(
            np.asarray(spec.initial_path_world_m), np.linspace(0, spec.length_m, spec.num_segments + 1)
        )
    frames = rod_frames_along_polyline(np.float32(points))
    curvature = np.zeros((spec.num_segments, 3), np.float32)
    scale = seeded_rest_curvature_scale(
        from_path=spec.rest_curvature_from_path,
        spec_scale=spec.rest_curvature_scale,
        override=rest_curvature_override(),
    )
    if scale is not None and spec.initial_path_world_m is not None:
        # Match the legacy dimensionless Im(conj(q0)*q1), retaining an unseeded tip.
        import torch

        from i4h_arena.medical.newton_catheter_physics import rest_darboux_along_polyline

        shaft = spec.num_segments - settings.tip_num_edges
        curvature[:shaft] = (
            rest_darboux_along_polyline(torch.as_tensor(np.float32(points)), length).numpy()[:shaft] * scale
        )
    assets = {
        "catheter": dict(
            type="rod",
            start=list(spec.origin_world_m),
            direction=list(spec.track_direction_world),
            startup_positions=np.asarray(points).tolist(),
            startup_frames=frames.tolist(),
            rest_lengths=[length] * spec.num_segments,
            rest_curvature=curvature.tolist(),
            count=spec.num_segments + 1,
            segment_length=length,
            radius=spec.radius_m,
            mass=settings.density * area * length,
            regions={
                "schema_version": 1,
                "default": "shaft",
                "regions": [
                    {"name": f"tip_{edge}", "interval": [edge / spec.num_segments, (edge + 1) / spec.num_segments]}
                    for edge in range(spec.num_segments - settings.tip_num_edges, spec.num_segments)
                ],
            },
        )
    }
    particles = dict(type="particles", id="catheter_particles", body="catheter", damping=1 - settings.linear_damping)
    rod = dict(
        type="rod_direct",
        id="rod",
        body="catheter",
        young=settings.young_modulus,
        torsion=settings.torsion_modulus or settings.young_modulus / 2.6,
        bend=4 * settings.bend_stiffness * inertia / length**2,
        twist=8 * settings.twist_stiffness * inertia / length**2,
        angular_damping=settings.angular_damping,
        inertia="physical" if spec.physical_rotational_inertia else "unit",
    )
    control = dict(
        type="catheter_control",
        id="catheter_control",
        rod="rod",
        track_length=0.0,
        tip_edges=settings.tip_num_edges,
        tip_mapping="polyline",
        root_mode="track" if spec.rigid_bodies_enabled else "external_pose",
        target_mode="absolute",
        max_delta=1.0,
        lower=[-1e9, -1e9, -1.5],
        upper=[1e9, 1e9, 1.5],
    )
    if spec.track_guidance and spec.initial_path_world_m is not None:
        free = (
            min(spec.num_segments, int(spec.track_free_distal_length_m / length))
            if spec.track_free_distal_length_m is not None
            else settings.tip_num_edges
        )
        control.update(
            track_path=spec.initial_path_world_m,
            distal_free_nodes=free,
            track_stiffness=spec.track_stiffness,
        )
    material = dict(type="rod_direct_material", id="catheter_material", body="catheter")
    systems = [particles, material, rod, control]
    if spec.wants_vessel:
        from i4h_arena.medical.patient_twin import PatientTwin
        from i4h_arena.medical.vessel_deformation import length_scale_from_affine

        twin = PatientTwin.load(spec.patient_twin_manifest)
        line = twin.centerline()
        if line is None or line.radii_mm is None:
            raise ValueError("numerical deformable vessel requires the patient centerline and radii")
        radii = np.asarray(line.radii_mm) * 0.001 * length_scale_from_affine(twin.world_from_patient_m)
        assets["lumen"] = dict(
            type="navigation_lumen",
            points=twin.patient_mm_to_world(line.points_mm).tolist(),
            edges=np.asarray(line.edges).tolist(),
            radii=radii.tolist(),
            endpoints_locked=spec.vessel_endpoints_locked,
        )
        systems = [
            particles,
            dict(
                type="particles",
                id="wall_particles",
                body="lumen",
                gravity_scale=0.0,
                damping_at_finalize=True,
                damping=1 - spec.vessel_linear_damping,
            ),
            dict(
                type="rod_pbd",
                body="lumen",
                scheme="colored",
                pre_sweeps=4,
                sweeps=4,
                stretch=1.0,
                bend=1.0,
                twist=1.0,
                angular_damping=spec.vessel_angular_damping,
                max_delta=0.005,
                limit_motion=True,
            ),
            dict(
                type="graph_containment",
                body="catheter",
                centreline="lumen",
                two_way=True,
                target_response=spec.vessel_response,
                spline=False,
                candidates=False,
                open_root=True,
                max_distance=0.05,
                iterations=2,
                source_max_delta=0.005,
                target_max_delta=0.005,
                interior_deadband=spec.containment_interior_deadband,
                interior_stiffness=spec.containment_interior_stiffness,
            ),
            material,
            rod,
            control,
        ]
    return I4hComponentCfg(
        assets=assets,
        systems=systems,
        sim=dict(
            dt=1 / 120,
            substeps=1,
            iterations=spec.contact_coupling_iterations if spec.wants_vessel else 1,
            gravity=list(spec.gravity_world),
        ),
        loaders={**LOADERS, "navigation_lumen": add_lumen},
        systems_registry=SYSTEMS,
    )


class NumericalNavigationEnv(I4hHostMixin, IsaacLabArenaManagerBasedRLEnv):
    def i4h_save_terminal(self, env_ids):
        if hasattr(self, "reward_buf"):
            env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
            if not len(env_ids):
                return
            self.extras["navigation_terminal"] = {
                "env_ids": env_ids.clone(),
                "reward": self.reward_buf[env_ids].clone(),
                "episode_length": self.episode_length_buf[env_ids].clone(),
                "points": self.scene["catheter"].data.positions_world_m[env_ids].clone(),
            }

    def step(self, action):
        self.extras.pop("navigation_terminal", None)
        return super().step(action)

    def capture_complete_state(self, env_ids=None):
        from i4h_arena.medical.navigation_continuation import capture

        return capture(self, env_ids)

    def restore_complete_state(self, bundle, env_ids=None):
        from i4h_arena.medical.navigation_continuation import restore

        restore(self, bundle, env_ids)

    def reset_to(self, state, env_ids=slice(None), seed=None, is_relative=False):
        from i4h_arena.medical.navigation_continuation import numpy_bundle, validate

        if "navigation" not in state:
            raise ValueError("numerical navigation replay requires initial_state/navigation")
        bundle = numpy_bundle(state["navigation"])
        validate(self, bundle, env_ids)
        host = {key: value for key, value in state.items() if key != "navigation"}
        super().reset_to(host, env_ids, seed, is_relative)
        self.restore_complete_state(bundle, env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=False)
        return self.obs_buf, self.extras

    def _bind_i4h(self, component):
        from i4h_arena.medical.navigation_goal import invalidate_arrival_readout

        invalidate_arrival_readout(self)
        super()._bind_i4h(component)
        if getattr(self, "scene", None) is not None and "catheter" in self.scene.sensors:
            self.scene["catheter"].bind_component(component)
            from i4h_arena.medical.newton_catheter_physics import tip_bend_stiffness_profile

            spec = self.cfg.navigation_rod_spec
            tip_edges = rod_solver_cfg(spec).tip_num_edges
            profile = tip_bend_stiffness_profile(1.0, spec.num_segments, tip_edges, spec.tip_bend_fraction)
            props = component.scene.body("catheter").region_properties
            for edge in range(spec.num_segments - tip_edges, spec.num_segments):
                props.edit({"bend_multiplier": float(profile[edge])}, f"tip_{edge}")
            sensor = self.scene["fluoroscopy"]
            sensor._catheter_provider = None
            sensor._carm_provider = None


class NumericalCatheterEmbodiment(CatheterEmbodiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene_config.catheter = NumericalCatheterAssetCfg(
            prim_path="{ENV_REGEX_NS}/Catheter",
            update_period=0.0,
            debug_vis=True,
            rod_spec=copy.deepcopy(self.rod_spec),
        )

    def get_recorder_term_cfg(self, record_trajectories=False):
        from isaaclab.managers import RecorderTermCfg

        from i4h_arena.medical.navigation_continuation import NavigationInitialStateRecorder

        cfg = super().get_recorder_term_cfg(record_trajectories)
        from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
        from isaaclab.utils import configclass
        from isaaclab_arena.utils.configclass import combine_configclass_instances

        @configclass
        class InitialTerms:
            record_navigation_initial_state: RecorderTermCfg = RecorderTermCfg(
                class_type=NavigationInitialStateRecorder
            )

        return combine_configclass_instances(
            "NumericalNavigationRecorders", cfg, InitialTerms(), bases=(RecorderManagerBaseCfg,)
        )

    def tune_env_cfg(self, cfg):
        cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 1 / 120, 4, 4
        cfg.scene.replicate_physics, cfg.scene.env_spacing = True, 0.0
        component = I4hComponent(component_configuration(self.rod_spec))
        if self.rod_spec.rigid_bodies_enabled:
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

            # The legacy coupled manager constructs SolverMuJoCo with its
            # implicitfast default. Lab's cfg defaults to Euler, which makes
            # this arm's stiff position/finger drives unstable at 120 Hz.
            cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator="implicitfast"))
            component.cfg.mounts = {
                "catheter": {
                    "body": self.rod_spec.drive_body_name,
                    "offset": (self.rod_spec.drive_mount_local, (0.0, 0.0, 0.0, 1.0)),
                }
            }
            component.cfg.coupling_mode = "two_way"
            self._seed_the_arm_pose_on_reset(cfg)
        component.configure(cfg.sim, rigid=self.rod_spec.rigid_bodies_enabled)
        cfg.navigation_rod_spec = copy.deepcopy(self.rod_spec)
        cfg.compute_final_obs = True
        return cfg


class NumericalFrankaCatheterEmbodiment(NumericalCatheterEmbodiment, FrankaCatheterEmbodiment):
    """The existing arm controller over Numerical Sim's shared-model mount coupling."""
