# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit navigation case construction for the numerical_sim migration.

This module does not activate a replacement manager. A host must configure and
construct I4hComponent after qualification; no legacy solver import is needed.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .catheter_initialization import rod_frames_along_polyline


# These experiment switches had silently ignored malformed values and selected
# scheduling/backends we do not carry into the staged runtime.
_OBSOLETE = (
    'I4H_CATHETER_CONTAINMENT', 'I4H_CATHETER_CLEANUP', 'I4H_CATHETER_ROUNDS',
    'I4H_CATHETER_DAMPING', 'I4H_CATHETER_INTERIOR', 'I4H_CATHETER_SEGMENTS',
    'I4H_CATHETER_REST_CURVATURE', 'I4H_CATHETER_TIP_EDGES', 'I4H_CATHETER_TIP_LENGTH_MM',
    'I4H_CATHETER_CONTACT_ITERATIONS', 'I4H_CATHETER_VESSEL', 'I4H_CATHETER_BEND',
)


def navigation_case(spec, *, environ=None):
    """Translate the existing scene spec into explicit SI numerical_sim settings.

    Returned assets and systems are suitable for I4hComponentCfg. A patient
    vessel must be added separately from its complete graph; a route is not a
    substitute for a branching vessel. Material parameters are deliberately
    converted rather than copying dimensionless legacy bend/twist multipliers.
    The case control consumes normalized rates. The future workflow action adapter
    must divide its SI rate columns by the authored control rates (0.060, 1.5, 1.5)
    before writing actions; external root poses instead come from the route drive.
    """
    env = os.environ if environ is None else environ
    obsolete = sorted(key for key in _OBSOLETE if key in env)
    if obsolete:
        raise ValueError(f"numerical_sim migration: replace obsolete overrides {obsolete} with authored case settings")
    retired = {
        'containment_stage': 'post', 'track_stage': 'pre',
        'containment_cleanup_iterations': 128, 'containment_cleanup_rounds': 8,
        'containment_cleanup_relaxation': 1.0,
    }
    changed = [name for name, default in retired.items() if getattr(spec, name) != default]
    if changed:
        raise ValueError(f"numerical_sim migration: obsolete scheduling/cleanup settings {changed}")
    material = dict(spec.solver_overrides)
    allowed = {'density', 'young_modulus', 'poisson_ratio', 'bend_stiffness', 'twist_stiffness',
               'linear_damping', 'angular_damping'}
    unknown = sorted(set(material) - allowed)
    if unknown or spec.soft_contact_overrides:
        raise ValueError(f"numerical_sim migration: unsupported legacy solver/contact overrides: {unknown or spec.soft_contact_overrides}")
    if not spec.physical_rotational_inertia:
        raise ValueError('numerical_sim migration: navigation requires physical rotational inertia')
    segments = int(spec.num_segments)
    length, radius = float(spec.length_m), float(spec.radius_m)
    density = float(material.get('density', 7800.0))
    young = float(material.get('young_modulus', 1.0e9))
    poisson = float(material.get('poisson_ratio', 0.3))
    bend = float(material.get('bend_stiffness', 0.1))
    twist = float(material.get('twist_stiffness', 0.4))
    damping = float(material.get('linear_damping', 0.01))
    angular_damping = float(material.get('angular_damping', 0.01))
    values = [length, radius, density, young, poisson, bend, twist, damping, angular_damping]
    if not np.isfinite(values).all() or min(length, radius, density, young, bend, twist) <= 0:
        raise ValueError('navigation material and geometry must be finite and positive')
    if segments < 2 or not -1 < poisson < 0.5 or not 0 <= damping <= 1 or not 0 <= angular_damping <= 1:
        raise ValueError('invalid navigation segment count, Poisson ratio or damping')
    segment = length / segments
    tip_edges = max(1, math.ceil(float(spec.tip_length_m) / segment))
    if tip_edges >= segments:
        raise ValueError("numerical_sim migration: tip must leave at least one shaft edge")
    if not 0 < float(spec.tip_length_m) < length:
        raise ValueError('numerical_sim migration: tip length must be positive and shorter than the rod')
    startup = np.asarray(spec.initial_positions(), dtype=np.float32)
    frames = rod_frames_along_polyline(startup)
    curvature = np.zeros((segments, 3), dtype=np.float32)
    if spec.rest_curvature_from_path:
        a, b = frames[:-1], frames[1:]
        relative = a[:, 3:4] * b[:, :3] - b[:, 3:4] * a[:, :3] - np.cross(a[:, :3], b[:, :3])
        curvature[:-tip_edges] = relative[:-tip_edges] * float(spec.rest_curvature_scale)
    free = tip_edges if spec.track_free_distal_length_m is None else min(
        segments, int(float(spec.track_free_distal_length_m) / segment))
    route = spec.initial_path_world_m
    # Rigid mounts have exclusive ownership. A route may still guide the shaft.
    root_mode = 'external_pose' if route is not None and spec.drive_body_name is None else 'track'
    if not 0 < spec.tip_bend_fraction <= 1 or not 0 <= spec.rest_curvature_scale <= 1:
        raise ValueError('invalid tip stiffness fraction or material curvature scale')
    if spec.contact_coupling_iterations < 1:
        raise ValueError('numerical_sim migration: staged contact requires at least one iteration')
    regions = {'schema_version': 1, 'default': 'shaft', 'regions': [
        {'name': f'tip_{i}', 'interval': [(segments-tip_edges+i)/segments, (segments-tip_edges+i+1)/segments]}
        for i in range(tip_edges)]}
    moment = math.pi * radius**4 / 4.0
    return {
        'sim': {'device': spec.device, 'envs': spec.num_envs, 'dt': 1.0 / 120.0, 'substeps': 1,
                'iterations': int(spec.contact_coupling_iterations), 'gravity': list(spec.gravity_world)},
        'assets': {'catheter': {'type': 'rod', 'start': startup[0].tolist(),
                               'direction': list(spec.track_direction_world), 'count': segments + 1,
                               'segment_length': segment, 'radius': radius,
                               'mass': density * math.pi * radius**2 * segment,
                               'startup_positions': startup.tolist(), 'startup_frames': frames.tolist(), 'regions': regions,
                               'rest_lengths': [segment] * segments, 'rest_curvature': curvature.tolist()}},
        'systems': [
            {'type': 'particles', 'damping': 1.0 - damping},
            {'type': 'rod_direct_material', 'id': 'catheter_material', 'body': 'catheter'},
            {'type': 'rod_direct', 'id': 'rod', 'body': 'catheter', 'young': young,
             'torsion': young / (2.0 * (1.0 + poisson)), 'bend': 4.0 * bend * moment / segment**2,
             'twist': 4.0 * twist * 2.0 * moment / segment**2, 'angular_damping': angular_damping,
             'inertia': 'physical'},
            {'type': 'catheter_control', 'id': 'catheter_control', 'rod': 'rod', 'root_mode': root_mode,
             'track_length': length if spec.track_guidance else 0.0,
             'track_path': route if spec.track_guidance else None, 'track_stiffness': float(spec.track_stiffness),
             'distal_free_nodes': free, 'tip_edges': tip_edges, 'tip_mapping': 'polyline',
             'rates': [0.060, 1.5, 1.5], 'max_delta': 0.005},
        ],
    }


def configure_material(component, spec):
    """Apply the authored cosine taper through public regional tables after each build.

    Register on I4hComponent.on_rebuild. The tables persist through selective reset
    and are consumed by the existing rod_direct_material system inside capture.
    """
    body = component.scene.body('catheter')
    properties = body.region_properties
    regions = [name for name in properties.membership.names if name.startswith('tip_')]
    for i, name in enumerate(regions):
        u = i / max(len(regions)-1, 1)
        scale = float(spec.tip_bend_fraction) + (1.0-float(spec.tip_bend_fraction)) * 0.5 * (1.0+math.cos(math.pi*u))
        properties.edit({'bend_multiplier': scale, 'twist_multiplier': scale}, name)
