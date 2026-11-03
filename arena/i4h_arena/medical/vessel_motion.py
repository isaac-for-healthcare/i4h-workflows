# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Carry the deformation solver's wall motion across to the renderer's centerline.

The two sides hold the same vessel in different forms. The solver is handed the centerline as a
list of edges and welds their endpoints into its own node list, stitches disconnected pieces and
re-roots the result, so rod node ``k`` is not twin node ``k`` and there may not even be the same
number of them. The renderer indexes everything -- radii, contrast, the carve's measured
attenuation -- by the twin's own node order.

The correspondence is therefore established once, at rest, by nearest neighbour, and what
crosses afterwards is *displacement* rather than position. That matters for more than
robustness: the renderer smooths the centerline before building any geometry on it and the
solver does not, so transferring absolute positions would snap the rendered vessel back onto the
voxel staircase the smoothing exists to remove.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .patient_twin import PatientTwin

if TYPE_CHECKING:
    from .patient_volume import PatientVolume

logger = logging.getLogger("i4h_arena.medical")

#: A rod node further than this from every twin node means the two are not describing the same
#: vessel -- a different centerline artifact, or a transform applied on one side only.
_MATCH_TOLERANCE_M = 0.005


@dataclass(frozen=True, slots=True)
class VesselMotion:
    """One frame of wall motion and contrast, indexed by the twin's centerline nodes.

    Args:
        displacement_world_m: Movement from rest, ``(envs, nodes, 3)``, in world metres. Rest is
            whatever the renderer already holds, so an all-zero displacement renders the
            undeformed vessel.
        concentration_mg_per_ml: Iodine at each node, ``(envs, nodes)`` or ``(nodes,)``. ``None``
            leaves the vessel at the scan's own contrast.
    """

    displacement_world_m: np.ndarray
    concentration_mg_per_ml: np.ndarray | None = None


@runtime_checkable
class VesselMotionProvider(Protocol):
    """Physics-independent source of the latest wall motion, mirroring the catheter's."""

    def snapshot(self, num_envs: int) -> VesselMotion:
        """Return this frame's displacement for every environment."""


class ContrastRun:
    """Keep contrast running down the tree from the catheter tip.

    The injection runs continuously by default, which is not what an interventionalist does --
    they inject in short runs and watch each wash through -- but a cycling bolus leaves the
    vessel invisible for part of every cycle, and someone watching the scene to see whether the
    wall moves should not have to wait for the contrast to come back. Set ``cycle_s`` to inject
    in runs instead.

    One transport is shared across environments. The bolus therefore follows the first
    environment's catheter, which is right for watching a single scene and wrong for a batched
    rollout; splitting it per environment is a matter of holding a list here, deferred until
    something needs it.
    """

    def __init__(
        self,
        patient: PatientVolume,
        *,
        dt_s: float,
        duration_s: float = 1.5,
        cycle_s: float | None = None,
        flow_ml_per_s: float = 20.0,
    ) -> None:
        """Build the transport from the patient's own centerline.

        Args:
            patient: Loaded volume, whose twin supplies the centerline and the carve's fill.
            dt_s: Simulation time between consecutive reads of this run.
            duration_s: How long each injection lasts, when injecting in runs.
            cycle_s: Period between the start of one run and the next. ``None`` injects
                continuously and ignores ``duration_s``.
            flow_ml_per_s: Injector rate, diluted by blood flow at the site.
        """
        from .contrast_bolus import ContrastAgent, ContrastTransport, HuToMu
        from .vessel_carve import carve_lumen, centerline_in_volume_mm

        points_mm, radii_mm = centerline_in_volume_mm(patient)
        edges = np.load(patient.twin.artifacts["centerline_edges"])
        self._patient = patient
        self._transport = ContrastTransport(points_mm, edges, radii_mm)
        self._agent = ContrastAgent()
        self._curve = HuToMu.from_patient(patient)
        self._baseline_mu = carve_lumen(patient).fill_mu_per_mm
        self._dt_s = float(dt_s)
        self._duration_s = float(duration_s)
        self._cycle_s = None if cycle_s is None else float(cycle_s)
        self._flow_ml_per_s = float(flow_ml_per_s)

    @property
    def baseline_mu_per_mm(self) -> float:
        """Attenuation of the material the carve filled the lumen with."""
        return self._baseline_mu

    def advance(self, tip_world_m: np.ndarray | None) -> np.ndarray:
        """Step the bolus one frame and return each node's iodine concentration.

        Args:
            tip_world_m: Catheter tips, ``(envs, 3)`` in world metres, or ``None`` to inject
                where blood enters instead.
        """
        from .contrast_bolus import Injection

        site = self._transport.inflow_node
        if tip_world_m is not None:
            site = int(self._transport.nearest_node(self._patient.world_to_volume_mm(tip_world_m[:1])[0]))
        elapsed = self._transport.time_s
        # One injection covering this step is enough; the window only has to contain ``elapsed``,
        # so a continuous run is the cycling one with the gap removed rather than a second path.
        start_s = 0.0 if self._cycle_s is None else elapsed - (elapsed % self._cycle_s)
        duration_s = elapsed + self._dt_s if self._cycle_s is None else self._duration_s
        injection = Injection(
            node=site,
            start_s=start_s,
            duration_s=duration_s,
            flow_ml_per_s=self._flow_ml_per_s,
        )
        self._transport.step(self._dt_s, injection, agent=self._agent)
        return np.asarray(self._transport.concentration_mg_per_ml, dtype=np.float32)


class CenterlineVesselMotion:
    """Read a :class:`CenterlineVesselRuntime`'s wall and report it against the twin's nodes."""

    def __init__(
        self,
        twin: PatientTwin,
        vessel: Any,
        *,
        contrast: ContrastRun | None = None,
        tip_source: Any = None,
    ) -> None:
        """Match the solver's rod nodes to the twin's, once, while both are at rest.

        Args:
            twin: The patient whose centerline the renderer is indexed by.
            vessel: The solver runtime, read through ``positions_per_env``.
            contrast: Runs a bolus down the tree, so the rendered lumen carries contrast rather
                than whatever the scan happened to be acquired with. ``None`` leaves it at the
                scan's own, which on an unenhanced study is close to invisible.
            tip_source: Catheter asset whose last node is the injection site. Without it the
                bolus enters where blood does, giving a whole-tree aortogram instead of a
                selective run.

        Raises:
            ValueError: If the twin carries no centerline, or if a twin node has no rod node
                within :data:`_MATCH_TOLERANCE_M`, which means the two are not the same vessel.
        """
        from scipy.spatial import cKDTree

        points = twin.artifacts.get("centerline_points")
        if points is None:
            raise ValueError("vessel motion needs the twin's centerline_points artifact")

        self._vessel = vessel
        self._contrast = contrast
        self._tip_source = tip_source
        self._rest_rod_m = np.asarray(vessel.positions_per_env, dtype=np.float64).copy()
        rest_twin_m = np.asarray(twin.patient_mm_to_world(np.load(points)), dtype=np.float64)

        distance, self._match = cKDTree(self._rest_rod_m[0]).query(rest_twin_m, k=1, workers=-1)
        if distance.max() > _MATCH_TOLERANCE_M:
            raise ValueError(
                f"a centerline node is {1000.0 * distance.max():.1f} mm from the nearest node of "
                "the deformable vessel; the solver and the renderer are not holding the same "
                "centerline"
            )
        logger.info(
            "matched %d centerline nodes to %d solver nodes, worst gap %.2f mm",
            len(rest_twin_m),
            self._rest_rod_m.shape[1],
            1000.0 * distance.max(),
        )

    def snapshot(self, num_envs: int) -> VesselMotion:
        """Displacement of every twin node, taken from the rod node matched to it.

        Raises:
            ValueError: If the solver is running a different number of environments.
        """
        live = np.asarray(self._vessel.positions_per_env, dtype=np.float64)
        if len(live) != num_envs:
            raise ValueError(f"the deformable vessel has {len(live)} environments; expected {num_envs}")
        concentration = None
        if self._contrast is not None:
            concentration = self._contrast.advance(self._tip_world_m(num_envs))
        return VesselMotion(
            displacement_world_m=(live - self._rest_rod_m)[:, self._match],
            concentration_mg_per_ml=concentration,
        )

    def _tip_world_m(self, num_envs: int) -> np.ndarray | None:
        """Where the catheter is delivering contrast, or ``None`` when nothing is reading it."""
        positions = getattr(getattr(self._tip_source, "data", None), "positions_world_m", None)
        if positions is None:
            return None
        values = positions.detach().cpu().numpy() if hasattr(positions, "detach") else np.asarray(positions)
        return values.reshape(num_envs, -1, 3)[:, -1, :]


__all__ = ["CenterlineVesselMotion", "ContrastRun", "VesselMotion", "VesselMotionProvider"]
