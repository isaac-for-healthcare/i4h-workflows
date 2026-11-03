# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contrast transport along the centerline, and the attenuation it produces.

Once the lumen is carved out (:mod:`~i4h_arena.medical.vessel_carve`) the vessel carries no
contrast at all, and an injection has to put it back as a quantity that varies along the
centerline and in time. That is what makes an angiogram look like an angiogram: contrast
enters at the catheter tip, fills distally, dilutes, and washes out, and an agent that learns
to inject has to see the difference between a good bolus and a wasted one.

The transport model is one-dimensional advection with dispersion along the vessel tree,

.. math:: \\frac{\\partial c}{\\partial t} + v \\frac{\\partial c}{\\partial s}
          = D \\frac{\\partial^2 c}{\\partial s^2}

solved with first-order upwinding on the centerline nodes, which are already a graph with
radii. Velocity comes from continuity, :math:`v = Q / \\pi r^2`, and at a bifurcation the
flow splits by Murray's law, :math:`Q \\propto r^3` -- the relation that follows from
minimising pumping work plus metabolic cost, and the reason a contrast column slows and
thins as it moves into smaller branches instead of racing through them.

What this is not: it has no pulsatility, no secondary flow, and no reflux past the catheter.
Those matter for bolus *shape* in a real injection, and a CFD model would be the honest way to
get them. The claim here is narrower -- that arrival order, transit time and dilution down a
branching tree are the parts an injection policy can actually act on, and they follow from
continuity and the tree topology rather than from resolving the flow field.

Concentration becomes attenuation through the *scan's own* transfer function, recorded in the
twin's ``volume_metadata`` as ``hu_to_mu``. Using the recorded preset rather than a table of
physical iodine coefficients is what keeps the injected contrast on the same calibration as
the anatomy behind it: the renderer's grey levels were fixed by that curve, so contrast mapped
any other way would be internally inconsistent no matter how physically accurate it was. The
scan's own iodine concentration is likewise *fitted* to the attenuation the carve measured,
not assumed, so a brightness matching the original study is the default rather than a tuning
exercise.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .patient_volume import PatientVolume

#: Attenuation change per milligram of iodine per millilitre, in Hounsfield units, at about
#: 120 kVp. Iodine's K-edge sits at 33.2 keV, so this is tube-voltage dependent; 25 HU is the
#: figure for the spectrum interventional work is normally done at.
HU_PER_MG_IODINE_PER_ML = 25.0
#: Iodine concentration of undiluted contrast as supplied, in mg I/mL.
STOCK_CONCENTRATION_MG_PER_ML = 350.0
#: Longitudinal dispersion, mm^2/s. Lumps the velocity-profile spreading that a 1-D model
#: cannot represent (Taylor dispersion) into one coefficient.
DISPERSION_MM2_PER_S = 40.0
#: Mean flow speed at the inlet, mm/s. Inlet flow is derived from this and the inlet's own
#: cross-section rather than fixed in mL/s, because a rate that is right for the aorta is two
#: orders of magnitude wrong for a cerebral branch, and the error shows up as transit times
#: that are visibly absurd rather than subtly off. 150 mm/s is a cycle-averaged large-artery
#: figure; systolic peak is several times higher, which a steady model does not represent.
INLET_VELOCITY_MM_PER_S = 150.0


@dataclass(frozen=True, slots=True)
class HuToMu:
    """The scan's recorded Hounsfield-to-attenuation curve, as piecewise-linear control points.

    Carried explicitly rather than recomputed because it is the calibration the rendered image
    was built on. Both directions are needed: forward to turn an injected concentration into
    attenuation, and inverse to recover the Hounsfield value of the material the carve filled
    with, so the injection is added on top of the right baseline.
    """

    hu: np.ndarray
    mu: np.ndarray

    @classmethod
    def from_metadata(cls, metadata: dict) -> HuToMu:
        """Read ``hu_to_mu.control_points`` out of a twin's volume metadata.

        Raises:
            ValueError: If no control points are recorded, or they are not monotone in
                Hounsfield units, which would make the curve unusable in either direction.
        """
        points = (metadata.get("hu_to_mu") or {}).get("control_points")
        if not points:
            raise ValueError("volume_metadata carries no hu_to_mu.control_points")
        table = np.asarray(points, dtype=np.float64)
        hu, mu = table[:, 0], table[:, 1]
        if np.any(np.diff(hu) <= 0.0) or np.any(np.diff(mu) < 0.0):
            raise ValueError("hu_to_mu control points must increase in HU and not decrease in mu")
        return cls(hu=hu, mu=mu)

    @classmethod
    def from_patient(cls, patient: PatientVolume) -> HuToMu:
        """Read the curve from the twin's declared ``volume_metadata`` artifact.

        Raises:
            ValueError: If the twin declares no ``volume_metadata``.
        """
        path = patient.twin.artifacts.get("volume_metadata")
        if path is None:
            raise ValueError("the contrast model needs the twin's volume_metadata artifact")
        with open(path) as handle:
            return cls.from_metadata(json.load(handle))

    def mu_of(self, hu: np.ndarray | float) -> np.ndarray:
        """Attenuation per millimetre at the given Hounsfield values."""
        return np.interp(np.asarray(hu, dtype=np.float64), self.hu, self.mu)

    def hu_of(self, mu: np.ndarray | float) -> np.ndarray:
        """Hounsfield value at the given attenuation, inverting the curve.

        The curve is only invertible where it strictly increases, and the interventional preset
        is flat at zero from -1000 to -300 HU. Each flat run collapses to its *last* point, so
        zero attenuation inverts to -300 -- the densest material that still renders as air --
        and, more importantly, the segment above it spans the right Hounsfield range. Keeping
        the first point of a run instead stretches that segment back to -1000 and drags every
        soft-tissue inversion hundreds of Hounsfield units negative.
        """
        last_of_run = np.diff(self.mu, append=np.inf) > 0.0
        return np.interp(
            np.asarray(mu, dtype=np.float64),
            self.mu[last_of_run],
            self.hu[last_of_run],
        )


@dataclass(frozen=True, slots=True)
class ContrastAgent:
    """The contrast medium and how it registers on this scan's calibration."""

    #: Iodine concentration of the syringe, mg I/mL.
    stock_mg_per_ml: float = STOCK_CONCENTRATION_MG_PER_ML
    #: Hounsfield rise per mg I/mL, at the tube voltage in use.
    hu_per_mg_per_ml: float = HU_PER_MG_IODINE_PER_ML

    def mu_excess(self, concentration_mg_per_ml: np.ndarray, *, baseline_mu: float, curve: HuToMu) -> np.ndarray:
        """Attenuation the contrast adds over the material the carve filled with.

        The addition happens in Hounsfield units, where iodine's contribution is linear, and
        is only then mapped through the transfer function, which is not. Doing it the other way
        round -- scaling attenuation directly -- would be wrong wherever the preset's slope
        changes, which for the interventional curve is right across the soft-tissue range the
        lumen sits in.
        """
        baseline_hu = float(self.hu_of_baseline(baseline_mu, curve))
        enhanced_hu = baseline_hu + self.hu_per_mg_per_ml * np.asarray(concentration_mg_per_ml, dtype=np.float64)
        return np.maximum(curve.mu_of(enhanced_hu) - baseline_mu, 0.0)

    @staticmethod
    def hu_of_baseline(baseline_mu: float, curve: HuToMu) -> float:
        """Hounsfield value of the carve's fill material."""
        return float(curve.hu_of(baseline_mu))

    def concentration_matching(self, mu_excess: float, *, baseline_mu: float, curve: HuToMu) -> float:
        """Concentration that reproduces a measured attenuation excess, in mg I/mL.

        This is how the scan's own contrast concentration is recovered: hand it what the carve
        measured for ``contrast_mu_per_mm`` and it returns the concentration the study was
        acquired at, by inverting the same two steps :meth:`mu_excess` applies. Replaying that
        concentration therefore reproduces the original study's brightness by construction,
        which is the calibration the whole injection path hangs off.
        """
        baseline_hu = self.hu_of_baseline(baseline_mu, curve)
        enhanced_hu = float(curve.hu_of(baseline_mu + float(mu_excess)))
        return max(enhanced_hu - baseline_hu, 0.0) / self.hu_per_mg_per_ml


@dataclass(frozen=True, slots=True)
class Injection:
    """One contrast injection, as an interventionalist would specify it.

    The site is part of the specification because it is what the catheter controls and what
    distinguishes a diagnostic run from a wasted one. It is deliberately separate from the
    tree's inflow: blood runs from the aortic root downstream whatever the catheter is doing,
    so contrast delivered mid-aorta fills distally and nothing travels back up to the root.
    """

    #: Centerline node the catheter tip is delivering into.
    node: int
    #: Simulation time the injection starts, in seconds.
    start_s: float
    #: How long the plunger is down, in seconds.
    duration_s: float
    #: Injection rate, mL/s.
    flow_ml_per_s: float = 2.0
    #: Dilution of the stock agent, where ``1.0`` is neat contrast.
    dilution: float = 1.0

    def flow_at(self, time_s: float) -> float:
        """Injection rate at a given time, zero outside the injection window."""
        if self.start_s <= time_s < self.start_s + self.duration_s:
            return float(self.flow_ml_per_s)
        return 0.0


class ContrastTransport:
    """Contrast concentration at every centerline node, advanced in time.

    The centerline graph is rooted at the vessel's *inflow* and walked breadth-first, which
    gives every node a single upstream parent and fixes the flow direction along each edge.
    Flow is split at bifurcations by Murray's law and converted to velocity by continuity, so a
    node's transit time follows from the anatomy's own radii rather than a per-case tuning
    parameter.

    Args:
        points_mm: Centerline node positions, in the frame the renderer will sample.
        edges: Undirected node index pairs, shape ``(E, 2)``.
        radii_mm: Lumen radius at each node.
        inflow: Node blood enters the anatomy at, which for an aortic tree is the root. This
            is a property of the vessel, not of the procedure; the catheter's position enters
            through :class:`Injection` instead. Defaults to the widest endpoint, which is a
            heuristic standing in for provenance the twin manifest does not record -- worth
            passing explicitly wherever the answer is known.
        flow_ml_per_s: Blood flow entering there. Defaults to that node's own cross-section
            carrying :data:`INLET_VELOCITY_MM_PER_S`, so transit times come out right for
            whatever caliber of vessel the twin describes.
        dispersion_mm2_per_s: Longitudinal dispersion coefficient.

    Raises:
        ValueError: If the arrays disagree in length, ``inflow`` is out of range, or the graph
            leaves nodes unreachable from it, which means a disconnected centerline rather
            than a modelling choice.
    """

    def __init__(
        self,
        points_mm: np.ndarray,
        edges: np.ndarray,
        radii_mm: np.ndarray,
        *,
        inflow: int | None = None,
        flow_ml_per_s: float | None = None,
        dispersion_mm2_per_s: float = DISPERSION_MM2_PER_S,
    ) -> None:
        self._points = np.asarray(points_mm, dtype=np.float64)
        self._radii = np.asarray(radii_mm, dtype=np.float64).reshape(-1)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        node_count = len(self._points)
        if len(self._radii) != node_count:
            raise ValueError(f"{len(self._radii)} radii for {node_count} centerline nodes")
        if inflow is None:
            inflow = _widest_endpoint(edges, self._radii)
        if not 0 <= int(inflow) < node_count:
            raise ValueError(f"inflow node {inflow} is outside the centerline's {node_count} nodes")

        self._parent, self._order = _root_at(edges, node_count, int(inflow))
        if len(self._order) != node_count:
            raise ValueError(
                f"{node_count - len(self._order)} centerline nodes are unreachable from inflow node "
                f"{inflow}; the centerline graph is disconnected"
            )

        self._length_mm = np.ones(node_count)
        interior = self._order[1:]
        self._length_mm[interior] = np.maximum(
            np.linalg.norm(self._points[interior] - self._points[self._parent[interior]], axis=-1),
            1.0e-6,
        )
        area_mm2 = np.pi * np.square(self._radii)
        if flow_ml_per_s is None:
            # Sized from the trunk's widest caliber rather than the inflow node's own radius.
            # Centerline radii come from the inscribed sphere, which collapses at a segmentation's
            # cut face, so a cropped aorta reports a 4 mm root on a 15 mm vessel -- and sizing
            # flow from that starves the whole tree, giving transit times an order of magnitude
            # long. The widest radius is the robust estimate of the vessel's true caliber, and
            # because transit time is the tree's blood volume over its flow, getting the total
            # right fixes the timing even where individual radii stay noisy.
            inflow_mm3_per_s = float(np.pi * np.square(self._radii.max())) * INLET_VELOCITY_MM_PER_S
        else:
            inflow_mm3_per_s = float(flow_ml_per_s) * 1.0e3
        self._flow_mm3_per_s = _split_flow(self._parent, self._order, self._radii, inflow_mm3_per_s)
        self._velocity_mm_per_s = self._flow_mm3_per_s / np.maximum(area_mm2, 1.0e-9)
        self._dispersion = float(dispersion_mm2_per_s)
        self._inflow = int(inflow)
        self._concentration = np.zeros(node_count)
        self._time_s = 0.0

        # Explicit upwinding is stable only below the Courant limit, and dispersion adds its own
        # diffusive limit. Both are properties of the anatomy, so the safe substep is known here
        # once rather than guessed per call.
        courant = float(np.min(self._length_mm / np.maximum(self._velocity_mm_per_s, 1.0e-9)))
        diffusive = float(np.min(np.square(self._length_mm))) / max(2.0 * self._dispersion, 1.0e-9)
        self._max_substep_s = 0.5 * min(courant, diffusive)

    @property
    def concentration_mg_per_ml(self) -> np.ndarray:
        """Iodine concentration at each centerline node."""
        return self._concentration

    @property
    def velocity_mm_per_s(self) -> np.ndarray:
        """Flow speed at each node, from continuity and the Murray split."""
        return self._velocity_mm_per_s

    @property
    def flow_ml_per_s(self) -> np.ndarray:
        """Blood flow at each node.

        Worth reading before choosing an injection rate: the concentration delivered at the tip
        is the flow-weighted mix of injectate and the blood already passing it, so an injection
        much slower than the local flow is diluted away however long it runs. This is why
        aortography needs tens of millilitres per second where a selective run needs two.
        """
        return self._flow_mm3_per_s * 1.0e-3

    @property
    def inflow_node(self) -> int:
        """Node the tree was rooted at, which fixes every edge's flow direction."""
        return self._inflow

    def nearest_node(self, point_mm: np.ndarray) -> int:
        """Centerline node closest to a point, for turning a catheter tip into an injection site."""
        return int(np.argmin(np.linalg.norm(self._points - np.asarray(point_mm, dtype=np.float64), axis=-1)))

    @property
    def transit_time_s(self) -> np.ndarray:
        """Time for contrast to reach each node from the inflow, along its own path."""
        transit = np.zeros(len(self._points))
        for node in self._order[1:]:
            transit[node] = transit[self._parent[node]] + self._length_mm[node] / max(
                self._velocity_mm_per_s[node], 1.0e-9
            )
        return transit

    @property
    def time_s(self) -> float:
        """Simulation time reached so far."""
        return self._time_s

    def reset(self) -> None:
        """Clear all contrast and return to time zero."""
        self._concentration[:] = 0.0
        self._time_s = 0.0

    def step(self, dt_s: float, injection: Injection | None = None, *, agent: ContrastAgent | None = None) -> None:
        """Advance the transport by ``dt_s``, substepping to stay inside the stability limit.

        Args:
            dt_s: Interval to advance, typically the control step.
            injection: Injection whose window decides the inlet concentration.
            agent: The medium, for its stock concentration. Defaults to neat iodinated contrast.

        Raises:
            ValueError: If ``dt_s`` is negative.
        """
        if dt_s < 0.0:
            raise ValueError(f"dt_s must not be negative, got {dt_s}")
        agent = agent or ContrastAgent()
        remaining = float(dt_s)
        while remaining > 1.0e-12:
            substep = min(remaining, self._max_substep_s)
            self._substep(substep, injection, agent)
            remaining -= substep
            self._time_s += substep

    def _substep(self, dt_s: float, injection: Injection | None, agent: ContrastAgent) -> None:
        concentration = self._concentration
        # The root is fed from outside the model, by blood that carries no contrast. Falling
        # back to the root's own concentration instead would make it a perfect reservoir with
        # nothing to flush it, and the bolus would never wash out.
        upstream = np.where(self._parent >= 0, concentration[self._parent], 0.0)

        if injection is not None:
            injected = injection.flow_at(self._time_s) * 1.0e3
            if injected > 0.0:
                # The injectate mixes with the blood already passing the tip, so what enters the
                # stream is the flow-weighted mean rather than the syringe concentration. This
                # is why a faster injection into the same vessel is brighter, and why a slow one
                # into a high-flow vessel washes out no matter how much is given. Delivering at
                # the tip replaces what the tip's parent would otherwise have fed it, so the
                # bolus travels downstream from there and never retrograde to the root.
                site = int(injection.node)
                blood_flow = self._flow_mm3_per_s[site]
                stock = agent.stock_mg_per_ml * injection.dilution
                upstream[site] = stock * injected / (injected + blood_flow)

        # First-order upwind: each node is fed by its parent at the edge velocity.
        advection = (self._velocity_mm_per_s / self._length_mm) * (upstream - concentration)

        # Dispersion over the graph's own neighbours, so a bifurcation spreads into both branches.
        laplacian = np.zeros_like(concentration)
        interior = self._order[1:]
        parents = self._parent[interior]
        scale = 1.0 / np.square(self._length_mm[interior])
        delta = (concentration[parents] - concentration[interior]) * scale
        laplacian[interior] += delta
        np.add.at(laplacian, parents, -delta)

        self._concentration = np.maximum(concentration + dt_s * (advection + self._dispersion * laplacian), 0.0)

    def mu_excess_per_mm(self, *, baseline_mu: float, curve: HuToMu, agent: ContrastAgent | None = None) -> np.ndarray:
        """Attenuation each node's contrast adds over the carve's fill material.

        This is what the per-frame compositing integrates along a ray, and the only quantity
        the renderer needs out of the transport.
        """
        agent = agent or ContrastAgent()
        return agent.mu_excess(self._concentration, baseline_mu=baseline_mu, curve=curve)


def _adjacency(edges: np.ndarray, node_count: int) -> list[list[int]]:
    """Neighbour lists for an undirected centerline graph, dropping self-edges."""
    neighbours: list[list[int]] = [[] for _ in range(node_count)]
    for left, right in edges:
        if left != right:
            neighbours[int(left)].append(int(right))
            neighbours[int(right)].append(int(left))
    return neighbours


def _widest_endpoint(edges: np.ndarray, radii: np.ndarray) -> int:
    """Best guess at the vessel's inflow: the widest node with only one neighbour.

    An arterial tree tapers away from its inflow, so the widest endpoint is normally the root.
    It is a guess, and a weak one where a segmentation has been cropped mid-vessel so that both
    ends are similar in caliber -- on a cropped aorta the two iliac stumps and the root can sit
    within a millimetre of each other. The honest fix is for the twin manifest to record which
    end blood enters at; until it does, callers who know should pass ``inflow``.
    """
    neighbours = _adjacency(edges, len(radii))
    endpoints = [node for node, adjacent in enumerate(neighbours) if len(adjacent) == 1]
    if not endpoints:
        return int(np.argmax(radii))
    return int(endpoints[int(np.argmax(radii[endpoints]))])


def _root_at(edges: np.ndarray, node_count: int, root: int) -> tuple[np.ndarray, np.ndarray]:
    """Parent pointers and a root-first visit order for the centerline graph.

    Breadth-first from the root, which orients every edge downstream. A cycle, were the
    centerline to contain one, is broken at whichever edge is reached second, so the result is
    a spanning tree and the flow split below stays conservative.
    """
    neighbours = _adjacency(edges, node_count)
    parent = np.full(node_count, -1, dtype=np.int64)
    order = [root]
    seen = np.zeros(node_count, dtype=bool)
    seen[root] = True
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for neighbour in neighbours[node]:
            if not seen[neighbour]:
                seen[neighbour] = True
                parent[neighbour] = node
                order.append(neighbour)
                queue.append(neighbour)
    return parent, np.asarray(order, dtype=np.int64)


def _split_flow(parent: np.ndarray, order: np.ndarray, radii: np.ndarray, inlet_flow_mm3_per_s: float) -> np.ndarray:
    """Volumetric flow at each node, splitting by Murray's law at every branch.

    Murray's law, :math:`Q \\propto r^3`, is the division that minimises the sum of pumping
    work and the metabolic cost of maintaining blood volume, and it holds well enough across
    the arterial tree to predict how a contrast column divides at a bifurcation.
    """
    flow = np.zeros(len(radii))
    flow[order[0]] = inlet_flow_mm3_per_s

    children: list[list[int]] = [[] for _ in range(len(radii))]
    for node in order[1:]:
        children[int(parent[node])].append(int(node))

    for node in order:
        downstream = children[node]
        if not downstream:
            continue
        weights = np.power(radii[downstream], 3.0)
        total = float(weights.sum())
        share = weights / total if total > 0.0 else np.full(len(downstream), 1.0 / len(downstream))
        flow[downstream] = flow[node] * share
    return flow


__all__ = [
    "DISPERSION_MM2_PER_S",
    "HU_PER_MG_IODINE_PER_ML",
    "INLET_VELOCITY_MM_PER_S",
    "STOCK_CONCENTRATION_MG_PER_ML",
    "ContrastAgent",
    "ContrastTransport",
    "HuToMu",
    "Injection",
]
