# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CatheterProvider protocol and built-in adapters.

This module defines the :class:`CatheterProvider` protocol, which decouples
:class:`~fluorosim.simulator.FluoroSimulator` from any specific physics backend.

Any object implementing ``get_catheter_segments()`` is a valid provider::

    class MyProvider:
        def get_catheter_segments(self, env_idx: int = 0) -> CatheterSegmentData | None:
            ...

Two concrete adapters are shipped for common solver types:

* :class:`SolverCatheterAdapter` -- wraps any solver that exposes a ``.positions``
  property (``XPBDRodSolver``, ``XCathRodSolver``, ``NewtonXPBDRodSolver``).
* :class:`StaticCatheterProvider` -- holds a fixed :class:`CatheterSegmentData`
  (useful for testing, replaying logged trajectories, or static phantom catheters).

Usage with FluoroSimulator::

    from fluorosim import FluoroSimulator
    from fluorosim.catheter_provider import SolverCatheterAdapter
    from fluorosim.catheter import XPBDRodSolver

    solver = XPBDRodSolver(cfg)
    provider = SolverCatheterAdapter(solver, radii=0.5, mu_values=1.5)

    sim = FluoroSimulator(volume, config)
    sim.set_catheter_provider(provider)

    # Render loop -- catheter geometry is fetched from the solver automatically
    for pose in trajectory:
        solver.step(dt)
        frame = sim.render_frame(pose=pose)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .rendering.diffdrr_slang_renderer import CatheterSegmentData

__all__ = [
    "CatheterProvider",
    "SolverCatheterAdapter",
    "StaticCatheterProvider",
]


@runtime_checkable
class CatheterProvider(Protocol):
    """Protocol for objects that supply catheter geometry to the renderer.

    Any class implementing this single method can be passed to
    :meth:`~fluorosim.simulator.FluoroSimulator.set_catheter_provider`.

    Args:
        env_idx: Index of the parallel environment to query.  For single-env
            solvers this is always 0 and can be ignored.

    Returns:
        :class:`~fluorosim.rendering.diffdrr_slang_renderer.CatheterSegmentData`
        for the requested environment, or ``None`` if the catheter should be
        omitted from the rendered frame.
    """

    def get_catheter_segments(self, env_idx: int = 0) -> "CatheterSegmentData | None": ...


class SolverCatheterAdapter:
    """Adapts any rod solver with a ``.positions`` property to :class:`CatheterProvider`.

    Compatible solver types (any object whose ``.positions`` returns a
    ``(N, 3)`` or ``(num_envs, N, 3)`` array-like):

    * ``XPBDRodSolver``
    * ``XCathRodSolver``
    * ``NewtonXPBDRodSolver``
    * ``RodSolver`` (legacy)

    The adapter does **not** import the solver at construction time, so the
    ``fluorosim.simulator`` module stays free of physics-backend imports.

    Args:
        solver:    Any solver object with a ``.positions`` attribute.
        radii:     Cylinder radius per segment in mm.  Scalar or ``(N-1,)`` array.
        mu_values: Linear attenuation coefficient per segment in mm^-1.
                   Scalar or ``(N-1,)`` array.
                   Typical values: nitinol shaft ~0.8, platinum marker ~5.0.
        scale:     Multiplicative scale applied to positions before passing them
                   to the renderer.  Set to ``1000.0`` if the solver works in
                   **metres** and the renderer expects **millimetres** (common for
                   Warp/Isaac Lab solvers).  Defaults to 1.0 (no conversion).

    Example::

        adapter = SolverCatheterAdapter(solver, radii=0.5, mu_values=1.5, scale=1000.0)
        sim.set_catheter_provider(adapter)
    """

    def __init__(
        self,
        solver,
        radii: Union[float, np.ndarray] = 0.5,
        mu_values: Union[float, np.ndarray] = 1.0,
        scale: float = 1.0,
    ) -> None:
        self._solver = solver
        self._radii = radii
        self._mu_values = mu_values
        self._scale = scale

    def get_catheter_segments(self, env_idx: int = 0) -> "CatheterSegmentData":
        """Extract current positions from the solver and wrap in CatheterSegmentData.

        The solver's ``.positions`` tensor is copied to CPU numpy to avoid
        holding onto live GPU memory after this call returns.

        Args:
            env_idx: Environment index for batched solvers.

        Returns:
            :class:`~fluorosim.rendering.diffdrr_slang_renderer.CatheterSegmentData`
            populated with the solver's current particle positions.
        """
        from .rendering.diffdrr_slang_renderer import CatheterSegmentData

        raw = self._solver.positions

        # Support torch tensors, warp arrays, and plain numpy arrays.
        if hasattr(raw, "cpu"):
            # torch.Tensor  (XPBDRodSolver returns wp.to_torch(...).clone())
            pos = raw.cpu().numpy()
        elif hasattr(raw, "numpy"):
            # warp array or similar
            pos = raw.numpy()
        else:
            pos = np.asarray(raw)

        # Shape: (num_envs, N, 3) -- select the requested environment row
        if pos.ndim == 3:
            pos = pos[env_idx]

        pos = pos.astype(np.float32)

        if self._scale != 1.0:
            pos = pos * self._scale

        return CatheterSegmentData(
            positions=pos,
            radii=self._radii,
            mu_values=self._mu_values,
        )


class StaticCatheterProvider:
    """Holds a fixed :class:`CatheterSegmentData` returned on every call.

    Useful for:

    * Unit tests that need deterministic catheter geometry.
    * Replaying pre-recorded catheter trajectories frame-by-frame.
    * Static phantom catheters that do not move during a render sequence.

    Args:
        catheter: The :class:`CatheterSegmentData` to return.  Pass ``None``
                  to disable the catheter (``get_catheter_segments`` returns ``None``).

    Example::

        positions = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float32)
        catheter = CatheterSegmentData(positions=positions, radii=0.5, mu_values=1.5)
        sim.set_catheter_provider(StaticCatheterProvider(catheter))
    """

    def __init__(self, catheter: "CatheterSegmentData | None") -> None:
        self._catheter = catheter

    def get_catheter_segments(self, env_idx: int = 0) -> "CatheterSegmentData | None":
        return self._catheter

    def update(self, catheter: "CatheterSegmentData | None") -> None:
        """Replace the stored catheter geometry at runtime."""
        self._catheter = catheter
