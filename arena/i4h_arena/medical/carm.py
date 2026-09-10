# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""C-arm state boundary between Isaac scene assets and image formation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .patient_volume import PatientVolume


def _numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value)


def _quat_xyzw_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate one vector per environment by an Isaac Lab XYZW quaternion."""
    xyz = quaternion[:, :3]
    w = quaternion[:, 3:4]
    vec = np.broadcast_to(np.asarray(vector, dtype=np.float64), xyz.shape)
    return vec + 2.0 * np.cross(xyz, np.cross(xyz, vec) + w * vec)


SUPPORTED_PATIENT_FRAME = "DICOM_LPS"


def anatomical_projection_basis(twin: Any) -> np.ndarray:
    """World-space columns ``(detector u, orbit axis, beam)`` for the frontal view.

    The projection frame has to come from the patient's anatomy, because that is
    what the view names mean: AP is a beam along anterior-posterior, and LAO/RAO
    sweep about the body's long axis. Deriving it from the volume's storage axes
    instead ties both to however the CT happened to be written. On ``s0011``
    those axes map to patient left, posterior and superior, which put the AP beam
    along the patient's *superior* axis -- an axial projection down the length of
    the body, with the orbit sweeping toward the left rather than around it. No
    orbit angle reached a true AP.

    Columns are ordered to match what the caller does with them: the beam is the
    third, so an orbit about the local Y axis is an orbit about the second, and
    the first is the detector's horizontal axis.

    Sign choices, given that the renderer fixes the vertical axis as
    ``beam x u`` and so leaves only one free choice:

    - Beam along **anterior**, i.e. the tube below the table and the detector
      above it, which is how the frontal view is acquired in a cath lab and how
      the visible rig in the scene is posed.
    - Detector horizontal along patient **right**, which is what makes the
      vertical axis come out inferior and so puts the head at the top of the
      image. The cost is that the image reads as if viewed from behind, where
      radiographic convention shows the patient's right on the viewer's left.
      Those two cannot both hold here: flipping to patient left to recover the
      convention flips the vertical axis with it and stands the patient on their
      head. Left-right mirroring is a display convention that can be undone on
      the image; the beam and orbit axes cannot.

    Args:
        twin: Patient twin carrying ``coordinate_frame`` and
            ``world_from_patient_m``.

    Returns:
        ``(3, 3)`` orthonormal right-handed matrix whose columns are world-space
        directions.

    Raises:
        ValueError: If the twin declares an unsupported coordinate frame, or its
            patient-to-world transform is not a rotation.
    """
    frame = str(getattr(twin, "coordinate_frame", ""))
    if frame != SUPPORTED_PATIENT_FRAME:
        raise ValueError(
            f"patient twin declares coordinate_frame={frame!r}; the projection basis is only "
            f"defined for {SUPPORTED_PATIENT_FRAME!r}, whose axes are left, posterior, superior"
        )
    rotation = np.asarray(twin.world_from_patient_m, dtype=np.float64)[:3, :3]
    norms = np.linalg.norm(rotation, axis=0)
    if not np.isfinite(norms).all() or norms.min() <= 1.0e-9:
        raise ValueError("patient twin world_from_patient_m has a degenerate rotation")
    rotation = rotation / norms
    # Columns of world_from_patient are the world directions of the patient axes,
    # which in LPS are left, posterior and superior.
    left, posterior, superior = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    basis = np.column_stack((-left, -superior, -posterior))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1.0e-6):
        raise ValueError("patient twin world_from_patient_m is not orthonormal; cannot form a projection basis")
    if float(np.linalg.det(basis)) < 0.0:
        raise ValueError("patient twin world_from_patient_m is a reflection; the projection basis would mirror anatomy")
    return basis


@dataclass(frozen=True, slots=True)
class CArmState:
    """World-space source and detector geometry for every environment."""

    source_world_m: np.ndarray
    detector_center_world_m: np.ndarray
    detector_x_axis_world: np.ndarray
    detector_size_m: tuple[float, float]

    def __post_init__(self) -> None:
        source = np.asarray(self.source_world_m, dtype=np.float64)
        detector = np.asarray(self.detector_center_world_m, dtype=np.float64)
        x_axis = np.asarray(self.detector_x_axis_world, dtype=np.float64)
        if source.ndim != 2 or source.shape[-1] != 3:
            raise ValueError("source_world_m must have shape (num_envs, 3)")
        if detector.shape != source.shape or x_axis.shape != source.shape:
            raise ValueError("detector position and x axis must match source shape")
        if not np.isfinite(source).all() or not np.isfinite(detector).all() or not np.isfinite(x_axis).all():
            raise ValueError("C-arm state must contain only finite values")
        norms = np.linalg.norm(x_axis, axis=-1)
        if np.any(norms < 1e-9):
            raise ValueError("detector x axis must be non-zero")
        if len(self.detector_size_m) != 2 or min(self.detector_size_m) <= 0.0:
            raise ValueError("detector_size_m must contain positive width and height")
        object.__setattr__(self, "source_world_m", source)
        object.__setattr__(self, "detector_center_world_m", detector)
        object.__setattr__(self, "detector_x_axis_world", x_axis / norms[:, None])

    @property
    def num_envs(self) -> int:
        return int(self.source_world_m.shape[0])


@runtime_checkable
class CArmStateProvider(Protocol):
    def snapshot(self, num_envs: int) -> CArmState:
        """Return current C-arm geometry without advancing the simulator."""


class SceneCArmStateProvider:
    """Read C-arm poses from Isaac Lab scene assets."""

    def __init__(self, source_asset: Any, detector_asset: Any, *, detector_size_m: tuple[float, float]) -> None:
        self._source_asset = source_asset
        self._detector_asset = detector_asset
        self._detector_size_m = detector_size_m

    def snapshot(self, num_envs: int) -> CArmState:
        source_pos, _source_quat = self._source_asset.get_world_poses()
        detector_pos, detector_quat = self._detector_asset.get_world_poses()
        source = _numpy(source_pos)[..., :3]
        detector = _numpy(detector_pos)[..., :3]
        quaternion = _numpy(detector_quat)
        if source.shape[0] != num_envs or detector.shape[0] != num_envs:
            raise ValueError(f"C-arm provider returned {source.shape[0]} environments; expected {num_envs}")
        x_axis = _quat_xyzw_rotate(quaternion, np.array([1.0, 0.0, 0.0]))
        return CArmState(source, detector, x_axis, self._detector_size_m)


_LOGGER = logging.getLogger(__name__)

FOLLOW_TIP_ENV_VAR = "I4H_CARM_FOLLOW_TIP"


def follow_tip_enabled() -> bool:
    """Whether the isocenter should track the catheter tip.

    On by default. A fixed isocenter leaves roughly 40% of an ``s0011`` run with
    the tip off the detector, which an operator can work around using the
    distance readout but a policy cannot: those frames pair an action with an
    image that does not contain the thing being moved.

    Panning is the first thing here to give the renderer a non-zero pose
    translation -- every earlier projection put the isocenter on the volume
    centre, and the tests assert that translation is zero -- so it was verified
    by rendering rather than by arithmetic. An 85.3 mm pan moves the spine
    79.2 mm on the detector, and the 7% shortfall is parallax: the residual puts
    the spine about 39 mm posterior of the isocenter, which is what an AP abdomen
    should give. Set this variable falsey to pin the isocenter to the volume
    centre.
    """
    setting = os.environ.get(FOLLOW_TIP_ENV_VAR)
    if setting is None:
        return True
    return setting.strip().lower() in {"1", "true", "yes", "on"}


def panned_isocenter_offsets(
    tip_offsets_m: np.ndarray,
    current_pan_m: np.ndarray,
    *,
    half_fov_m: float,
    keep_fraction: float,
    limits_m: tuple[float, float],
) -> np.ndarray:
    """Follow the catheter tip along one axis, the way a table pans.

    A fixed isocenter leaves most of a long route unobserved: on ``s0011`` the
    detector covers 307 mm of a 510 mm run, so roughly 40% of every episode has
    the tip off the frame. That is survivable for an operator, who still has the
    distance readout, and fatal for imitation learning -- a demonstration whose
    image does not contain the tip pairs an action with no visible cause, and
    teaches a policy that confident motion is appropriate when it cannot see.

    The follow is hysteretic rather than continuous, for two reasons. Continuous
    centering would pin the tip to the middle of every frame, which throws away
    tip position within the frame as a signal and leaves only the anatomy
    scrolling past. And the tip jitters, so a frame locked to it would shake.
    Instead nothing moves while the tip stays inside ``keep_fraction`` of the
    half-field, and beyond that the pan brings it back exactly to that boundary
    -- which is also how the pan reads in a lab, as occasional repositioning
    rather than a tracking shot.

    Args:
        tip_offsets_m: Per-environment tip position along the pan axis, measured
            from the unpanned isocenter.
        current_pan_m: Per-environment pan offset in force now. Carried between
            calls because the rule is hysteretic and has no fixed point of its
            own.
        half_fov_m: Half the field of view along the pan axis at the isocenter
            plane, which is where the anatomy is.
        keep_fraction: Fraction of ``half_fov_m`` the tip may wander over before
            the frame moves, in ``[0, 1]``.
        limits_m: Inclusive clamp on the pan, keeping the beam on the scanned
            volume. Panning past the anatomy would render empty air.

    Returns:
        The new per-environment pan offsets.
    """
    margin = max(0.0, min(1.0, float(keep_fraction))) * float(half_fov_m)
    # Where the tip sits within the frame as currently positioned.
    error = np.asarray(tip_offsets_m, dtype=np.float64) - np.asarray(current_pan_m, dtype=np.float64)
    overshoot = np.clip(np.abs(error) - margin, 0.0, None)
    panned = np.asarray(current_pan_m, dtype=np.float64) + np.sign(error) * overshoot
    # A tip that is nowhere yet, before Newton has particles, must not drag the
    # frame off to infinity.
    panned = np.where(np.isfinite(panned), panned, np.asarray(current_pan_m, dtype=np.float64))
    return np.clip(panned, float(limits_m[0]), float(limits_m[1]))


class ReferenceProjectionCArmStateProvider:
    """Calibrate a visible C-arm angle to the reference xray_simulator projections.

    The visible assembly keeps its intuitive patient-surrounding motion. The
    shared orbit angle independently defines the renderer's AP/LAO/lateral/RAO
    coordinate frame, avoiding a visual equipment pose dictated by CT storage
    axes.

    The frame comes from :func:`anatomical_projection_basis`, which is the part
    that makes those view names true. Building it from the volume's own axes is
    the CT-storage dependence this class exists to avoid, and it does not merely
    rotate the picture: on ``s0011`` it aimed the AP beam down the length of the
    body and left a true AP unreachable at any orbit angle.
    """

    def __init__(
        self,
        patient: PatientVolume,
        orbit_action: Any,
        *,
        detector_size_m: tuple[float, float],
        source_to_detector_m: float = 1.020,
        tip_source: Any = None,
        pan_keep_fraction: float = 0.6,
    ) -> None:
        self._patient = patient
        self._orbit_action = orbit_action
        self._detector_size_m = detector_size_m
        self._half_sdd_m = 0.5 * float(source_to_detector_m)
        # Left unset the isocenter stays on the volume centre, which is the
        # behaviour every projection test was written against.
        self._tip_source = tip_source
        self._pan_keep_fraction = float(pan_keep_fraction)
        self._pan_m: np.ndarray | None = None
        # A handful of frames is enough to see whether the live tip agrees with
        # the centerline, without flooding a teleop session's log.
        self._pan_log_countdown = 5

    def _pan_axis_limits_m(self, isocenter: np.ndarray, pan_axis: np.ndarray) -> tuple[float, float]:
        """How far the isocenter may pan before the beam leaves the scanned volume.

        Derived from the volume's own corners rather than a constant, so it holds
        for any twin and any anatomical basis: project all eight into world,
        take their span along the pan axis, and measure it from the unpanned
        isocenter. Panning further would put the beam in empty air and render a
        blank detector, which reads as a broken sensor.
        """
        extent_mm = 2.0 * np.asarray(self._patient.center_xyz_mm, dtype=np.float64)
        corners_mm = np.array(
            [[x, y, z] for x in (0.0, extent_mm[0]) for y in (0.0, extent_mm[1]) for z in (0.0, extent_mm[2])],
            dtype=np.float64,
        )
        offsets = (self._patient.volume_mm_to_world(corners_mm) - isocenter) @ pan_axis
        return float(offsets.min()), float(offsets.max())

    def _tip_offsets_m(self, num_envs: int, isocenter: np.ndarray, pan_axis: np.ndarray) -> np.ndarray | None:
        """Tip position along the pan axis, or ``None`` when there is no tip to read."""
        if self._tip_source is None:
            return None
        positions = getattr(getattr(self._tip_source, "data", None), "positions_world_m", None)
        if positions is None:
            return None
        tips = _numpy(positions).reshape(num_envs, -1, 3)[:, -1, :]
        return (tips - isocenter) @ pan_axis

    def snapshot(self, num_envs: int) -> CArmState:
        angles = _numpy(self._orbit_action.angle_rad).reshape(-1)
        if angles.shape != (num_envs,):
            raise ValueError(f"C-arm orbit action returned {angles.shape[0]} environments; expected {num_envs}")
        patient_to_world = anatomical_projection_basis(self._patient.twin)
        isocenter = self._patient.volume_mm_to_world(self._patient.center_xyz_mm)
        # The basis puts the patient's head-foot axis on the detector's vertical,
        # and orbit is a rotation about that same axis, so the pan direction is
        # the one part of the frame every view agrees on. Panning along it is
        # what a table does; panning laterally is not, and is not needed here
        # since the route's lateral spread fits the field several times over.
        pan_axis = patient_to_world[:, 1]
        isocenters = np.broadcast_to(isocenter, (num_envs, 3)).astype(np.float64, copy=True)
        tip_offsets = self._tip_offsets_m(num_envs, isocenter, pan_axis)
        if tip_offsets is not None:
            if self._pan_m is None or self._pan_m.shape != (num_envs,):
                self._pan_m = np.zeros(num_envs, dtype=np.float64)
            self._pan_m = panned_isocenter_offsets(
                tip_offsets,
                self._pan_m,
                # Vertical half-field at the isocenter, which is where the
                # anatomy is. The isocenter sits halfway along a beam converging
                # on the source, so it sees half the detector's extent: a
                # 614 mm detector covers 307 mm of patient.
                half_fov_m=0.25 * self._detector_size_m[1],
                keep_fraction=self._pan_keep_fraction,
                limits_m=self._pan_axis_limits_m(isocenter, pan_axis),
            )
            isocenters += self._pan_m[:, None] * pan_axis
            # Static analysis says this geometry is sound -- the pan is
            # perpendicular to the beam, the rotation is untouched, and the
            # renderer applies the translation in the volume frame. The first
            # live frame said otherwise, so report what the tip actually reads
            # rather than what the centerline predicts it should.
            if self._pan_log_countdown > 0:
                self._pan_log_countdown -= 1
                _LOGGER.info(
                    "carm tip-follow: pan_axis=%s tip_offset_mm=%s pan_mm=%s limits_mm=%s",
                    np.round(pan_axis, 3).tolist(),
                    np.round(tip_offsets * 1000.0, 1).tolist(),
                    np.round(self._pan_m * 1000.0, 1).tolist(),
                    [round(v * 1000.0, 1) for v in self._pan_axis_limits_m(isocenter, pan_axis)],
                )
        source = np.zeros((num_envs, 3), dtype=np.float64)
        detector = np.zeros_like(source)
        detector_x = np.zeros_like(source)
        for index, angle in enumerate(angles):
            # Negated so a positive angle carries the detector toward the
            # patient's left, which is what LAO names. The basis puts the
            # detector's horizontal axis along patient right, so the unnegated
            # rotation would make positive angles RAO and invert every preset.
            cosine = np.cos(-float(angle))
            sine = np.sin(-float(angle))
            renderer_orbit = np.array(
                [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
                dtype=np.float64,
            )
            local_to_world = patient_to_world @ renderer_orbit
            beam_axis = local_to_world[:, 2]
            source[index] = isocenters[index] - self._half_sdd_m * beam_axis
            detector[index] = isocenters[index] + self._half_sdd_m * beam_axis
            detector_x[index] = local_to_world[:, 0]
        return CArmState(source, detector, detector_x, self._detector_size_m)

    def select_angle(self, angle_rad: float) -> float:
        """Select one reference projection on the shared visible C-arm action."""
        setter = getattr(self._orbit_action, "set_orbit_angle", None)
        if not callable(setter):
            raise TypeError("the C-arm orbit action does not support named projections")
        return float(setter(angle_rad))
