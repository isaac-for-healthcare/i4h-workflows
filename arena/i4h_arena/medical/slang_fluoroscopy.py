# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patient-backed Slang DRR adapter for the Arena fluoroscopy sensor."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from xray_simulator.config import DisplaySettings
from xray_simulator.display import apply_display, calibrate_display

from .carm import CArmState
from .catheter import CatheterState
from .catheter_attenuation import CatheterAttenuation, CatheterMaterial
from .centerline import ordered_centerline_path
from .fluoroscopy_guidance import draw_catheter_guidance
from .patient_volume import PatientVolume

logger = logging.getLogger("i4h_arena.fluoroscopy")

# Window adjustments are expressed against the window fitted from the first frame rather than
# in absolute line-integral units, because the useful range depends on patient size and on the
# μ scaling baked into the twin. A slider in absolute units would need different bounds per
# patient; a multiple of the fitted width behaves the same way on every twin.
_DISPLAY_CONTROL_BOUNDS = {
    "window_level": (-1.0, 1.0),
    "window_width": (0.25, 4.0),
}


def _contrast_bolus_mask(patient: PatientVolume) -> np.ndarray:
    """Build the thin centerline-derived mask used by the reference DSA view."""
    centerline_path = patient.twin.artifacts.get("centerline_points")
    if centerline_path is None:
        vessel_path = patient.twin.artifacts.get("vessel_mask")
        if vessel_path is None:
            raise ValueError("DSA fluoroscopy requires centerline_points or vessel_mask in the patient twin")
        return np.asarray(np.load(vessel_path), dtype=np.uint8)

    from scipy.ndimage import binary_dilation, generate_binary_structure

    points_patient_mm = np.asarray(np.load(centerline_path), dtype=np.float64)
    edges_path = patient.twin.artifacts.get("centerline_edges")
    radii_path = patient.twin.artifacts.get("centerline_radii")
    if edges_path is not None:
        reference_segment_mm = 0.65 * float(patient.shape_zyx[2]) * float(patient.spacing_xyz_mm[0]) / 40.0
        points_patient_mm = ordered_centerline_path(
            points_patient_mm,
            np.load(edges_path),
            target_spacing_mm=reference_segment_mm,
            radii_mm=np.load(radii_path) if radii_path is not None else None,
        )
    points_world_m = patient.twin.patient_mm_to_world(points_patient_mm)
    points_volume_mm = patient.world_to_volume_mm(points_world_m)
    spacing_xyz = np.asarray(patient.spacing_xyz_mm, dtype=np.float64)
    indices_xyz = np.rint(points_volume_mm / spacing_xyz).astype(np.int64)
    shape_xyz = np.asarray(patient.shape_zyx[::-1], dtype=np.int64)
    valid = np.all((indices_xyz >= 0) & (indices_xyz < shape_xyz), axis=1)
    indices_xyz = indices_xyz[valid]
    mask = np.zeros(patient.shape_zyx, dtype=np.uint8)
    mask[indices_xyz[:, 2], indices_xyz[:, 1], indices_xyz[:, 0]] = 1

    # Match the reference's tight connected guide tube rather than boosting
    # the complete artery tree, which produces a dark projected blob.
    dilation_mm = 1.2
    iterations = max(1, int(round(dilation_mm / min(patient.spacing_xyz_mm))))
    return binary_dilation(
        mask,
        structure=generate_binary_structure(3, 1),
        iterations=iterations,
    ).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class ProjectionGeometry:
    """Slang pose plus matrices needed to project the catheter consistently."""

    rotation_zxy_rad: tuple[float, float, float]
    translation_xyz_mm: tuple[float, float, float]
    source_to_detector_mm: float
    source_to_isocenter_mm: float
    pixel_spacing_mm: float
    local_to_volume: np.ndarray
    isocenter_volume_mm: np.ndarray


def _matrix_to_zxy(matrix: np.ndarray) -> tuple[float, float, float]:
    """Invert the renderer's Rz * Rx * Ry Euler convention."""
    sx = float(np.clip(matrix[2, 1], -1.0, 1.0))
    rx = float(np.arcsin(sx))
    if abs(float(np.cos(rx))) < 1e-7:
        ry = 0.0
        rz = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        ry = float(np.arctan2(-matrix[2, 0], matrix[2, 2]))
        rz = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
    return rx, ry, rz


def solve_projection_geometry(
    patient: PatientVolume,
    carm: CArmState,
    *,
    width: int,
    height: int,
    env_index: int = 0,
) -> ProjectionGeometry:
    """Convert Isaac C-arm prim poses into the renderer's centred volume coordinates."""
    source = patient.world_to_volume_mm(carm.source_world_m[env_index])
    detector = patient.world_to_volume_mm(carm.detector_center_world_m[env_index])
    axis_end = patient.world_to_volume_mm(
        carm.detector_center_world_m[env_index] + carm.detector_x_axis_world[env_index]
    )
    z_axis = detector - source
    sdd = float(np.linalg.norm(z_axis))
    if sdd < 1e-6:
        raise ValueError("C-arm source and detector center must be distinct")
    z_axis /= sdd
    x_axis = axis_end - detector
    x_axis -= z_axis * float(np.dot(x_axis, z_axis))
    if np.linalg.norm(x_axis) < 1e-8:
        candidate = np.eye(3)[int(np.argmin(np.abs(z_axis)))]
        x_axis = candidate - z_axis * float(np.dot(candidate, z_axis))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    local_to_volume = np.column_stack((x_axis, y_axis, z_axis))

    sid = 0.5 * sdd
    isocenter = source + sid * z_axis
    translation = isocenter - patient.center_xyz_mm
    pixel_spacing_x = 1000.0 * float(carm.detector_size_m[0]) / float(width)
    pixel_spacing_y = 1000.0 * float(carm.detector_size_m[1]) / float(height)
    if not np.isclose(pixel_spacing_x, pixel_spacing_y, rtol=1e-3, atol=1e-6):
        raise ValueError("the upstream xray_simulator renderer currently requires square detector pixels")
    return ProjectionGeometry(
        rotation_zxy_rad=_matrix_to_zxy(local_to_volume),
        translation_xyz_mm=tuple(float(value) for value in translation),
        source_to_detector_mm=sdd,
        source_to_isocenter_mm=sid,
        pixel_spacing_mm=0.5 * (pixel_spacing_x + pixel_spacing_y),
        local_to_volume=local_to_volume,
        isocenter_volume_mm=isocenter,
    )


class SlangFluoroscopyRenderer:
    """Render patient attenuation with upstream xray_simulator and composite catheter geometry."""

    def __init__(
        self,
        patient: PatientVolume,
        carm: CArmState,
        *,
        width: int,
        height: int,
        step_mm: float,
        device_type: str,
        dsa: bool = False,
        dsa_boost: float = 6.0,
        dsa_gamma: float = 1.0,
        visual_style: str = "default",
        display: DisplaySettings | None = None,
        catheter_attenuation: bool = True,
        catheter_material: CatheterMaterial | None = None,
        catheter_device: str | None = None,
    ) -> None:
        if patient.twin.coordinate_frame not in {"DICOM_LPS", "NIFTI_RAS"}:
            raise ValueError(f"unsupported patient coordinate frame {patient.twin.coordinate_frame!r}")
        self._patient = patient
        self._display = display if display is not None else DisplaySettings()
        self._display_calibrated = False
        self._base_log_window: tuple[float, float] | None = None
        self._window_level = 0.0
        self._window_width = 1.0
        self._catheter_attenuation = (
            CatheterAttenuation(catheter_material, device=catheter_device) if catheter_attenuation else None
        )
        self.width = int(width)
        self.height = int(height)
        self._dsa = bool(dsa)
        self._dsa_gamma = float(np.clip(dsa_gamma, 0.3, 3.0))
        self._visual_style = str(visual_style)
        if self._visual_style not in {"default", "cinematic"}:
            raise ValueError("visual_style must be 'default' or 'cinematic'")
        self._temporal_frames: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(42)
        self._projection = solve_projection_geometry(patient, carm, width=width, height=height)
        logger.info(
            "initial projection rotation_zxy_deg=%s translation_mm=%s source_world_m=%s detector_world_m=%s",
            tuple(round(float(np.degrees(value)), 3) for value in self._projection.rotation_zxy_rad),
            tuple(round(float(value), 3) for value in self._projection.translation_xyz_mm),
            tuple(round(float(value), 6) for value in carm.source_world_m[0]),
            tuple(round(float(value), 6) for value in carm.detector_center_world_m[0]),
        )

        from xray_simulator.rendering.diffdrr_slang_renderer import SlangDiffDRRConfig, SlangDiffDRRRenderer

        projection = self._projection
        base_config = SlangDiffDRRConfig(
            det_height_px=self.height,
            det_width_px=self.width,
            pixel_spacing_mm=projection.pixel_spacing_mm,
            source_to_detector_mm=projection.source_to_detector_mm,
            source_to_isocenter_mm=projection.source_to_isocenter_mm,
            step_mm=float(step_mm),
            normalize=False,
            invert=False,
            device_type=device_type,
        )
        # The renderer takes origin_xyz_mm ahead of the config, and leaving it at the default
        # keeps the volume centred, which is the frame solve_projection_geometry poses into.
        self._renderer = SlangDiffDRRRenderer(patient.mu_volume, patient.spacing_zyx_mm, cfg=base_config)
        self._dsa_renderer = None
        if self._dsa:
            vessel_mask = _contrast_bolus_mask(patient)
            if vessel_mask.shape != patient.mu_volume.shape:
                raise ValueError(
                    f"contrast bolus mask shape {vessel_mask.shape} does not match attenuation volume "
                    f"{patient.mu_volume.shape}"
                )
            boosted_volume = patient.mu_volume.copy()
            boosted_volume[vessel_mask > 0] *= float(dsa_boost)
            self._dsa_renderer = SlangDiffDRRRenderer(boosted_volume, patient.spacing_zyx_mm, cfg=base_config)

    def render(self, catheter: CatheterState, carm: CArmState | None = None) -> dict[str, np.ndarray]:
        if catheter.num_envs != 1:
            raise ValueError("the pinned xray_simulator release currently supports one Arena environment")
        projection = self._projection
        if carm is not None:
            current = solve_projection_geometry(self._patient, carm, width=self.width, height=self.height)
            if not np.isclose(current.source_to_detector_mm, projection.source_to_detector_mm, atol=0.1):
                raise ValueError("changing source-detector distance requires rebuilding the fluoroscopy sensor")
            projection = current
        transmission = self._renderer.render(projection.rotation_zxy_rad, projection.translation_xyz_mm)
        valid = int(catheter.valid_nodes[0])
        catheter_transmission = self._catheter_transmission(catheter, valid, projection)
        if catheter_transmission is not None:
            transmission = transmission * catheter_transmission
        if not self._display_calibrated:
            self._calibrate_display(transmission)
        drr_image = apply_display(transmission, self._display)
        dsa_image = drr_image
        vessel_alpha = None
        if self._dsa_renderer is not None:
            boosted_transmission = self._dsa_renderer.render(projection.rotation_zxy_rad, projection.translation_xyz_mm)
            if catheter_transmission is not None:
                boosted_transmission = boosted_transmission * catheter_transmission
            dsa_image = apply_display(boosted_transmission, self._display)
            vessel_signal = np.clip(transmission - boosted_transmission, 0.0, None)
            scale = float(np.percentile(vessel_signal, 99.5))
            if scale > 1e-8:
                vessel_alpha = np.clip(vessel_signal / scale, 0.0, 0.82).astype(np.float32)

        grayscale = np.clip(np.rint(255.0 * drr_image), 0.0, 255.0).astype(np.uint8)
        dsa_grayscale = np.clip(np.rint(255.0 * dsa_image), 0.0, 255.0).astype(np.uint8)
        if self._visual_style == "cinematic":
            grayscale = self._apply_cinematic_style(grayscale, key="drr")
            dsa_grayscale = self._apply_cinematic_style(dsa_grayscale, key="dsa")
        if abs(self._dsa_gamma - 1.0) > 1e-3:
            normalized_dsa = dsa_grayscale.astype(np.float32) / 255.0
            dsa_grayscale = np.rint(255.0 * np.power(normalized_dsa, self._dsa_gamma)).astype(np.uint8)
        rgb = np.repeat(grayscale[None, ..., None], 3, axis=-1)
        dsa_rgb = np.repeat(dsa_grayscale[None, ..., None], 3, axis=-1)
        guidance = rgb.copy()
        if vessel_alpha is not None:
            guidance[0] = self._draw_vessel_overlay(guidance[0], vessel_alpha)
        dsa_guidance = dsa_rgb.copy()
        if valid >= 2:
            pixels, visible = self._project_catheter(catheter.positions_world_m[0, :valid], projection)
            if catheter_transmission is None:
                # Without beam compositing the catheter exists only as an annotation.
                self._draw_catheter(rgb[0], pixels, visible)
                self._draw_catheter(guidance[0], pixels, visible)
                self._draw_catheter(dsa_rgb[0], pixels, visible)
            guidance[0] = draw_catheter_guidance(guidance[0], pixels, visible)
            dsa_guidance[0] = draw_catheter_guidance(dsa_rgb[0], pixels, visible)
        attenuation = (1.0 - transmission.astype(np.float32))[None, ..., None]
        return {
            "rgb": rgb,
            "guidance": guidance,
            "dsa": dsa_rgb,
            "dsa_guidance": dsa_guidance,
            "attenuation": attenuation,
        }

    def _calibrate_display(self, transmission: np.ndarray) -> None:
        """Fit the log window once, from the first frame, and then hold it fixed.

        ``calibrate_display`` rejects a frame that spans no attenuation range, which is the
        right answer for a one-off render but not part way through an episode: a sweep that
        briefly clears the patient would end the run. Keep the preset window in that case and
        retry on the next frame.
        """
        try:
            self._display = calibrate_display(transmission, self._display)
        except ValueError:
            logger.warning("fluoroscopy frame spans no attenuation range; keeping the preset display window")
            return
        self._display_calibrated = True
        self._base_log_window = self._display.log_window
        self._apply_display_window()
        logger.info(
            "fitted %s display window to line integral %s from the first frame",
            self._display.polarity,
            tuple(round(float(value), 4) for value in self._display.log_window),
        )

    def _catheter_transmission(
        self,
        catheter: CatheterState,
        valid: int,
        projection: ProjectionGeometry,
    ) -> np.ndarray | None:
        """Transmission through the catheter alone, or ``None`` when it does not attenuate.

        Returned as its own factor because the plain and contrast-boosted volumes both sit behind
        the same instrument, so one solve serves both projections.
        """
        if self._catheter_attenuation is None or valid < 2:
            return None
        points_volume_mm = self._patient.world_to_volume_mm(catheter.positions_world_m[0, :valid])
        line_integral = self._catheter_attenuation.line_integral(
            points_volume_mm,
            1000.0 * float(catheter.radius_m),
            projection,
            width=self.width,
            height=self.height,
        )
        return np.exp(-line_integral).astype(np.float32)

    def adjust_dsa_gamma(self, delta: float) -> float:
        """Adjust live angiogram brightness and return the bounded gamma."""
        self._dsa_gamma = float(np.clip(self._dsa_gamma + float(delta), 0.3, 3.0))
        return self._dsa_gamma

    def set_dsa_gamma(self, gamma: float) -> float:
        """Set live angiogram gamma and return the bounded value."""
        self._dsa_gamma = float(np.clip(gamma, 0.3, 3.0))
        return self._dsa_gamma

    @property
    def display_polarity(self) -> str:
        """Polarity the next frame will be mapped with."""
        return self._display.polarity

    def set_display_appearance(self, appearance: str) -> str:
        """Switch between the fluoroscopy and radiograph looks and return the polarity.

        Both come from the same render, so this only re-maps intensity and never re-renders.
        The named preset supplies the polarity alone: taking its window instead would discard
        the one calibrated from the first frame and flatten the image mid-run.
        """
        polarity = DisplaySettings.preset(appearance).polarity
        if polarity != self._display.polarity:
            self._display = replace(self._display, polarity=polarity)
            logger.info("switched fluoroscopy display to %s polarity", polarity)
        return self._display.polarity

    @property
    def display_window(self) -> tuple[float, float]:
        """Line-integral interval the next frame will be mapped across."""
        return self._display.log_window

    def set_display_control(self, control: str, value: float) -> float:
        """Widen, narrow or shift the display window and return the bounded setting.

        Like polarity, this only re-maps intensity, so contrast can be tuned while the
        catheter is moving.
        """
        try:
            minimum, maximum = _DISPLAY_CONTROL_BOUNDS[control]
        except KeyError:
            raise ValueError(f"unknown display control {control!r}") from None
        bounded = float(np.clip(value, minimum, maximum))
        if control == "window_level":
            self._window_level = bounded
        else:
            self._window_width = bounded
        self._apply_display_window()
        return bounded

    def recalibrate_display(self) -> None:
        """Re-fit the window to the next frame and drop manual adjustments.

        The fit holds for the body region that was in the beam at the time, so a large
        oblique or a move along the table can leave it a poor match with no way back.
        """
        self._display_calibrated = False
        self._base_log_window = None
        self._window_level = 0.0
        self._window_width = 1.0

    def _apply_display_window(self) -> None:
        base_low, base_high = self._base_log_window or self._display.log_window
        base_width = base_high - base_low
        center = 0.5 * (base_low + base_high) + self._window_level * base_width
        width = max(base_width * self._window_width, 1e-6)
        # The window is a line integral, so it cannot start below zero; hold the width and
        # slide the window instead of letting a clamp silently narrow it.
        low = max(center - 0.5 * width, 0.0)
        self._display = replace(self._display, log_window=(low, low + width))

    @staticmethod
    def _draw_vessel_overlay(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        result = image.astype(np.float32)
        result[..., 0] *= 1.0 - alpha
        result[..., 1] = result[..., 1] * (1.0 - alpha) + 255.0 * alpha
        result[..., 2] = result[..., 2] * (1.0 - alpha) + 255.0 * alpha
        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    def _apply_cinematic_style(self, grayscale: np.ndarray, *, key: str) -> np.ndarray:
        """Apply the reference viewport's cine blur, persistence, noise, and vignette."""
        source = Image.fromarray(grayscale)
        base = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=0.9)), dtype=np.float32) / 255.0
        bloom = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32) / 255.0
        frame = np.clip(0.85 * base + 0.15 * bloom, 0.0, 1.0)
        if key not in self._temporal_frames:
            self._temporal_frames[key] = frame.copy()
        else:
            self._temporal_frames[key] = 0.76 * self._temporal_frames[key] + 0.24 * frame
        styled = self._temporal_frames[key] + self._rng.normal(0.0, 0.012, frame.shape).astype(np.float32)
        height, width = styled.shape
        yy, xx = np.ogrid[:height, :width]
        radius = np.sqrt(((xx - 0.5 * width) / width) ** 2 + ((yy - 0.5 * height) / height) ** 2)
        vignette = np.clip(1.0 - 0.9 * radius, 0.78, 1.0).astype(np.float32)
        styled = np.clip(np.power(np.clip(styled, 0.0, 1.0) * vignette, 1.06), 0.0, 1.0)
        return np.rint(255.0 * styled).astype(np.uint8)

    def _project_catheter(
        self,
        points_world_m: np.ndarray,
        projection: ProjectionGeometry,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = self._patient.world_to_volume_mm(points_world_m)
        local = (points - projection.isocenter_volume_mm) @ projection.local_to_volume
        denominator = local[:, 2] + projection.source_to_isocenter_mm
        visible = denominator > 1e-6
        scale = np.zeros_like(denominator)
        scale[visible] = projection.source_to_detector_mm / denominator[visible]
        u = scale * local[:, 0] / projection.pixel_spacing_mm + 0.5 * self.width
        v = scale * local[:, 1] / projection.pixel_spacing_mm + 0.5 * self.height
        return np.stack((u, v), axis=-1), visible

    def _draw_catheter(self, image: np.ndarray, pixels: np.ndarray, visible: np.ndarray) -> None:
        valid_pixels = [tuple(point) for point, keep in zip(pixels, visible, strict=True) if keep]
        if len(valid_pixels) < 2:
            return
        canvas = Image.fromarray(image)
        ImageDraw.Draw(canvas).line(valid_pixels, fill=(12, 12, 12), width=3, joint="curve")
        image[...] = np.asarray(canvas)
