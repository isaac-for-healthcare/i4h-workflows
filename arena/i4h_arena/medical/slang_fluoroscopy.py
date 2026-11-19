# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patient-backed Slang DRR adapter for the Arena fluoroscopy sensor."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import torch
from PIL import Image, ImageDraw
from xray_simulator.config import DisplaySettings
from xray_simulator.display import calibrate_display

from .carm import CArmState
from .catheter import CatheterState
from .catheter_attenuation import CatheterAttenuation, CatheterMaterial
from .centerline import ordered_centerline_path
from .fluoroscopy_display import CinematicStyle, display_frames, percentile, vessel_overlay
from .fluoroscopy_guidance import draw_catheter_guidance
from .patient_volume import PatientVolume
from .slang_cuda import DEVICE_FAILURE, cuda_device_error

logger = logging.getLogger("i4h_arena.fluoroscopy")

# Window adjustments are expressed against the window fitted from the first frame rather than
# in absolute line-integral units, because the useful range depends on patient size and on the
# μ scaling baked into the twin. A slider in absolute units would need different bounds per
# patient; a multiple of the fitted width behaves the same way on every twin.
_DISPLAY_CONTROL_BOUNDS = {
    "window_level": (-1.0, 1.0),
    "window_width": (0.25, 4.0),
}


# What the pinned xray_simulator's SlangDiffDRRRenderer._init_slang creates before it uploads
# the volume; it takes no device argument, so a second renderer would open a second device.
_SHARED_SLANG_STATE = ("_device", "_module", "_forward_fn", "_forward_batch_fn", "_backward_fn", "_normalize_fn")


def _renderer_sharing_device(renderer_class: type, source: object) -> type:
    """Return a renderer class whose instances reuse ``source``'s Slang device and kernels.

    Only the volume and output textures are new. A second Slang device per process fails on a
    multi-GPU host when CUDA_VISIBLE_DEVICES hides GPU 0, which is where the RLinf simulator
    runs by default, and recompiling the same shader for it is wasted work.
    """

    class SharedDeviceRenderer(renderer_class):
        def _init_slang(self) -> None:
            for name in _SHARED_SLANG_STATE:
                setattr(self, name, getattr(source, name))
            # Batched pose/output buffers belong to each volume renderer.
            self._batch_cache = None
            self._create_resources()

    return SharedDeviceRenderer


def _build_renderer(renderer_class: type, volume: np.ndarray, spacing_zyx_mm, cfg, device_type: str):
    try:
        return renderer_class(volume, spacing_zyx_mm, cfg=cfg)
    except RuntimeError as error:
        if device_type == "cuda" and DEVICE_FAILURE in str(error):
            raise RuntimeError(cuda_device_error(error)) from error
        raise


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
        share_device_with: SlangFluoroscopyRenderer | None = None,
    ) -> None:
        """
        Args:
            share_device_with: Another patient's renderer whose Slang device and
                compiled kernels this one reuses, so several patients render in
                one process on one device.
        """
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
        # Frames stay on the device the catheter is composited on, the simulation's own.
        self._device = torch.device(catheter_device) if catheter_device is not None else torch.device("cpu")
        self._cinematic = (
            CinematicStyle(height=self.height, width=self.width, device=self._device)
            if self._visual_style == "cinematic"
            else None
        )
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
        renderer_class = (
            SlangDiffDRRRenderer
            if share_device_with is None
            else _renderer_sharing_device(SlangDiffDRRRenderer, share_device_with._renderer)
        )
        self._renderer = _build_renderer(
            renderer_class, patient.mu_volume, patient.spacing_zyx_mm, base_config, device_type
        )
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
            self._dsa_renderer = _build_renderer(
                _renderer_sharing_device(SlangDiffDRRRenderer, self._renderer),
                boosted_volume,
                patient.spacing_zyx_mm,
                base_config,
                device_type,
            )

    def render(self, catheter: CatheterState, carm: CArmState | None = None) -> dict[str, torch.Tensor]:
        """Render one detector frame per environment, stacked along the leading axis.

        Every environment is posed in one batched dispatch and mapped to display values as a
        batch, on the renderer's torch device; only the guidance overlays are drawn on the host.
        Every catheter embodiment pins ``env_spacing`` to zero, which puts each environment's
        catheter and C-arm in the patient's own world frame; no origin shift is applied here.
        """
        num_envs = catheter.num_envs
        if carm is not None and carm.num_envs != num_envs:
            raise ValueError(f"C-arm state has {carm.num_envs} environments; catheter has {num_envs}")
        projections = [self._projection_for(carm, env_index) for env_index in range(num_envs)]
        transmission = self._render_volume(self._renderer, projections)
        catheter_transmission = self._catheter_transmission(catheter, projections)
        if catheter_transmission is not None:
            transmission = transmission * catheter_transmission
        if not self._display_calibrated:
            self._calibrate_display(transmission[0].cpu().numpy())
        drr_image = display_frames(transmission, self._display)
        dsa_image = drr_image
        vessel_alpha = None
        if self._dsa_renderer is not None:
            boosted_transmission = self._render_volume(self._dsa_renderer, projections)
            if catheter_transmission is not None:
                boosted_transmission = boosted_transmission * catheter_transmission
            dsa_image = display_frames(boosted_transmission, self._display)
            vessel_signal = (transmission - boosted_transmission).clamp(min=0.0)
            scale = percentile(vessel_signal, 99.5)[:, None, None]
            # A frame with no vessel in view gets no tint rather than amplified noise.
            vessel_alpha = torch.where(
                scale > 1e-8, (vessel_signal / scale.clamp(min=1e-8)).clamp(0.0, 0.82), torch.zeros_like(vessel_signal)
            )

        grayscale = torch.round(255.0 * drr_image).clamp(0.0, 255.0)
        dsa_grayscale = torch.round(255.0 * dsa_image).clamp(0.0, 255.0)
        if self._cinematic is not None:
            grayscale = self._cinematic(grayscale, key="drr")
            dsa_grayscale = self._cinematic(dsa_grayscale, key="dsa")
        if abs(self._dsa_gamma - 1.0) > 1e-3:
            dsa_grayscale = torch.round(255.0 * (dsa_grayscale / 255.0).pow(self._dsa_gamma))
        rgb = grayscale.to(torch.uint8)[..., None].expand(-1, -1, -1, 3).contiguous()
        dsa_rgb = dsa_grayscale.to(torch.uint8)[..., None].expand(-1, -1, -1, 3).contiguous()
        guidance = vessel_overlay(rgb, vessel_alpha) if vessel_alpha is not None else rgb.clone()
        dsa_guidance = dsa_rgb.clone()
        rgb, guidance, dsa_rgb, dsa_guidance = self._draw_catheter_overlays(
            catheter,
            projections,
            annotate=self._catheter_attenuation is None,
            frames=(rgb, guidance, dsa_rgb, dsa_guidance),
        )
        return {
            "rgb": rgb,
            "guidance": guidance,
            "dsa": dsa_rgb,
            "dsa_guidance": dsa_guidance,
            "attenuation": (1.0 - transmission)[..., None],
        }

    def _render_volume(self, renderer, projections: list[ProjectionGeometry]) -> torch.Tensor:
        """Transmission ``(N, H, W)`` for every pose, in one dispatch when the renderer batches."""
        render_batch = getattr(renderer, "render_batch", None)
        if render_batch is not None:
            images = render_batch(
                np.asarray([projection.rotation_zxy_rad for projection in projections], dtype=np.float32),
                np.asarray([projection.translation_xyz_mm for projection in projections], dtype=np.float32),
                as_torch=True,
            )
        else:
            images = torch.from_numpy(
                np.stack(
                    [
                        renderer.render(projection.rotation_zxy_rad, projection.translation_xyz_mm)
                        for projection in projections
                    ]
                )
            )
        return images.to(device=self._device, dtype=torch.float32)

    def _draw_catheter_overlays(
        self,
        catheter: CatheterState,
        projections: list[ProjectionGeometry],
        *,
        annotate: bool,
        frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw the guidance overlays, and the catheter itself when it is not in the beam, on the host.

        These are vector drawings with no batched GPU equivalent, so the frames that carry them make
        one round trip together. ``rgb`` only makes it when the catheter is an annotation.
        """
        drawn = [index for index in range(catheter.num_envs) if int(catheter.valid_nodes[index]) >= 2]
        if not drawn:
            return frames
        rgb, guidance, dsa_rgb, dsa_guidance = frames
        host_rgb = rgb.cpu().numpy() if annotate else None
        host_guidance = guidance.cpu().numpy()
        host_dsa = dsa_rgb.cpu().numpy()
        host_dsa_guidance = dsa_guidance.cpu().numpy()
        for env_index in drawn:
            valid = int(catheter.valid_nodes[env_index])
            pixels, visible = self._project_catheter(
                catheter.positions_world_m[env_index, :valid], projections[env_index]
            )
            if annotate:
                # Without beam compositing the catheter exists only as an annotation.
                self._draw_catheter(host_rgb[env_index], pixels, visible)
                self._draw_catheter(host_guidance[env_index], pixels, visible)
                self._draw_catheter(host_dsa[env_index], pixels, visible)
            host_guidance[env_index] = draw_catheter_guidance(host_guidance[env_index], pixels, visible)
            host_dsa_guidance[env_index] = draw_catheter_guidance(host_dsa[env_index], pixels, visible)

        device = rgb.device
        return (
            torch.from_numpy(host_rgb).to(device) if annotate else rgb,
            torch.from_numpy(host_guidance).to(device),
            torch.from_numpy(host_dsa).to(device) if annotate else dsa_rgb,
            torch.from_numpy(host_dsa_guidance).to(device),
        )

    def _projection_for(self, carm: CArmState | None, env_index: int) -> ProjectionGeometry:
        if carm is None:
            return self._projection
        current = solve_projection_geometry(
            self._patient, carm, width=self.width, height=self.height, env_index=env_index
        )
        if not np.isclose(current.source_to_detector_mm, self._projection.source_to_detector_mm, atol=0.1):
            raise ValueError("changing source-detector distance requires rebuilding the fluoroscopy sensor")
        return current

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
        self, catheter: CatheterState, projections: list[ProjectionGeometry]
    ) -> torch.Tensor | None:
        """Transmission ``(N, H, W)`` through the catheter alone, or ``None`` when none attenuates.

        Returned as its own factor because the plain and contrast-boosted volumes both sit behind
        the same instrument, so one solve serves both projections. An environment whose catheter
        has fewer than two nodes transmits fully.
        """
        if self._catheter_attenuation is None:
            return None
        valid_nodes = [int(valid) for valid in catheter.valid_nodes]
        if max(valid_nodes, default=0) < 2:
            return None
        line_integral = torch.zeros((catheter.num_envs, self.height, self.width), device=self._device)
        for env_index, valid in enumerate(valid_nodes):
            if valid < 2:
                continue
            points_volume_mm = self._patient.world_to_volume_mm(catheter.positions_world_m[env_index, :valid])
            line_integral[env_index] = self._catheter_attenuation.line_integral(
                points_volume_mm,
                1000.0 * float(catheter.radius_m),
                projections[env_index],
                width=self.width,
                height=self.height,
                as_torch=True,
            ).to(self._device)
        return torch.exp(-line_integral)

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
