# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera-compatible Isaac Lab sensor for non-RTX fluoroscopy backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import warp as wp
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.configclass import configclass
from xray_simulator.config import DisplaySettings

from i4h_arena.medical.carm import CArmStateProvider
from i4h_arena.medical.catheter import CatheterState, CatheterStateProvider, StaticCatheterStateProvider
from i4h_arena.medical.catheter_attenuation import CatheterMaterial
from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume
from i4h_arena.medical.slang_fluoroscopy import SlangFluoroscopyRenderer
from i4h_arena.medical.synthetic_fluoroscopy import SyntheticFluoroscopyRenderer


@dataclass(slots=True)
class FluoroscopySensorData:
    """Sensor buffers following the image output convention used by Arena."""

    output: dict[str, torch.Tensor] = field(default_factory=dict)
    frame_id: torch.Tensor | None = None


class FluoroscopySensor(SensorBase):
    """Lazy fluoroscopy sensor with a replaceable renderer and catheter provider.

    Phase 0 uses :class:`SyntheticFluoroscopyRenderer`. The patient-specific
    Slang renderer will implement the same ``render(CatheterState)`` boundary.
    """

    cfg: FluoroscopySensorCfg

    def __init__(self, cfg: FluoroscopySensorCfg):
        self._data = FluoroscopySensorData()
        self._renderer: SyntheticFluoroscopyRenderer | SlangFluoroscopyRenderer | None = None
        #: One Slang renderer per distinct patient, in :attr:`patient_groups` order.
        self._patient_renderers: list[SlangFluoroscopyRenderer] = []
        manifests = cfg.patient_twin_manifest
        self._env_manifests: tuple[str, ...] | None = None
        if manifests is not None and not isinstance(manifests, str):
            self._env_manifests = tuple(str(manifest) for manifest in manifests)
            manifests = self._env_manifests[0] if self._env_manifests else None
        distinct = list(dict.fromkeys(self._env_manifests or ()))
        if len(distinct) < 2:
            distinct = []
        self._group_twins = tuple(PatientTwin.load(manifest) for manifest in distinct)
        self._group_env_ids = tuple(
            np.flatnonzero([manifest == name for manifest in self._env_manifests]) for name in distinct
        )
        if self._group_twins:
            self._patient_twin: PatientTwin | None = self._group_twins[0]
        else:
            self._patient_twin = PatientTwin.load(manifests) if manifests is not None else None
        self._catheter_provider: CatheterStateProvider | None = None
        self._carm_provider: CArmStateProvider | None = None
        self._dsa_brightness = 1.0
        super().__init__(cfg)

    @property
    def data(self) -> FluoroscopySensorData:
        self._update_outdated_buffers()
        return self._data

    @property
    def patient_twin(self) -> PatientTwin | None:
        """The patient imaged by environment 0, or by every environment when they share one."""
        return self._patient_twin

    @property
    def patient_groups(self) -> tuple[tuple[PatientTwin, np.ndarray], ...]:
        """Each distinct patient with the environments holding it, or empty when all share one."""
        return tuple(zip(self._group_twins, self._group_env_ids))

    @property
    def is_renderable(self) -> bool:
        """Whether reading :attr:`data` can produce a frame rather than raise.

        The slang backend projects through the C-arm, so reads before provider
        binding cannot render. Navigation observations bind the providers
        themselves; other readers can use this flag to defer rendering.
        """
        return self.cfg.backend != "slang" or self._carm_provider is not None

    @property
    def state_providers_bound(self) -> bool:
        """Whether both live geometry sources are ready for image formation."""
        return self._catheter_provider is not None and self._carm_provider is not None

    def bind_catheter_provider(self, provider: CatheterStateProvider) -> None:
        """Connect the physics-owned state source before the first image read."""
        if not isinstance(provider, CatheterStateProvider):
            raise TypeError("provider must implement CatheterStateProvider")
        self._catheter_provider = provider

    def bind_carm_provider(self, provider: CArmStateProvider) -> None:
        """Connect source/detector poses after the Isaac scene is built."""
        if not isinstance(provider, CArmStateProvider):
            raise TypeError("provider must implement CArmStateProvider")
        self._carm_provider = provider

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        if self.cfg.backend == "synthetic":
            self._renderer = SyntheticFluoroscopyRenderer(
                width=self.cfg.width,
                height=self.cfg.height,
                world_bounds_m=self.cfg.world_bounds_m,
            )
        elif self.cfg.backend == "slang":
            if self._patient_twin is None:
                raise ValueError("the slang fluoroscopy backend requires patient_twin_manifest")
            if self._env_manifests is not None and len(self._env_manifests) != self._num_envs:
                raise ValueError(
                    f"patient_twin_manifest names {len(self._env_manifests)} patients for "
                    f"{self._num_envs} environments; give exactly one per environment"
                )
        else:
            raise ValueError(f"unsupported fluoroscopy backend {self.cfg.backend!r}; expected 'synthetic' or 'slang'")
        self._data.output = {
            "rgb": torch.zeros(
                (self._num_envs, self.cfg.height, self.cfg.width, 3), dtype=torch.uint8, device=self._device
            ),
            "guidance": torch.zeros(
                (self._num_envs, self.cfg.height, self.cfg.width, 3), dtype=torch.uint8, device=self._device
            ),
            "dsa": torch.zeros(
                (self._num_envs, self.cfg.height, self.cfg.width, 3), dtype=torch.uint8, device=self._device
            ),
            "dsa_guidance": torch.zeros(
                (self._num_envs, self.cfg.height, self.cfg.width, 3), dtype=torch.uint8, device=self._device
            ),
            "attenuation": torch.zeros(
                (self._num_envs, self.cfg.height, self.cfg.width, 1), dtype=torch.float32, device=self._device
            ),
        }
        self._data.frame_id = torch.zeros(self._num_envs, dtype=torch.int64, device=self._device)
        if self.cfg.demo_catheter and self._catheter_provider is None:
            points = self._demo_catheter_points()[None, ...]
            self._catheter_provider = StaticCatheterStateProvider(
                CatheterState(
                    positions_world_m=points,
                    valid_nodes=np.array([points.shape[1]], dtype=np.int32),
                )
            )

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        if self._data.frame_id is None:
            raise RuntimeError("fluoroscopy sensor has not been initialized")
        provider = self._catheter_provider
        catheter = provider.snapshot(self._num_envs) if provider is not None else CatheterState.empty(self._num_envs)
        if catheter.num_envs != self._num_envs:
            raise ValueError(f"catheter provider returned {catheter.num_envs} environments; expected {self._num_envs}")
        carm = self._carm_provider.snapshot(self._num_envs) if self._carm_provider is not None else None
        if self.cfg.backend == "slang" and carm is None:
            raise RuntimeError("the slang fluoroscopy backend requires a bound C-arm provider")
        mask = wp.to_torch(env_mask).to(device=self._device, dtype=torch.bool)
        if self._group_twins:
            assert carm is not None
            self._render_each_patient(catheter, carm, mask)
            self._data.frame_id[mask] += 1
            return
        if self._renderer is None:
            assert self._patient_twin is not None
            assert carm is not None
            self._renderer = self._slang_renderer(self._patient_twin, carm)
        rendered = self._renderer.render(catheter, carm)
        self._write(rendered, mask, slice(None))
        self._data.frame_id[mask] += 1

    def _render_each_patient(self, catheter: CatheterState, carm: Any, mask: torch.Tensor) -> None:
        """Image each environment through its own patient's volume.

        One renderer per distinct patient, all on the first one's Slang device,
        and each renders only the environments holding its patient.
        """
        if not self._patient_renderers:
            for twin, env_ids in self.patient_groups:
                shared = self._patient_renderers[0] if self._patient_renderers else None
                self._patient_renderers.append(self._slang_renderer(twin, carm.take(env_ids), shared))
            self._renderer = self._patient_renderers[0]
        for renderer, env_ids in zip(self._patient_renderers, self._group_env_ids):
            rendered = renderer.render(catheter.take(env_ids), carm.take(env_ids))
            rows = torch.as_tensor(env_ids, device=self._device)
            self._write(rendered, mask, rows)

    def _write(self, rendered: dict[str, Any], mask: torch.Tensor, rows: Any) -> None:
        """Store rendered frames into ``rows`` of the outputs, for rows the mask selects."""
        selected = mask[rows]
        for name, values in rendered.items():
            tensor = torch.as_tensor(values, device=self._device)
            target = self._data.output[name]
            if isinstance(rows, slice):
                target[selected] = tensor[selected]
            else:
                target[rows[selected]] = tensor[selected]

    def _slang_renderer(
        self, twin: PatientTwin, carm: Any, share_device_with: SlangFluoroscopyRenderer | None = None
    ) -> SlangFluoroscopyRenderer:
        return SlangFluoroscopyRenderer(
            PatientVolume.load(twin),
            carm,
            width=self.cfg.width,
            height=self.cfg.height,
            step_mm=self.cfg.step_mm,
            device_type=self.cfg.slang_device_type,
            dsa=self.cfg.dsa,
            dsa_boost=self.cfg.dsa_boost,
            dsa_gamma=self.cfg.dsa_gamma,
            visual_style=self.cfg.visual_style,
            display=DisplaySettings(polarity=self.cfg.display_polarity),
            catheter_attenuation=self.cfg.catheter_attenuation,
            # Keep the compositing kernel on the simulation's device instead of Warp's default,
            # which differs on a multi-GPU host.
            catheter_device=str(self._device),
            catheter_material=CatheterMaterial(
                shaft_mu_per_mm=self.cfg.catheter_shaft_mu_per_mm,
                tip_mu_per_mm=self.cfg.catheter_tip_mu_per_mm,
                tip_length_mm=self.cfg.catheter_tip_length_mm,
            ),
            share_device_with=share_device_with,
        )

    def _slang_renderers(self) -> list[SlangFluoroscopyRenderer]:
        if self._patient_renderers:
            return self._patient_renderers
        return [self._renderer] if isinstance(self._renderer, SlangFluoroscopyRenderer) else []

    def adjust_dsa_gamma(self, delta: float) -> float:
        """Adjust the live DSA brightness using the reference gamma range."""
        self.cfg.dsa_gamma = float(np.clip(self.cfg.dsa_gamma + float(delta), 0.3, 3.0))
        results = [renderer.adjust_dsa_gamma(delta) for renderer in self._slang_renderers()]
        return results[0] if results else self.cfg.dsa_gamma

    def set_dsa_brightness(self, brightness: float) -> float:
        """Set intuitive DSA brightness where larger values make the image brighter."""
        self._dsa_brightness = float(np.clip(brightness, 0.3, 3.0))
        gamma = float(np.clip(1.0 / self._dsa_brightness, 0.3, 3.0))
        self.cfg.dsa_gamma = gamma
        for renderer in self._slang_renderers():
            renderer.set_dsa_gamma(gamma)
        return self._dsa_brightness

    def set_display_appearance(self, appearance: str) -> str | None:
        """Select the fluoroscopy or radiograph look, or report None when unsupported.

        The synthetic CI phantom has no display mapping, so it keeps its fixed appearance.
        """
        renderers = self._slang_renderers()
        if not renderers:
            return None
        polarity = [renderer.set_display_appearance(appearance) for renderer in renderers][0]
        self.cfg.display_polarity = polarity
        return polarity

    def set_display_control(self, control: str, value: float) -> float | None:
        """Adjust the display window, or report None when the backend has no mapping."""
        renderers = self._slang_renderers()
        if not renderers:
            return None
        return [renderer.set_display_control(control, value) for renderer in renderers][0]

    def recalibrate_display(self) -> bool:
        """Re-fit the display window to the next rendered frame."""
        renderers = self._slang_renderers()
        for renderer in renderers:
            renderer.recalibrate_display()
        return bool(renderers)

    def select_projection(self, angle_rad: float) -> float:
        """Select a calibrated projection through the bound C-arm provider."""
        select = getattr(self._carm_provider, "select_angle", None)
        if not callable(select):
            raise RuntimeError("the bound C-arm provider does not support named projections")
        return float(select(angle_rad))

    def _demo_catheter_points(self) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.cfg.world_bounds_m
        t = np.linspace(0.0, 1.0, 48, dtype=np.float32)
        x = x_min + (x_max - x_min) * (0.20 + 0.60 * t)
        y = y_max - (y_max - y_min) * (0.12 + 0.72 * t) + 0.012 * np.sin(2.5 * np.pi * t)
        z = np.full_like(t, 0.85)
        return np.stack((x, y, z), axis=-1)


@configclass
class FluoroscopySensorCfg(SensorBaseCfg):
    """Configuration for the custom fluoroscopy sensor."""

    class_type: type[FluoroscopySensor] = FluoroscopySensor
    width: int = 512
    height: int = 512
    backend: str = "synthetic"
    #: One twin for every environment, or a sequence naming one per environment.
    patient_twin_manifest: Any = None
    world_bounds_m: tuple[float, float, float, float] = (-0.35, 0.35, -0.30, 0.30)
    demo_catheter: bool = False
    step_mm: float = 1.0
    slang_device_type: str = "vulkan"
    dsa: bool = False
    dsa_boost: float = 6.0
    dsa_gamma: float = 1.0
    visual_style: str = "default"
    display_polarity: str = "fluoro"
    catheter_attenuation: bool = True
    catheter_shaft_mu_per_mm: float = 0.8
    catheter_tip_mu_per_mm: float = 3.0
    catheter_tip_length_mm: float = 2.0
