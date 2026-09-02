# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera-compatible Isaac Lab sensor for non-RTX fluoroscopy backends."""

from __future__ import annotations

from dataclasses import dataclass, field

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
        self._patient_twin = (
            PatientTwin.load(cfg.patient_twin_manifest) if cfg.patient_twin_manifest is not None else None
        )
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
        return self._patient_twin

    def bind_catheter_provider(self, provider: CatheterStateProvider) -> None:
        """Connect the physics-owned state source after the environment is built."""
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
        if self._renderer is None:
            assert self._patient_twin is not None
            assert carm is not None
            self._renderer = SlangFluoroscopyRenderer(
                PatientVolume.load(self._patient_twin),
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
            )
        rendered = self._renderer.render(catheter, carm)
        mask = wp.to_torch(env_mask).to(device=self._device, dtype=torch.bool)
        for name, values in rendered.items():
            tensor = torch.as_tensor(values, device=self._device)
            self._data.output[name][mask] = tensor[mask]
        self._data.frame_id[mask] += 1

    def adjust_dsa_gamma(self, delta: float) -> float:
        """Adjust the live DSA brightness using the reference gamma range."""
        self.cfg.dsa_gamma = float(np.clip(self.cfg.dsa_gamma + float(delta), 0.3, 3.0))
        if isinstance(self._renderer, SlangFluoroscopyRenderer):
            return self._renderer.adjust_dsa_gamma(delta)
        return self.cfg.dsa_gamma

    def set_dsa_brightness(self, brightness: float) -> float:
        """Set intuitive DSA brightness where larger values make the image brighter."""
        self._dsa_brightness = float(np.clip(brightness, 0.3, 3.0))
        gamma = float(np.clip(1.0 / self._dsa_brightness, 0.3, 3.0))
        self.cfg.dsa_gamma = gamma
        if isinstance(self._renderer, SlangFluoroscopyRenderer):
            self._renderer.set_dsa_gamma(gamma)
        return self._dsa_brightness

    def set_display_appearance(self, appearance: str) -> str | None:
        """Select the fluoroscopy or radiograph look, or report None when unsupported.

        The synthetic CI phantom has no display mapping, so it keeps its fixed appearance.
        """
        if not isinstance(self._renderer, SlangFluoroscopyRenderer):
            return None
        polarity = self._renderer.set_display_appearance(appearance)
        self.cfg.display_polarity = polarity
        return polarity

    def set_display_control(self, control: str, value: float) -> float | None:
        """Adjust the display window, or report None when the backend has no mapping."""
        if not isinstance(self._renderer, SlangFluoroscopyRenderer):
            return None
        return self._renderer.set_display_control(control, value)

    def recalibrate_display(self) -> bool:
        """Re-fit the display window to the next rendered frame."""
        if not isinstance(self._renderer, SlangFluoroscopyRenderer):
            return False
        self._renderer.recalibrate_display()
        return True

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
    patient_twin_manifest: str | None = None
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
