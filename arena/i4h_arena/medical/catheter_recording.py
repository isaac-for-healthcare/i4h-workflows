# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What an endoluminal recording holds beyond Isaac Lab's actions and states.

The LeRobot converter builds a catheter dataset from three things a plain Isaac Lab
recording lacks: the procedure state (catheter insertion, rotation and bend, the
C-arm angle, and the carrier arm's joints where there is one) as ``obs/joint_pos``;
the fluoroscopy frame as ``obs/fluoroscopy``, with its display-independent signal
beside it; and the rod's measurements under
``diagnostics/``, from whose tip trajectory it derives the goal columns. These
recorder terms write all three before each step, beside the action taken from them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from i4h_arena.medical.catheter_diagnostics import RECORDED_CONTAINMENT_FIELDS, CatheterEpisodeDiagnostics

#: Shortest side the fluoroscopy frame is recorded at. The converter resizes to 256
#: before training, so recording more is disk spent on pixels it discards.
RECORDED_FRAME_EDGE = 256


def downsample_frame(frame: np.ndarray, edge: int = RECORDED_FRAME_EDGE) -> np.ndarray:
    """Box-average ``frame`` down until its short side is near ``edge``.

    Averaged rather than subsampled because the catheter is about a pixel wide at
    1024: taking every fourth pixel drops the wire out of the frames it happens to
    fall between. Only exact integer factors are used, so every pixel weighs the
    same; a frame that does not divide, or is already small, is returned untouched.
    """
    if edge <= 0 or frame.ndim < 2:
        return frame
    rows, cols = frame.shape[:2]
    factor = min(rows // edge, cols // edge)
    if factor < 2 or rows % factor or cols % factor:
        return frame
    blocks = frame.reshape(rows // factor, factor, cols // factor, factor, *frame.shape[2:])
    reduced = blocks.mean(axis=(1, 3))
    # Rounded before the cast: numpy truncates toward zero, which would darken every frame by half a level.
    return (np.rint(reduced) if np.issubdtype(frame.dtype, np.integer) else reduced).astype(frame.dtype)


class CatheterStateRecorder(RecorderTerm):
    """``obs/joint_pos``: catheter and C-arm joints, then the carrier arm's, in the embodiment manifest's order."""

    def __init__(self, cfg: RecorderTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self._provider: Any = None

    def record_pre_step(self) -> tuple[str, torch.Tensor]:
        if self._provider is None:
            from i4h_arena.embodiments.catheter import CatheterCArmJointStateProvider  # noqa: PLC0415
            from i4h_arena.embodiments.franka_catheter import (  # noqa: PLC0415
                ArmCatheterCArmJointStateProvider,
                ArmDrivenCatheterAction,
            )

            catheter = self._env.scene["catheter"]
            carm = self._env.action_manager.get_term("carm_orbit")
            drive = self._env.action_manager.get_term("catheter")
            self._provider = (
                ArmCatheterCArmJointStateProvider(catheter, carm, drive)
                if isinstance(drive, ArmDrivenCatheterAction)
                else CatheterCArmJointStateProvider(catheter, carm)
            )
        state = np.asarray(self._provider.joint_state().pos, dtype=np.float32).reshape(self._env.num_envs, -1)
        return "obs/joint_pos", torch.from_numpy(state)


class FluoroscopyFrameRecorder(RecorderTerm):
    """``obs/fluoroscopy`` and ``obs/fluoroscopy_attenuation``: the detector image and its signal.

    The image is the uint8 RGB the operator or policy saw, after the display mapping.
    The attenuation, ``1 - exp(-integral of mu)`` as float32, is what the renderer
    produced before that mapping, so it compares across runs however the view was set.
    """

    def record_pre_step(self) -> tuple[str, dict[str, torch.Tensor]]:
        output = self._env.scene["fluoroscopy"].data.output
        frames = output["rgb"].detach().cpu().numpy()
        values = {"fluoroscopy": np.stack([downsample_frame(frame[..., :3].astype(np.uint8)) for frame in frames])}
        if "attenuation" in output:
            signal = output["attenuation"].detach().cpu().numpy().astype(np.float32)
            values["fluoroscopy_attenuation"] = np.stack([downsample_frame(frame) for frame in signal])
        return "obs", {key: torch.from_numpy(value) for key, value in values.items()}


class CatheterDiagnosticsRecorder(RecorderTerm):
    """``diagnostics/<name>``: the rod's tip and root, distance to target and containment.

    Every name is written every step, with NaN where a measurement is not available
    yet, so the datasets stay one row per step.
    """

    def __init__(self, cfg: "CatheterDiagnosticsRecorderCfg", env: Any) -> None:
        super().__init__(cfg, env)
        self._provider = CatheterEpisodeDiagnostics(env.scene["catheter"], target_world_m=cfg.target_world_m)
        widths = {"tip_world_m": 3, "root_world_m": 3}
        if cfg.target_world_m is not None:
            widths["tip_target_distance_m"] = 1
        widths.update(dict.fromkeys(RECORDED_CONTAINMENT_FIELDS, 1))
        self._widths = widths

    def record_pre_step(self) -> tuple[str, dict[str, torch.Tensor]]:
        measured = self._provider.diagnostics()
        n = self._env.num_envs
        values = {}
        for name, width in self._widths.items():
            value = np.full((n, width), np.nan, dtype=np.float32)
            if name in measured:
                value[:] = np.asarray(measured[name], dtype=np.float32).reshape(1, width)
            values[name] = torch.from_numpy(value.squeeze(-1) if width == 1 else value)
        return "diagnostics", values


@configclass
class CatheterDiagnosticsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CatheterDiagnosticsRecorder
    target_world_m: tuple[float, float, float] | None = None
    """The route's distal end, to record the tip's distance to it; ``None`` without a patient twin."""


@configclass
class CatheterRecorderTermsCfg:
    record_catheter_state: RecorderTermCfg = RecorderTermCfg(class_type=CatheterStateRecorder)
    record_fluoroscopy_frames: RecorderTermCfg = RecorderTermCfg(class_type=FluoroscopyFrameRecorder)
    record_catheter_diagnostics: CatheterDiagnosticsRecorderCfg = CatheterDiagnosticsRecorderCfg()


def catheter_recorders(base: Any, target_world_m: tuple[float, float, float] | None) -> Any:
    """``base`` (the embodiment's action and state recorders) with the catheter's own terms.

    The flat policy-observation term is dropped: with a patient twin it would store the
    full-size fluoroscopy image a second time, and the converter reads the terms above.
    """
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg  # noqa: PLC0415
    from isaaclab_arena.utils.configclass import combine_configclass_instances  # noqa: PLC0415

    base.record_pre_step_flat_policy_observations = None
    terms = CatheterRecorderTermsCfg()
    terms.record_catheter_diagnostics.target_world_m = target_world_m
    return combine_configclass_instances("CatheterRecorderCfg", base, terms, bases=(RecorderManagerBaseCfg,))
