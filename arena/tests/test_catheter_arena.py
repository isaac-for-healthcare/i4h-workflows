# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the endoluminal Arena pieces: scripted sweeps, recorders and the live UI handle."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i4h_arena.medical import catheter_recording
from i4h_arena.policies import catheter
from i4h_arena.ui import live


def _env(num_envs: int = 1, step_dt: float = 1 / 30) -> SimpleNamespace:
    env = SimpleNamespace(num_envs=num_envs, device="cpu", step_dt=step_dt)
    env.unwrapped = env
    return env


def test_the_demo_sweep_follows_the_task_graph_schedule_to_the_step() -> None:
    policy = catheter.CatheterSweepPolicy(catheter.CatheterSweepPolicyCfg())
    env = _env()
    commands = torch.cat([policy.get_action(env, None) for _ in range(192)])

    assert torch.equal(commands[29], torch.zeros(4))  # settling for 1 s
    assert commands[30, 0] == pytest.approx(0.012)  # advance
    # 111 steps of 1/30 s is just short of 3.7 s in double precision: still unbending.
    assert commands[111].tolist() == pytest.approx([0.0, 0.0, -0.5, 0.0])
    assert commands[112, 0] == pytest.approx(-0.012)  # retract
    assert commands[150, 3] == pytest.approx(0.45)  # orbit out
    assert commands[190, 3] == pytest.approx(-0.45)  # and back

    policy.reset()
    assert torch.equal(policy.get_action(env, None), torch.zeros(1, 4))


def test_the_carm_sweep_reports_whether_the_image_followed(capsys) -> None:
    env = _env()
    frames = iter([np.zeros((1, 8, 8, 3), np.uint8), np.full((1, 8, 8, 3), 9, np.uint8)] * 2)
    env.scene = {"fluoroscopy": SimpleNamespace(data=SimpleNamespace(output={}))}
    policy = catheter.FluoroscopyCArmSweepPolicy(catheter.FluoroscopyCArmSweepPolicyCfg())
    for _ in range(2):
        env.scene["fluoroscopy"].data.output["rgb"] = torch.from_numpy(next(frames))
        assert policy.get_action(env, None)[0, 3] == pytest.approx(0.45)
    policy.reset()

    assert policy.verdicts == [True]
    assert "C-arm sweep PASS" in capsys.readouterr().out


def test_the_carm_sweep_reports_when_it_is_back_without_a_reset(capsys) -> None:
    env = _env(step_dt=0.5)
    env.scene = {"fluoroscopy": SimpleNamespace(data=SimpleNamespace(output={"rgb": torch.zeros(1, 8, 8, 3)}))}
    policy = catheter.FluoroscopyCArmSweepPolicy(catheter.FluoroscopyCArmSweepPolicyCfg(orbit_s=1.0))
    commands = [float(policy.get_action(env, None)[0, 3]) for _ in range(6)]

    assert commands == pytest.approx([0.45, 0.45, -0.45, -0.45, 0.0, 0.0])
    assert policy.verdicts == [False]  # once, at 2 s, though the episode runs on
    policy.reset()
    assert policy.verdicts == [False]
    assert capsys.readouterr().out.count("C-arm sweep FAIL") == 1


def test_frames_are_box_averaged_to_the_recorded_edge() -> None:
    frame = np.zeros((512, 512, 3), np.uint8)
    frame[0, 0] = 255
    small = catheter_recording.downsample_frame(frame)
    assert small.shape == (256, 256, 3)
    assert small[0, 0, 0] == 64  # one bright pixel in a 2x2 box, rounded
    odd = np.zeros((300, 300, 3), np.uint8)
    assert catheter_recording.downsample_frame(odd) is odd


def test_fluoroscopy_is_recorded_as_the_image_and_its_signal() -> None:
    rgb = torch.full((1, 512, 512, 4), 200, dtype=torch.uint8)
    attenuation = torch.full((1, 512, 512, 1), 0.25)
    env = SimpleNamespace(scene={"fluoroscopy": SimpleNamespace(data=SimpleNamespace(output={}))})
    env.scene["fluoroscopy"].data.output.update(rgb=rgb, attenuation=attenuation)
    term = catheter_recording.FluoroscopyFrameRecorder(catheter_recording.RecorderTermCfg(), env)

    key, values = term.record_pre_step()

    assert key == "obs"
    assert values["fluoroscopy"].shape == (1, 256, 256, 3) and values["fluoroscopy"].dtype == torch.uint8
    assert values["fluoroscopy_attenuation"].shape == (1, 256, 256, 1)
    assert values["fluoroscopy_attenuation"].dtype == torch.float32
    assert float(values["fluoroscopy_attenuation"].max()) == pytest.approx(0.25)


def test_diagnostics_are_recorded_every_step_with_nan_until_measured(monkeypatch) -> None:
    measurements = [{}, {"tip_world_m": np.array([1.0, 2.0, 3.0]), "kinked_nodes": 2.0}]

    class Provider:
        def __init__(self, catheter, *, target_world_m=None) -> None:
            pass

        def diagnostics(self) -> dict:
            return measurements.pop(0)

    monkeypatch.setattr(catheter_recording, "CatheterEpisodeDiagnostics", Provider)
    env = SimpleNamespace(num_envs=1, scene={"catheter": None})
    term = catheter_recording.CatheterDiagnosticsRecorder(
        catheter_recording.CatheterDiagnosticsRecorderCfg(target_world_m=(0.0, 0.0, 0.0)), env
    )

    _key, first = term.record_pre_step()
    _key, second = term.record_pre_step()

    assert set(first) == set(second)
    assert torch.isnan(first["tip_world_m"]).all()
    assert second["tip_world_m"].tolist() == [[1.0, 2.0, 3.0]]
    assert second["kinked_nodes"].tolist() == [2.0]
    assert torch.isnan(second["tip_target_distance_m"]).all()


def test_attach_registers_the_env_without_a_window() -> None:
    class Env:
        sim = SimpleNamespace(has_gui=False)

    env = Env()
    live.attach(env, None, "endoluminal_navigation")
    assert live.current_env() is env
