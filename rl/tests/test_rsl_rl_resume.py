# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from i4h_rl import cli
from i4h_rl.backends import rsl_rl
from i4h_rl.profile import load_profile

ROOT = Path(__file__).resolve().parents[2]


def test_native_checkpoint_training_resumes_optimizer_and_records_source(tmp_path, monkeypatch):
    checkpoint = tmp_path / "previous.pt"
    checkpoint.write_bytes(b"previous")
    run_dir = tmp_path / "next-run"
    args = cli._parser().parse_args(["liver_reach_retract", "--checkpoint", str(checkpoint), "--run-dir", str(run_dir)])
    profile = load_profile("liver_reach_retract")
    rsl_rl.validate_launch(args, profile, ROOT)
    monkeypatch.setattr(rsl_rl, "_runtime_python", lambda *args: Path("/python"))
    monkeypatch.setattr(rsl_rl, "_preflight", lambda *args: None)
    commands = []

    def call(command, *, env, cwd):
        commands.append(command)
        assert command[command.index("--checkpoint") + 1] == str(checkpoint)
        assert "--resume_optimizer_state" in command
        folder = run_dir / "trainer"
        (folder / "params").mkdir(parents=True)
        (folder / "model_1.pt").write_bytes(b"resumed")
        (folder / "params/agent.yaml").write_text("actor: {}\n")
        return 0

    monkeypatch.setattr(rsl_rl.subprocess, "call", call)
    assert rsl_rl.launch(args, profile, ROOT, num_envs=16, epochs=1, overrides=()) == 0
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["mode"] == "rl-resume" and metadata["checkpoint"] == str(checkpoint)
    assert len(commands) == 1
