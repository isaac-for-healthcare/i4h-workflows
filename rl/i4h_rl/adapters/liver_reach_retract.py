# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Liver evaluation uses explicit pre-reset terminal results from Arena."""


def evaluation_metrics(env):
    return {}


def terminal_result(extras, env_id, objective="full_cycle"):
    matches = [r for r in extras.get("terminal_results", []) if r["env_id"] == env_id]
    if len(matches) != 1:
        raise RuntimeError(f"missing or duplicate pre-reset liver result for world {env_id}")
    result = matches[0]
    if result["objective"] != objective:
        raise ValueError(f"requested {objective}, received curriculum result {result['objective']}")
    return dict(result)


def export_contract(env, actor, checkpoint, policy):
    import hashlib
    import json
    import subprocess
    from importlib.metadata import distribution, version
    from pathlib import Path
    from urllib.parse import unquote, urlparse

    import i4h_physics

    from i4h_arena.medical.liver import ENCODER, REWARD_REVISION, TASK_CONTRACT

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    origin = json.loads(distribution("i4h-physics").read_text("direct_url.json") or "{}")
    url = urlparse(origin.get("url", ""))
    numerical = Path(unquote(url.path)).resolve()
    if (
        url.scheme != "file"
        or not origin.get("dir_info", {}).get("editable")
        or not Path(i4h_physics.__file__).resolve().is_relative_to(numerical)
    ):
        raise RuntimeError("Liver export requires an identifiable editable Numerical Sim checkout; run ./setup.sh")
    revision = subprocess.check_output(["git", "-C", str(numerical), "rev-parse", "HEAD"], text=True).strip()
    contract = dict(
        objective=env.liver.objective.objective,
        encoder=ENCODER,
        task=TASK_CONTRACT,
        numerical_revision=revision,
        action_layout=["delta_x", "delta_y", "delta_z", "grip"],
        xyz_vector_limit_m=0.001,
        grip_mapping="(clip(action,-1,1)+1)/2",
        physics_hz=60,
        control_hz=30,
        decimation=2,
        quaternion="fixed demonstrated XYZW",
        energy=0.0,
        fire=0.0,
        reward_revision=REWARD_REVISION,
        checkpoint_sha256=digest(checkpoint),
        policy_sha256=digest(policy),
        std_type=actor.distribution.std_type,
        rsl_rl_version=version("rsl-rl-lib"),
    )
    policy.with_name("contract.json").write_text(json.dumps(contract, indent=2) + "\n")
