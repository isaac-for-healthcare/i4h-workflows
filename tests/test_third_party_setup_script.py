# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public component sources, release checkout reuse, and physics solver compatibility checks."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_SETUP = ROOT / "third_party" / "setup.sh"
SOLVER_API = ("register_rod", "set_root_pose_gpu", "proximal_reaction")
# Real Git, isolated from the developer's config so signing or hooks cannot interfere.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "i4h",
    "GIT_AUTHOR_EMAIL": "i4h@example.com",
    "GIT_COMMITTER_NAME": "i4h",
    "GIT_COMMITTER_EMAIL": "i4h@example.com",
}


def _run_setup(
    tmp_path: Path,
    root_origin: str,
    *,
    solver_api: tuple[str, ...] = SOLVER_API,
) -> tuple[subprocess.CompletedProcess[str], str]:
    workspace = tmp_path / "workspace"
    third_party = workspace / "third_party"
    third_party.mkdir(parents=True)
    shutil.copy2(THIRD_PARTY_SETUP, third_party / "setup.sh")

    # Fake Git does not populate working trees. Supply the solver source that
    # setup inspects so these tests model a compatible component checkout.
    physics = third_party / "i4h-physics-simulation-internal" / "xpbd_rod"
    physics.mkdir(parents=True)
    (physics / "solver.py").write_text(
        "\n\n".join(f"def {name}():\n    pass\n" for name in solver_api),
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git_log = tmp_path / "git.log"
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GIT_TEST_LOG"\n'
        "repo=''\n"
        "if [ \"${1:-}\" = '-C' ]; then repo=$2; shift 2; fi\n"
        'if [ "${1:-}" = \'init\' ]; then mkdir -p "$repo/.git"; exit 0; fi\n'
        "if [ \"${1:-} ${2:-} ${3:-}\" = 'remote get-url origin' ]; then\n"
        '  if [ "$repo" = "$GIT_TEST_WORKSPACE" ]; then\n'
        '    if [ -n "$GIT_TEST_ROOT_ORIGIN" ]; then printf \'%s\\n\' "$GIT_TEST_ROOT_ORIGIN"; else exit 2; fi\n'
        "  else\n"
        "    printf '%s\\n' 'https://github.com/previous/origin.git'\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"${1:-}\" = 'rev-parse' ]; then\n"
        # A fresh checkout holds no pinned commits or release tags yet, so setup must fetch them.
        '  case "$*" in "rev-parse --verify --quiet "*) exit 1 ;; esac\n'
        "  printf '%s\\n' '1111111111111111111111111111111111111111'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "GIT_TEST_LOG": str(git_log),
            "GIT_TEST_ROOT_ORIGIN": root_origin,
            "GIT_TEST_WORKSPACE": str(workspace),
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        }
    )
    result = subprocess.run(
        [str(third_party / "setup.sh"), "arena"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, git_log.read_text(encoding="utf-8") if git_log.exists() else ""


@pytest.mark.parametrize(
    "root_origin",
    [
        "git@github.com:isaac-for-healthcare/i4h-workflows-internal.git",
        "https://github.com/isaac-for-healthcare/i4h-workflows-internal.git",
        "",
    ],
)
def test_release_components_use_public_https_sources(tmp_path: Path, root_origin: str) -> None:
    # Every release component is a public repository, so the checkout's own transport no
    # longer matters: SSH, HTTPS and archive builds all fetch the components over HTTPS.
    result, git_log = _run_setup(tmp_path, root_origin)

    assert result.returncode == 0, result.stderr
    assert "carries the coupled-arm solver contract" in result.stdout
    for repository in ("i4h-physics-simulation", "i4h-sensor-simulation", "i4h-digital-twin"):
        assert f"remote set-url origin https://github.com/isaac-for-healthcare/{repository}.git" in git_log
    assert "-internal.git" not in git_log
    assert "git@github.com" not in git_log


@pytest.mark.parametrize("missing_symbol", SOLVER_API)
def test_physics_checkout_missing_required_solver_api_is_rejected(tmp_path: Path, missing_symbol: str) -> None:
    result, _ = _run_setup(
        tmp_path,
        "https://github.com/isaac-for-healthcare/i4h-workflows.git",
        solver_api=tuple(symbol for symbol in SOLVER_API if symbol != missing_symbol),
    )

    assert result.returncode == 1
    assert f"coupled-arm solver contract: {missing_symbol}\n" in result.stderr
    assert "I4H_PHYSICS_SIM_REF" in result.stderr
    assert "carries the coupled-arm solver contract" not in result.stdout


def test_release_component_refs_are_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("I4H_PHYSICS_SIM_REF", "I4H_SENSOR_SIM_REF", "I4H_DIGITAL_TWIN_REF"):
        monkeypatch.delenv(variable, raising=False)
    result, git_log = _run_setup(tmp_path, "")
    assert result.returncode == 0, result.stderr
    for component, ref in (
        ("physics-simulation", "a7be3000454c143c9760c591fbb07a301ee564df"),
        ("sensor-simulation", "5d6ba8547a4c6e25af7eaf551f6a34371ccd6350"),
        ("digital-twin", "6b6470cff156ba15b573ae1b4b44c3a77654b8b3"),
    ):
        assert any(
            f"i4h-{component}-internal fetch" in line and line.endswith(f"origin {ref}")
            for line in git_log.splitlines()
        )


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=GIT_ENV, capture_output=True, text=True, check=True
    ).stdout.strip()


def _checkout_ref(third_party: Path, url: str, ref: str) -> subprocess.CompletedProcess[str]:
    source = THIRD_PARTY_SETUP.read_text(encoding="utf-8")
    functions = ""
    for name in ("ensure_origin_url", "checkout_ref"):
        match = re.search(rf"^{name}\(\) {{\n.*?^}}\n", source, re.MULTILINE | re.DOTALL)
        assert match, f"{name} is missing from {THIRD_PARTY_SETUP}"
        functions += match.group(0)
    script = f'set -euo pipefail\nTHIRD_PARTY_DIR="$1"\nLOG_PREFIX=test\n{functions}checkout_ref physics "$2" "$3"\n'
    return subprocess.run(
        ["bash", "-c", script, "checkout_ref", str(third_party), url, ref],
        env=GIT_ENV,
        capture_output=True,
        text=True,
        check=False,
    )


def test_annotated_release_tag_rerun_keeps_local_edits(tmp_path: Path) -> None:
    # The fake Git reports one object ID for every rev-parse, so it cannot tell an annotated
    # tag from the commit it names. Run the real checkout functions against a real tag.
    remote = tmp_path / "remote"
    remote.mkdir()
    _git("init", "-q", cwd=remote)
    (remote / "solver.py").write_text("release = 1\n", encoding="utf-8")
    _git("add", "solver.py", cwd=remote)
    _git("commit", "-q", "-m", "release", cwd=remote)
    _git("tag", "-a", "-m", "release", "v0.9-rc1", cwd=remote)
    release_commit = _git("rev-parse", "HEAD", cwd=remote)
    third_party = tmp_path / "third_party"
    checkout = third_party / "physics"

    first = _checkout_ref(third_party, remote.as_uri(), "v0.9-rc1")
    assert first.returncode == 0, first.stderr
    assert _git("rev-parse", "HEAD", cwd=checkout) == release_commit

    (checkout / "solver.py").write_text("release = 1  # local solver edit\n", encoding="utf-8")
    rerun = _checkout_ref(third_party, remote.as_uri(), "v0.9-rc1")
    assert rerun.returncode == 0, rerun.stderr
    assert "is now" not in rerun.stdout
    assert (checkout / "solver.py").read_text(encoding="utf-8") == "release = 1  # local solver edit\n"


def _commit(remote: Path, text: str) -> str:
    (remote / "solver.py").write_text(text, encoding="utf-8")
    _git("add", "solver.py", cwd=remote)
    _git("commit", "-q", "-m", text, cwd=remote)
    return _git("rev-parse", "HEAD", cwd=remote)


@pytest.mark.parametrize("kind", ["annotated", "lightweight", "commit"])
def test_release_ref_rerun_does_not_contact_origin(tmp_path: Path, kind: str) -> None:
    # The Docker image fetches the components at build time, and each container's first
    # start runs setup again. With the origin gone, a fetch fails, so a rerun that reuses the
    # build's checkouts must not fetch. The release defaults are full commits; a release
    # tag stays a supported *_REF override.
    remote = tmp_path / "remote"
    remote.mkdir()
    _git("init", "-q", cwd=remote)
    release_commit = _commit(remote, "release = 1\n")
    ref = release_commit if kind == "commit" else "v0.9.0-rc2"
    if kind != "commit":
        _git("tag", *(("-a", "-m", "release") if kind == "annotated" else ()), ref, cwd=remote)
    third_party = tmp_path / "third_party"

    first = _checkout_ref(third_party, remote.as_uri(), ref)
    assert first.returncode == 0, first.stderr
    shutil.rmtree(remote)

    rerun = _checkout_ref(third_party, remote.as_uri(), ref)
    assert rerun.returncode == 0, rerun.stderr
    assert "is now" not in rerun.stdout
    assert _git("rev-parse", "HEAD", cwd=third_party / "physics") == release_commit


def test_branch_rerun_follows_origin(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    _git("init", "-q", "-b", "main", cwd=remote)
    _commit(remote, "release = 1\n")
    third_party = tmp_path / "third_party"
    checkout = third_party / "physics"

    first = _checkout_ref(third_party, remote.as_uri(), "main")
    assert first.returncode == 0, first.stderr
    newer_commit = _commit(remote, "release = 2\n")

    rerun = _checkout_ref(third_party, remote.as_uri(), "main")
    assert rerun.returncode == 0, rerun.stderr
    assert _git("rev-parse", "HEAD", cwd=checkout) == newer_commit
    assert _git("tag", "--list", cwd=checkout) == ""
