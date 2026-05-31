# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from common.utils import resolve_path


@dataclass(frozen=True)
class CosmosJob:
    params_file: Path
    output_dir: Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cosmos Transfer Docker jobs for exported Agentic specs.")
    parser.add_argument("--env", required=True, help="Environment id for resolving relative paths under runs/<env>.")
    parser.add_argument("--manifest", required=True, help="Manifest produced by the Cosmos exporter.")
    parser.add_argument("--docker-image", default="cosmos-transfer-2.5")
    parser.add_argument("--cosmos-root", default=os.environ.get("COSMOS_TRANSFER_ROOT"))
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many specs.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of specs to run concurrently.")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--gpu-devices", default=None, help="Comma-separated GPU ids, for example '0,1'.")
    return parser


def iter_params_files(manifest_path: Path) -> list[CosmosJob]:
    manifest = json.loads(manifest_path.read_text())
    jobs: list[CosmosJob] = []
    for demo in manifest["demos"]:
        for camera_entry in demo["cameras"].values():
            for variant in camera_entry["variants"]:
                jobs.append(CosmosJob(params_file=Path(variant["params_file"]), output_dir=Path(variant["output_dir"])))
    return jobs


def command_for_params(
    params_file: Path,
    output_dir: Path,
    docker_image: str,
    cosmos_root: Path | None = None,
    gpu_device: str | None = None,
) -> list[str]:
    repo_mount = _nearest_repo_parent(params_file)
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    hf_home.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--gpus",
        f"device={gpu_device}" if gpu_device is not None else "all",
        "--rm",
        "--ipc=host",
        "-v",
        f"{repo_mount}:{repo_mount}",
        "-v",
        f"{hf_home}:/tmp/huggingface",
        "-e",
        "HOME=/tmp",
        "-e",
        "HF_HOME=/tmp/huggingface",
        "-e",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-w",
        "/workspace",
    ]
    if cosmos_root is not None:
        command.extend(["-v", f"{cosmos_root.resolve()}:/workspace", "-v", "/workspace/.venv"])
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        command.extend(["-e", "HF_TOKEN", "-e", "HUGGING_FACE_HUB_TOKEN"])
    command.extend([docker_image, "bash", "-lc", _container_command(params_file, output_dir)])
    return command


def validate_cosmos_runtime(docker_image: str, cosmos_root: Path | None) -> None:
    if cosmos_root is not None:
        if (
            not (cosmos_root / "bin" / "entrypoint.sh").exists()
            or not (cosmos_root / "examples" / "inference.py").exists()
        ):
            raise SystemExit(f"{cosmos_root} does not look like a cosmos-transfer2.5 checkout")
        return
    check = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            docker_image,
            "-lc",
            "test -f /workspace/bin/entrypoint.sh && test -f /workspace/examples/inference.py",
        ],
        check=False,
    )
    if check.returncode != 0:
        raise SystemExit("Cosmos image is not standalone. Run scripts/build-image.sh or set COSMOS_TRANSFER_ROOT.")


def main() -> None:
    args = build_argparser().parse_args()
    manifest_path = resolve_path(args.manifest, args.env)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    cosmos_root = Path(args.cosmos_root) if args.cosmos_root else None
    if cosmos_root is not None and not cosmos_root.exists():
        raise FileNotFoundError(cosmos_root)
    validate_cosmos_runtime(args.docker_image, cosmos_root)

    jobs = iter_params_files(manifest_path)
    if not args.rerun_existing:
        jobs = [job for job in jobs if not _job_has_output(job)]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")

    gpu_devices = _parse_gpu_devices(args.gpu_devices)
    commands = [
        command_for_params(
            job.params_file, job.output_dir, args.docker_image, cosmos_root, _gpu_for_job(gpu_devices, i)
        )
        for i, job in enumerate(jobs)
    ]
    if args.parallel == 1:
        for command in commands:
            subprocess.run(command, check=True)
        return
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        for future in as_completed([executor.submit(subprocess.run, command, check=True) for command in commands]):
            future.result()


def _container_command(params_file: Path, output_dir: Path) -> str:
    inference = f"python examples/inference.py -i {shlex.quote(str(params_file))} -o {shlex.quote(str(output_dir))}"
    chown = f"chown -R {os.getuid()}:{os.getgid()} {shlex.quote(str(output_dir))}"
    return f"{inference}; status=$?; {chown}; exit $status"


def _job_has_output(job: CosmosJob) -> bool:
    if not job.output_dir.exists():
        return False
    try:
        name = json.loads(job.params_file.read_text()).get("name")
    except (OSError, json.JSONDecodeError):
        name = None
    return bool(name and (job.output_dir / f"{name}.mp4").exists()) or any(
        path.name.endswith(".mp4") and "_control_" not in path.name for path in job.output_dir.glob("*.mp4")
    )


def _parse_gpu_devices(value: str | None) -> list[str]:
    if value is None:
        return []
    devices = [device.strip() for device in value.split(",") if device.strip()]
    if not devices:
        raise SystemExit("--gpu-devices must contain at least one GPU id when provided")
    return devices


def _gpu_for_job(gpu_devices: list[str], index: int) -> str | None:
    return None if not gpu_devices else gpu_devices[index % len(gpu_devices)]


def _nearest_repo_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        current = current.parent
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        if (parent / ".git").exists() or (parent / "workflows" / "agentic").exists():
            return parent
    return current


if __name__ == "__main__":
    main()
