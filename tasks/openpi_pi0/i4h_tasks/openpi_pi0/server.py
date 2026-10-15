# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""openpi PI0 backend (Franka ultrasound liver scan).

Same protocol as the GR00T stacks — that is the point of putting it in
:class:`i4h_common.server.PolicyServer`. PI0 differs in two ways that matter here:
it wants a flat ``state`` vector rather than named modalities, and it returns a
long horizon (50) that the arena side consumes over many ticks.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from i4h_common.bus.messages import ObsFrame
from i4h_common.server import ActionContract, PolicyServer, Session

logger = logging.getLogger("i4h_tasks.openpi_pi0")

#: Image size PI0 was trained on; ``infer`` pads every camera to it.
IMAGE_SIZE = (224, 224)
#: Franka joints in the state vector; ``infer`` passes the first seven.
STATE_DIM = 7

#: Blackwell compute capabilities the XLA in the pinned jaxlib 0.5.3 can target
#: (``sm_100``, ``sm_101``, ``sm_120``). It compiles an unknown one, GB300's 10.3
#: or GB10's 12.1, as the nearest known target, and its Triton GEMM kernels then
#: fail in ptxas or LLVM and abort the backend (NVBug 6935977). Every earlier
#: architecture is known.
XLA_KNOWN_BLACKWELL = frozenset({(10, 0), (10, 1), (12, 0)})
TRITON_GEMM_FLAG = "--xla_gpu_enable_triton_gemm"


def cuda_compute_capability() -> tuple[int, int] | None:
    """Compute capability of CUDA device 0, asked of the driver without starting JAX."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), 0):
        return None
    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
    if cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device) or cuda.cuDeviceGetAttribute(
        ctypes.byref(minor), 76, device
    ):
        return None
    return major.value, minor.value


def configure_xla(capability: tuple[int, int] | None) -> None:
    """Route GEMMs through cuBLAS on a GPU the pinned XLA cannot build Triton kernels for.

    Must run before JAX creates its GPU backend, which reads ``XLA_FLAGS`` once.
    Only those GPUs pay for it: on an RTX PRO 6000 a warm PI0 inference takes
    0.05 s with Triton GEMMs and 0.36 s without. An ``XLA_FLAGS`` that already
    sets the option is left alone.
    """
    if capability is None or capability[0] < 10 or capability in XLA_KNOWN_BLACKWELL:
        return
    flags = os.environ.get("XLA_FLAGS", "")
    if TRITON_GEMM_FLAG in flags:
        return
    os.environ["XLA_FLAGS"] = f"{flags} {TRITON_GEMM_FLAG}=false".strip()
    logger.info(
        "compute capability %s.%s is newer than the pinned XLA targets; disabling its Triton GEMM fusions",
        *capability,
    )


class OpenPiPi0Server(PolicyServer):
    """Serve ``openpi_pi0/*`` tasks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._policies: dict[str, Any] = {}

    def load(self, session: Session) -> None:
        key = self._cache_key(session)
        if key in self._policies:
            return

        from openpi.policies import policy_config

        model = session.model
        path = session.checkpoint or str(model.get("repo", ""))
        if not path:
            raise ValueError(f"{session.task_id}: no model repo in the manifest and no --checkpoint")
        if not session.checkpoint and not Path(path).expanduser().exists():
            from huggingface_hub import snapshot_download

            cache_dir = os.environ.get(
                "ULTRASOUND_MODEL_CACHE",
                os.path.expanduser("~/.cache/ultrasound_models"),
            )
            path = snapshot_download(repo_id=path, cache_dir=cache_dir)
        from i4h_tasks.openpi_pi0.ultrasound.config import get_config

        config_name = str(model.get("config", "robotic_ultrasound"))
        repo_id = str(model.get("repo_id", "i4h/sim_liver_scan"))
        logger.info("loading PI0 %s (config=%s repo_id=%s)", path, config_name, repo_id)
        policy = policy_config.create_trained_policy(
            get_config(name=config_name, repo_id=repo_id),
            path,
        )
        self._warm_up(policy, session)
        self._policies[key] = policy

    def _warm_up(self, policy: Any, session: Session) -> None:
        """Run one inference on a blank frame so JAX compiles PI0 before the session is ready.

        JAX compiles the sampling loop on its first call. Left to the first real
        frame, that compile ran inside arena's 30 s action timeout, and longer
        than it on an L30, so the episode failed at step 0 with no action
        (NVBug 6935977). Here it runs inside ``load``: during preload, which
        run.sh waits for before starting the simulator, or before the session
        reports ready.
        """
        blank = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
        started = time.monotonic()
        policy.infer(self._observation(session, np.zeros(STATE_DIM), {"room": blank, "wrist": blank}))
        logger.info("compiled PI0 with a warm-up inference in %.1fs", time.monotonic() - started)

    def _cache_key(self, session: Session) -> str:
        return f"{session.checkpoint or session.model.get('repo', '')}|{session.model.get('config', '')}"

    def unload(self, session: Session) -> None:
        pass  # cached across episodes

    def action_contract(self, session: Session) -> ActionContract:
        return ActionContract(space="ee_pose", layout="delta_axis_angle", dof=6)

    def infer(self, session: Session, frame: ObsFrame) -> np.ndarray | None:
        policy = self._policies.get(self._cache_key(session))
        if policy is None:
            return None

        result = policy.infer(self._observation(session, frame.state, session.images(frame)))
        actions = np.atleast_2d(np.asarray(result["actions"], dtype=np.float32))
        # The simulator's relative IK action is exactly dx,dy,dz,axis-angle.
        return actions[: min(5, session.execution_steps, len(actions)), :6]

    @staticmethod
    def _observation(session: Session, state: Any, images: dict[str, np.ndarray]) -> dict[str, Any]:
        from openpi_client import image_tools

        observation: dict[str, Any] = {
            "observation/state": np.asarray(state, dtype=np.float64)[:STATE_DIM],
            "prompt": session.prompt,
        }
        if "room" in images:
            observation["observation/image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(images["room"], *IMAGE_SIZE)
            )
        if "wrist" in images:
            observation["observation/wrist_image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(images["wrist"], *IMAGE_SIZE)
            )
        return observation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openpi-pi0-server")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        help="load this task's checkpoint at startup instead of on the first spec",
    )
    parser.add_argument("--checkpoint", default="", help="checkpoint override used while preloading")
    parser.add_argument("--preload-only", action="store_true", help="load requested checkpoints and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[openpi-pi0] %(message)s",
    )
    configure_xla(cuda_compute_capability())
    server = OpenPiPi0Server(namespace=args.namespace)
    if args.preload_only:
        if not args.preload:
            parser.error("--preload-only requires --preload")
        server.preload_only(tuple(args.preload), checkpoint=args.checkpoint)
        return 0
    server.serve_forever(
        preload=tuple(args.preload),
        preload_checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
