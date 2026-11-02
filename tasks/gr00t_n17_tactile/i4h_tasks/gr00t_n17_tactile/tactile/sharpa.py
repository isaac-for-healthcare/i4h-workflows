# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SharpaWave implementations of the contracts in ``interface.py``.

Every third-party piece is public and pinned by ``third_party/setup.sh``; none
of it is copied here:

* **Deform encoder.** `sharpa-robotics/sharpawave-deform-encoder
  <https://github.com/sharpa-robotics/sharpawave-deform-encoder>`_ (Apache-2.0),
  weights on Hugging Face under the same name. Two strided convolutions
  (1 to 128, 128 to 64) in front of a ConvNeXt-V2 pico with a replaced stem;
  512 features per fingertip. Frozen.
* **Hand kinematics.** `sharpa-robotics/sharpa-urdf-usd-xml
  <https://github.com/sharpa-robotics/sharpa-urdf-usd-xml>`_ (Apache-2.0); the
  ``wave_01`` left and right hand URDFs carry one ``*_fingertip`` frame per finger.
* **Force encoder.** T-Rex's tactile VQ-VAE: code at `ZhuoyangLiu2005/T-Rex
  <https://github.com/ZhuoyangLiu2005/T-Rex>`_ (MIT), weights inside the
  published VLA checkpoint ``miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6``
  (``model.pt``; a file holding only its ``tactile_vqvae.*`` tensors loads the
  same way). A 16-row window of 6-axis force per fingertip to a 256-d quantized
  code. Frozen.
* **Live stream.** The SharpaWave SDK (`sharpa-robotics/sharpa-wave-sdk
  <https://github.com/sharpa-robotics/sharpa-wave-sdk>`_, Docker image
  ``sharpadev/sharpawavesdk``) runs on the robot's onboard computer; a small
  bridge there republishes every fingertip frame over ZMQ in the layout of
  ``sensor/wire.py``.

The encoder and kinematics classes are thin wrappers over the modules of the
patched Isaac-GR00T checkout (``patches/README.md``), which is where the model
runs them; they exist so that tooling outside the model can compute the same
features. Neither encoder repository is installed as a package: the Sharpa
repository ships no ``pyproject`` and T-Rex's pins a full VLA stack (Python
3.10, torch 2.6), so the patched GR00T imports both by module name and
:func:`import_public_sources` puts their checkouts on ``sys.path``.

Encoder weights. The checkpoint Sharpa publishes was trained from random
initialization and is a different training run from the internal SharpaWave
weights the checkpoints finetuned so far were trained with; the two produce
unrelated features for the same map. A finetuned GR00T checkpoint carries its
own copy of the frozen encoder, so serving one does not depend on which file
the manifest names (the file only builds the module before the checkpoint's
tensors are loaded over it). Training a new checkpoint uses the manifest's
file throughout, which is self-consistent.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from i4h_tasks.gr00t_n17_tactile.tactile.interface import (
    DEFORM_SHAPE,
    FINGERTIPS,
    FORCE_AXES,
    HAND_DOF,
    HANDS,
    POSE_DIM,
    TactileFrame,
)
from i4h_tasks.gr00t_n17_tactile.tactile.sensor.subscriber import SharpaTactileSubscriber

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.tactile")

#: Prefix of the encoder tensors inside the published autoencoder checkpoint.
DEFORM_ENCODER_PREFIX = "deform_encoder."
#: Where the pinned URDF checkout lives, relative to the repository root.
URDF_CHECKOUT = "third_party/sharpa-urdf-usd-xml"
#: Checkouts the patched GR00T imports by module name, relative to the repository root.
PUBLIC_SOURCES = ("third_party/sharpawave-deform-encoder", "third_party/T-Rex")
#: What to run when a checkout is missing.
SETUP_HINT = "run ./third_party/setup.sh tasks/gr00t_n17_tactile"


def _hand_finger(fingertip: str) -> tuple[str, str]:
    hand, _, finger = fingertip.partition("_")
    return hand, finger


def public_source_paths() -> list[Path]:
    """Absolute paths of the two encoder checkouts, whether or not they exist yet."""
    from i4h_common.paths import workflow_root

    root = workflow_root()
    return [root / relative for relative in PUBLIC_SOURCES]


def require_public_sources() -> list[Path]:
    """The encoder checkouts, or a FileNotFoundError that says how to get them."""
    paths = public_source_paths()
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"no checkout at {', '.join(str(path) for path in missing)}; {SETUP_HINT}")
    return paths


def import_public_sources() -> None:
    """Make ``sharpawave_deform_encoder`` and ``tactile_vqvae`` importable in this process."""
    for path in require_public_sources():
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def urdf_file(directory: str | Path, hand: str) -> Path:
    """The base-variant URDF of one hand inside its checkout directory.

    ``wave_01/<hand>_sharpa_wave/<hand>_sharpa_wave.urdf``; the ``_with_wrist``
    and ``_with_flange`` variants next to it carry extra links the fingertip
    kinematics do not need.
    """
    if hand not in HANDS:
        raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
    return Path(directory) / f"{hand}_sharpa_wave.urdf"


class SharpaLiveSource:
    """:class:`~.interface.TactileSource` over the robot's tactile bridge.

    ``fingertips`` are the ones the checkpoint consumes, in its order; a
    single-hand checkpoint names five, and the other hand's channels are then
    neither waited for nor returned.
    """

    def __init__(self, host: str, port: int, force_window: int = 16, fingertips: Iterable[str] = FINGERTIPS) -> None:
        self.fingertips = tuple(fingertips)
        unknown = [name for name in self.fingertips if name not in FINGERTIPS]
        if unknown:
            raise ValueError(f"unknown fingertips {unknown}; the names are {FINGERTIPS}")
        self._endpoint = f"tcp://{host}:{port}"
        self._subscriber = SharpaTactileSubscriber(host, port, force_window_len=force_window)
        self._force_window = force_window

    def missing(self) -> list[str]:
        """Fingertips of this source that have not reported yet."""
        snapshot = self._subscriber.snapshot()
        return [name for name in self.fingertips if snapshot[_hand_finger(name)[0]].get(_hand_finger(name)[1]) is None]

    def wait_ready(self, timeout: float = 2.0) -> None:
        """Block until every fingertip has sent one frame, or raise naming the silent ones.

        A dead bridge, a wrong host or port, or one dead sensor would otherwise
        serve the whole session on the request's tactile data without a word.
        A genuine no-contact frame still arrives and counts, so this only trips
        on fingertips that send nothing at all.
        """
        deadline = time.monotonic() + timeout
        while True:
            missing = self.missing()
            if not missing:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{len(missing)}/{len(self.fingertips)} fingertips sent nothing on {self._endpoint} within "
                    f"{timeout:.1f} s: {missing}; check that the bridge publishes every sensor and that "
                    "--tactile-host / --tactile-port are right"
                )
            time.sleep(0.05)

    def latest(self) -> TactileFrame | None:
        snapshot = self._subscriber.snapshot()
        deform, force, timestamp = [], [], float("-inf")
        for fingertip in self.fingertips:
            hand, finger = _hand_finger(fingertip)
            msg = snapshot[hand].get(finger)
            if msg is None or msg.deform.shape != DEFORM_SHAPE or msg.f6.size < FORCE_AXES:
                return None
            deform.append(msg.deform)
            force.append(msg.f6[:FORCE_AXES])
            timestamp = max(timestamp, msg.ts)
        hand_joints = {}
        for hand in HANDS:
            degrees = snapshot["joints"][hand]
            if degrees is not None:
                hand_joints[hand] = np.deg2rad(np.asarray(degrees, dtype=np.float32))
        return TactileFrame(
            timestamp=timestamp,
            deform=np.stack(deform, axis=0),
            force=np.stack(force, axis=0).astype(np.float32),
            hand_joints=hand_joints or None,
            fingertips=self.fingertips,
        )

    def force_window(self, length: int) -> np.ndarray:
        window = self._subscriber.force_window()
        rows = np.stack([window[hand][finger] for hand, finger in map(_hand_finger, self.fingertips)], axis=0)
        if rows.shape[1] < length:
            pad = np.repeat(rows[:, :1, :], length - rows.shape[1], axis=1)
            rows = np.concatenate([pad, rows], axis=1)
        return rows[:, -length:, :]

    def close(self) -> None:
        self._subscriber.close()


class SharpaDeformEncoder:
    """:class:`~.interface.TactileEncoder` backed by the frozen SharpaWave encoder.

    Wraps ``SharpaTactileEncoder`` of the patched GR00T, which builds the public
    ``DeformEncoder`` and loads the tensors under :data:`DEFORM_ENCODER_PREFIX`
    from a SafeTensors or ``.pth`` file. Input is the uint8 map cast to float32
    with no normalization, which is the distribution the encoder was trained on.
    """

    feature_dim = 512

    def __init__(self, weights: str | Path, device: str = "cuda") -> None:
        self.weights = Path(weights)
        self.device = device
        self._module: Any = None

    def _load(self) -> Any:
        if self._module is None:
            import_public_sources()
            from gr00t.model.modules.tactile.sharpa_encoder import SharpaTactileEncoder

            self._module = SharpaTactileEncoder(ckpt_path=self.weights).to(self.device)
        return self._module

    def encode(self, deform: np.ndarray) -> np.ndarray:
        import torch

        maps = np.asarray(deform)
        if maps.ndim != 3 or tuple(maps.shape[1:]) != DEFORM_SHAPE:
            raise ValueError(f"deform must be (N, {DEFORM_SHAPE[0]}, {DEFORM_SHAPE[1]}), got {maps.shape}")
        module = self._load()
        batch = torch.from_numpy(np.ascontiguousarray(maps, dtype=np.float32))[None, :, None].to(self.device)
        with torch.inference_mode():
            return module(batch)[0].float().cpu().numpy()


class SharpaWaveKinematics:
    """:class:`~.interface.FingertipKinematics` from the public SharpaWave URDFs.

    Wraps ``SharpaWaveFK`` of the patched GR00T: batched forward kinematics
    along each finger's chain to its ``*_fingertip`` frame, in torch so it runs
    inside the training step as well as at inference. The published URDFs are
    byte-identical to the ones the datasets were recorded against, so the joint
    order needs no remapping.
    """

    def __init__(self, left_urdf: str | Path | None, right_urdf: str | Path | None) -> None:
        if left_urdf is None and right_urdf is None:
            raise ValueError("at least one hand URDF is required")
        self.left_urdf = Path(left_urdf) if left_urdf else None
        self.right_urdf = Path(right_urdf) if right_urdf else None
        self._module: Any = None

    def _load(self) -> Any:
        if self._module is None:
            from gr00t.model.modules.tactile.fk import SharpaWaveFK

            self._module = SharpaWaveFK(left_urdf_path=self.left_urdf, right_urdf_path=self.right_urdf)
        return self._module

    def fingertip_poses(self, hand_joints: dict[str, np.ndarray]) -> np.ndarray:
        import torch

        angles = {}
        for hand in HANDS:
            joints = hand_joints.get(hand)
            if joints is None:
                continue
            vector = np.asarray(joints, dtype=np.float32).reshape(-1)
            if vector.shape[0] != HAND_DOF:
                raise ValueError(f"{hand} hand joints must have {HAND_DOF} values, got {vector.shape[0]}")
            angles[f"joint_angles_{hand}"] = torch.from_numpy(vector)[None]
        if not angles:
            raise ValueError(f"hand_joints must name at least one of {HANDS}")
        with torch.inference_mode():
            poses = self._load()(**angles)[0].cpu().numpy()
        assert poses.shape[-1] == POSE_DIM
        return poses


class TRexForceEncoder:
    """Frozen T-Rex tactile VQ-VAE: a ``(N, window, 6)`` force history to ``(N, 256)`` codes.

    Wraps ``TRexForceEncoder`` of the patched GR00T, which normalizes the raw
    window with T-Rex's per-axis statistics and returns the quantized codebook
    vectors. ``hand`` picks the statistics slice of the first five fingertips
    (``left`` is what the checkpoints were trained with; a right-hand-only
    caller passes ``right``). ``N`` must be a multiple of five, one T-Rex hand;
    the module is built for the ``N`` of the first call.
    """

    feature_dim = 256

    def __init__(self, checkpoint: str | Path, window: int = 16, device: str = "cuda", hand: str = "left") -> None:
        if hand not in HANDS:
            raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
        self.checkpoint = Path(checkpoint)
        self.window = window
        self.device = device
        self.hand = hand
        self._module: Any = None

    def _load(self, num_fingers: int) -> Any:
        if self._module is None:
            import_public_sources()
            from gr00t.model.modules.tactile.trex_force_encoder import TRexForceEncoder as _Encoder

            self._module = _Encoder(
                num_fingers=num_fingers,
                ckpt_path=self.checkpoint,
                vqvae_config={"window": self.window},
                stats_hand_offset=HANDS.index(self.hand),
            ).to(self.device)
        elif self._module.num_fingers != num_fingers:
            raise ValueError(f"encoder was built for {self._module.num_fingers} fingertips, got {num_fingers}")
        return self._module

    def encode(self, force_window: np.ndarray) -> np.ndarray:
        import torch

        window = np.asarray(force_window, dtype=np.float32)
        if window.ndim != 3 or window.shape[1] != self.window or window.shape[2] != FORCE_AXES:
            raise ValueError(f"force_window must be (N, {self.window}, {FORCE_AXES}), got {window.shape}")
        module = self._load(window.shape[0])
        batch = torch.from_numpy(np.ascontiguousarray(window))[None].to(self.device)
        with torch.inference_mode():
            return module(batch)[0].float().cpu().numpy()


def hf_uri(repo_id: str, filename: str, revision: str | None = None) -> str:
    """``hf://repo[@revision]/file``: one Hugging Face file, pinned when a revision is given."""
    return f"hf://{repo_id}{'@' + revision if revision else ''}/{filename}"


def parse_hf_uri(uri: str) -> tuple[str, str | None, str]:
    """``(repo_id, revision or None, filename)`` of an :func:`hf_uri`."""
    if not uri.startswith("hf://"):
        raise ValueError(f"not a Hugging Face URI: {uri!r}")
    repo_and_revision, _, filename = uri.removeprefix("hf://").rpartition("/")
    repo_id, _, revision = repo_and_revision.partition("@")
    if not repo_id or not filename:
        raise ValueError(f"malformed Hugging Face URI: {uri!r}")
    return repo_id, revision or None, filename


def asset_uris(tactile: dict[str, Any]) -> dict[str, str]:
    """Where every frozen asset the manifest names comes from, as URIs.

    ``hf://repo@revision/file`` entries are downloaded by :func:`resolve_assets`
    at that revision; the URDF entries are files of the pinned checkout under
    ``third_party/``.
    """
    encoder = tactile.get("deform_encoder") or {}
    kinematics = tactile.get("kinematics") or {}
    force = tactile.get("force") or {}
    uris = {
        "deform_encoder": hf_uri(encoder.get("hf_repo", ""), encoder.get("file", ""), encoder.get("revision")),
        "left_urdf": str(urdf_file(f"{URDF_CHECKOUT}/{kinematics.get('left_urdf_dir', '')}", "left")),
        "right_urdf": str(urdf_file(f"{URDF_CHECKOUT}/{kinematics.get('right_urdf_dir', '')}", "right")),
    }
    if force.get("enabled"):
        uris["force_encoder"] = hf_uri(force.get("encoder_hf_repo", ""), "model.pt", force.get("encoder_revision"))
    return uris


def resolve_assets(tactile: dict[str, Any], names: Iterable[str] | None = None) -> dict[str, Path]:
    """Download the Hugging Face assets and locate the URDF files.

    ``names`` restricts the work to some assets: serving a checkpoint needs the
    encoder file and the URDFs but not the force encoder, whose tensors the
    checkpoint carries.
    """
    from huggingface_hub import hf_hub_download

    from i4h_common.paths import workflow_root

    wanted = None if names is None else set(names)
    resolved: dict[str, Path] = {}
    for name, uri in asset_uris(tactile).items():
        if wanted is not None and name not in wanted:
            continue
        if uri.startswith("hf://"):
            repo_id, revision, filename = parse_hf_uri(uri)
            logger.info("downloading %s from %s @ %s", filename, repo_id, revision or "main")
            resolved[name] = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
            continue
        path = workflow_root() / uri
        if not path.is_file():
            raise FileNotFoundError(f"{name}: no URDF at {path}; {SETUP_HINT}")
        resolved[name] = path
    return resolved
