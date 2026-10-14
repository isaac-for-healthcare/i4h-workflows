# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SharpaWave implementations of the contracts in ``interface.py``.

Every third-party piece is public, so none of it is copied here:

* **Deform encoder.** `sharpa-robotics/sharpawave-deform-encoder
  <https://github.com/sharpa-robotics/sharpawave-deform-encoder>`_ (Apache-2.0),
  weights on Hugging Face under the same name. Two strided convolutions
  (1 to 128, 128 to 64) in front of a ConvNeXt-V2 pico with a replaced stem;
  512 features per fingertip. Frozen.
* **Hand kinematics.** `sharpa-robotics/sharpa-urdf-usd-xml
  <https://github.com/sharpa-robotics/sharpa-urdf-usd-xml>`_ (Apache-2.0); the
  ``wave_01`` left and right hand URDFs carry one ``*_fingertip`` frame per finger.
* **Force encoder.** T-Rex's tactile VQ-VAE: code at `ZhuoyangLiu2005/T-Rex
  <https://github.com/ZhuoyangLiu2005/T-Rex>`_ (MIT), weights on Hugging Face at
  ``miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`` (MIT). A 16-row
  window of 6-axis force per fingertip to a 256-d quantized code. Frozen.
* **Live stream.** The SharpaWave SDK (`sharpa-robotics/sharpa-wave-sdk
  <https://github.com/sharpa-robotics/sharpa-wave-sdk>`_, Docker image
  ``sharpadev/sharpawavesdk``) runs on the robot's onboard computer; a small
  bridge there republishes every fingertip frame over ZMQ in the layout of
  ``sensor/wire.py``.

Structure-first draft: the live source is complete; the encoders and the
kinematics are signatures whose docstrings give the loading recipe, and their
bodies land with the GR00T patch series (``patches/README.md``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from i4h_tasks.gr00t_n17_tactile.tactile.interface import DEFORM_SHAPE, FINGERTIPS, FORCE_AXES, HANDS, TactileFrame
from i4h_tasks.gr00t_n17_tactile.tactile.sensor.subscriber import SharpaTactileSubscriber

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.tactile")

#: Prefix of the encoder tensors inside the published autoencoder checkpoint.
DEFORM_ENCODER_PREFIX = "deform_encoder."
#: Where the pinned URDF checkout lives, relative to the repository root.
URDF_CHECKOUT = "third_party/sharpa-urdf-usd-xml"

_NOT_YET = "lands with the GR00T patch series; see tasks/gr00t_n17_tactile/patches/README.md"


def _hand_finger(fingertip: str) -> tuple[str, str]:
    hand, _, finger = fingertip.partition("_")
    return hand, finger


class SharpaLiveSource:
    """:class:`~.interface.TactileSource` over the robot's tactile bridge."""

    def __init__(self, host: str, port: int, force_window: int = 16) -> None:
        self._subscriber = SharpaTactileSubscriber(host, port, force_window_len=force_window)
        self._force_window = force_window

    def latest(self) -> TactileFrame | None:
        snapshot = self._subscriber.snapshot()
        deform, force, timestamp = [], [], float("-inf")
        for fingertip in FINGERTIPS:
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
        )

    def force_window(self, length: int) -> np.ndarray:
        window = self._subscriber.force_window()
        rows = np.stack([window[hand][finger] for hand, finger in map(_hand_finger, FINGERTIPS)], axis=0)
        if rows.shape[1] < length:
            pad = np.repeat(rows[:, :1, :], length - rows.shape[1], axis=1)
            rows = np.concatenate([pad, rows], axis=1)
        return rows[:, -length:, :]

    def close(self) -> None:
        self._subscriber.close()


class SharpaDeformEncoder:
    """:class:`~.interface.TactileEncoder` backed by the frozen SharpaWave encoder.

    Loading recipe, as in the public repository's ``weights.py``: read the
    safetensors file, keep the tensors under :data:`DEFORM_ENCODER_PREFIX`, load
    them into ``DeformEncoder``. Input is the uint8 map cast to float32 with no
    normalization, which is the distribution the encoder was trained on.
    """

    feature_dim = 512

    def __init__(self, weights: str | Path, device: str = "cuda") -> None:
        self.weights = Path(weights)
        self.device = device

    def encode(self, deform: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"SharpaDeformEncoder.encode {_NOT_YET}")


class SharpaWaveKinematics:
    """:class:`~.interface.FingertipKinematics` from the public SharpaWave URDFs.

    Batched forward kinematics along each finger's chain to its ``*_fingertip``
    frame, in torch so it runs inside the training step as well as at
    inference; the joint order of the published URDFs is checked against the
    dataset's hand joint order when the code lands.
    """

    def __init__(self, left_urdf: str | Path | None, right_urdf: str | Path | None) -> None:
        if left_urdf is None and right_urdf is None:
            raise ValueError("at least one hand URDF is required")
        self.left_urdf = Path(left_urdf) if left_urdf else None
        self.right_urdf = Path(right_urdf) if right_urdf else None

    def fingertip_poses(self, hand_joints: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError(f"SharpaWaveKinematics.fingertip_poses {_NOT_YET}")


class TRexForceEncoder:
    """Frozen T-Rex tactile VQ-VAE: a ``(N, window, 6)`` force history to ``(N, 256)`` codes."""

    feature_dim = 256

    def __init__(self, checkpoint: str | Path, window: int = 16, device: str = "cuda") -> None:
        self.checkpoint = Path(checkpoint)
        self.window = window
        self.device = device

    def encode(self, force_window: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"TRexForceEncoder.encode {_NOT_YET}")


def asset_uris(tactile: dict[str, Any]) -> dict[str, str]:
    """Where every frozen asset the manifest names comes from, as URIs.

    ``hf://repo/file`` entries are downloaded by :func:`resolve_assets`; the URDF
    entries are directories of the pinned checkout under ``third_party/``.
    """
    encoder = tactile.get("deform_encoder") or {}
    kinematics = tactile.get("kinematics") or {}
    force = tactile.get("force") or {}
    uris = {
        "deform_encoder": f"hf://{encoder.get('hf_repo', '')}/{encoder.get('file', '')}",
        "left_urdf": f"{URDF_CHECKOUT}/{kinematics.get('left_urdf_dir', '')}",
        "right_urdf": f"{URDF_CHECKOUT}/{kinematics.get('right_urdf_dir', '')}",
    }
    if force.get("enabled"):
        uris["force_encoder"] = f"hf://{force.get('encoder_hf_repo', '')}/model.pt"
    return uris


def resolve_assets(tactile: dict[str, Any]) -> dict[str, Path]:
    """Download the Hugging Face assets and locate the URDF checkout."""
    from huggingface_hub import hf_hub_download

    from i4h_common.paths import workflow_root

    resolved: dict[str, Path] = {}
    for name, uri in asset_uris(tactile).items():
        if uri.startswith("hf://"):
            repo_id, _, filename = uri.removeprefix("hf://").rpartition("/")
            logger.info("downloading %s from %s", filename, repo_id)
            resolved[name] = Path(hf_hub_download(repo_id=repo_id, filename=filename))
            continue
        path = workflow_root() / uri
        if not path.is_dir():
            raise FileNotFoundError(f"{name}: no URDF checkout at {path}; run ./setup.sh first")
        resolved[name] = path
    return resolved
