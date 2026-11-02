# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters between the control loop and the Unitree H2 + SharpaWave rig.

The loop in ``deploy.py`` talks to one object that satisfies :class:`RobotIO`:
it reads frames and joints from it and sends arm and hand targets to it.
:class:`H2Robot` implements that over the pinned ``xr_teleoperate`` checkout
(``third_party/xr_teleoperate``, patched by ``third_party/setup.sh``): the arm
controller publishes on Unitree DDS, the hands go through the DDS bridge on the
robot's onboard computer, and the cameras arrive as ROS 2 image topics.
:class:`FakeRobot` implements the same surface without hardware, so the whole
client can be exercised against a local policy server.

Everything third-party is imported inside the method that needs it. Importing
this module costs nothing beyond numpy, and ``--fake-robot`` never touches
``teleop``, ``unitree_sdk2py`` or ``rclpy``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from i4h_tasks.gr00t_n17_tactile.client.execution import countdown
from i4h_tasks.gr00t_n17_tactile.client.observation import RobotSnapshot
from i4h_tasks.gr00t_n17_tactile.tactile.interface import HAND_DOF, HANDS

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.client")

#: Overrides where the teleop checkout lives.
XR_TELEOPERATE_ENV = "XR_TELEOPERATE_PATH"
#: Where ``third_party/setup.sh`` puts it, relative to the repository root.
TELEOP_SUBDIR = "third_party/xr_teleoperate"
#: Files that together identify the repository root (the same markers as
#: ``i4h_common.paths``, which the client must not import: it runs outside the
#: task venv).
ROOT_MARKERS = ("DESIGN.md", "engine", "common")
SETUP_HINT = "run ./third_party/setup.sh tasks/gr00t_n17_tactile"

ARM_DOF = 14
RIGHT_ARM = slice(7, 14)
FRAME_NAMES = ("head", "left_wrist", "right_wrist")


# ── locating the teleop checkout ─────────────────────────────────────────────


def repository_root() -> Path:
    """The workflow repository this package is checked out in, found by walking up."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / marker).exists() for marker in ROOT_MARKERS):
            return parent
    raise FileNotFoundError(
        f"cannot locate the workflow repository above {here} (want {ROOT_MARKERS}); "
        f"set {XR_TELEOPERATE_ENV} to a teleop checkout instead"
    )


def teleop_root() -> Path:
    """The ``xr_teleoperate`` checkout that provides the ``teleop`` package.

    ``$XR_TELEOPERATE_PATH`` wins; otherwise ``third_party/xr_teleoperate`` under
    the repository root. The checkout must carry ``teleop/robot_control/robot_arm.py``.
    """
    override = os.environ.get(XR_TELEOPERATE_ENV)
    root = Path(override).expanduser().resolve() if override else repository_root() / TELEOP_SUBDIR
    if not (root / "teleop" / "robot_control" / "robot_arm.py").is_file():
        raise FileNotFoundError(
            f"no xr_teleoperate checkout at {root} (want teleop/robot_control/robot_arm.py); "
            f"{SETUP_HINT} or set {XR_TELEOPERATE_ENV}"
        )
    return root


def ensure_teleop_importable() -> Path:
    """Put the teleop checkout on ``sys.path`` once, so ``import teleop.*`` resolves.

    ``teleop`` has no ``__init__.py`` and imports as a namespace package, which
    is why nothing here reads ``teleop.__file__``.
    """
    root = teleop_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# ── options ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DdsOptions:
    """How the Unitree DDS participant finds the robot (old ``_init_unitree_dds``)."""

    domain: int = 0
    network_interface: str | None = None
    network_address: str | None = None
    network_peer: str | None = None
    body_peers: tuple[str, ...] = ()
    discovery_ssh: str | None = None
    discovery_timeout_s: float = 35.0
    #: False with ``--no-arm``: only the arm needs the H2 body participants.
    discover_body_peers: bool = True


@dataclass(frozen=True)
class ArmOptions:
    motion: bool = False
    head_pitch_home: float = 0.6
    velocity_limit: float = 20.0
    kp_low: float | None = None
    kp_wrist: float | None = None
    kd_low: float | None = None
    kd_wrist: float | None = None
    kp_head: float | None = None
    kd_head: float | None = None


@dataclass(frozen=True)
class CameraOptions:
    head_mode: str = "binocular"
    warmup_s: float = 5.0
    network_address: str | None = None
    network_peer: str | None = None


@dataclass(frozen=True)
class RobotOptions:
    arm: bool = True
    #: Sides to drive; ``()`` for ``--no-hands``.
    hands: tuple[str, ...] = HANDS
    cameras: bool = True
    dds: DdsOptions = field(default_factory=DdsOptions)
    arm_options: ArmOptions = field(default_factory=ArmOptions)
    camera_options: CameraOptions = field(default_factory=CameraOptions)


def init_unitree_dds(options: DdsOptions) -> None:
    """Initialise the Unitree DDS factory the way teleop does.

    A bare ``ChannelFactoryInitialize(domain)`` relies on multicast discovery on
    the default NIC and never finds the H2 body's lowstate publisher. Teleop pins
    exact addresses (workstation robot-net IP + onboard-computer peer) with
    SPDP-only multicast and discovers the body's current metatraffic ports by
    probing SPDP through the onboard computer. The ``networkConfig`` keyword
    exists only with ``third_party/unitree_sdk2_network_config.patch`` applied.
    """
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if options.network_interface:
        ChannelFactoryInitialize(options.domain, networkInterface=options.network_interface)
        logger.info(
            "DDS ChannelFactory initialised on domain %d (interface %s).", options.domain, options.network_interface
        )
        return
    from teleop.utils.unitree_dds_network import (
        H2_BODY_DDS_ADDRESSES,
        THOR_DDS_ADDRESS,
        WORKSTATION_ROBOT_DDS_ADDRESS,
        build_address_config,
    )

    address = options.network_address or WORKSTATION_ROBOT_DDS_ADDRESS
    peers = [options.network_peer or THOR_DDS_ADDRESS]
    body = list(options.body_peers)
    if not body and options.discover_body_peers:
        from teleop.utils.discover_h2_dds_peer import (
            DEFAULT_THOR_SSH_TARGET,
            H2PeerDiscoveryError,
            discover_h2_body_peers_via_ssh,
        )

        ssh_target = options.discovery_ssh or DEFAULT_THOR_SSH_TARGET
        logger.info("Discovering H2 body DDS ports through %s (SPDP probe)...", ssh_target)
        try:
            body = list(
                discover_h2_body_peers_via_ssh(
                    ssh_target=ssh_target,
                    source_ip=H2_BODY_DDS_ADDRESSES[0],
                    timeout_s=options.discovery_timeout_s,
                )
            )
        except H2PeerDiscoveryError as exc:
            raise SystemExit(
                f"H2 DDS peer discovery failed: {exc}; pass explicit --body-dds-peer IP:PORT for both body "
                "participants (basic_service + humanoid)."
            ) from exc
        logger.info("Discovered H2 body peers: %s", ", ".join(body))
    peers += body
    ChannelFactoryInitialize(
        options.domain,
        networkConfig=build_address_config(address, peers, allow_multicast="spdp"),
    )
    logger.info(
        "DDS ChannelFactory initialised on domain %d: address=%s peers=%s multicast=spdp",
        options.domain,
        address,
        peers,
    )


# ── the protocol ─────────────────────────────────────────────────────────────


class RobotIO(Protocol):
    """What ``deploy.py`` needs from a robot. Every read is thread-safe: the async
    workers capture observations while the main thread sends targets."""

    has_arm: bool
    hands: tuple[str, ...]

    def arm_q(self) -> np.ndarray:
        """``(14,)`` float64 radians, left arm then right; zeros when there is no arm."""

    def hand_q(self, side: str) -> np.ndarray:
        """``(22,)`` float32 radians; zeros when the side is absent or the read fails."""

    def frames(self) -> dict[str, np.ndarray | None]:
        """``head`` / ``left_wrist`` / ``right_wrist`` to the latest BGR frame or ``None``."""

    def snapshot(self, hands: Iterable[str]) -> RobotSnapshot:
        """Frames, arm joints and the requested hands' joints, stamped before the first read."""

    def send_arm(self, q: np.ndarray) -> None:
        """Latch a ``(14,)`` target; the controller's own thread rate-limits and publishes."""

    def send_hand(self, side: str, q: np.ndarray, blocking: bool = False) -> None:
        """Send a ``(22,)`` radian target to one hand."""

    def go_home(self) -> None:
        """Arms to the controller's home, hands to neutral."""

    def go_to_init(
        self, left_arm_q: np.ndarray | None, right_arm_q: np.ndarray | None, hand_q: dict[str, Sequence[float]]
    ) -> None:
        """One arm target (both halves) and the hands' init postures."""

    def close(self) -> None:
        """Hands to neutral and released, cameras closed. Idempotent."""


def read_hand_state(hand: Any | None) -> np.ndarray:
    """22 joint angles in radians from a hand client, zeros when it cannot say.

    The DDS client returns ``(0, degrees)``; the vendor SDK returns an error
    object with ``.code`` in the first slot. Both are accepted.
    """
    if hand is None:
        return np.zeros(HAND_DOF, dtype=np.float32)
    try:
        err, degrees = hand.get_joint_position_degree()
        code = getattr(err, "code", err)
        if code != 0:
            return np.zeros(HAND_DOF, dtype=np.float32)
        return np.radians(np.asarray(list(degrees)[:HAND_DOF], dtype=np.float64)).astype(np.float32)
    except Exception:
        return np.zeros(HAND_DOF, dtype=np.float32)


def home_and_hold(robot: RobotIO, hold_s: int) -> None:
    """Home the arms, neutral the hands, then hold for ``hold_s`` seconds."""
    logger.info("Going to home position (hold %ds)...", hold_s)
    robot.go_home()
    countdown(hold_s, "holding home")


def init_and_hold(robot: RobotIO, init_state: dict | None, right_arm_hold: np.ndarray | None, hold_s: int) -> None:
    """Stage the recorded init posture after go-home, then hold.

    Left arm from ``init_state["left_arm"]["qpos"]`` (the average over the
    training episodes), right arm from ``right_arm_hold`` when given (the fixed
    rest pose a single-arm policy never commands) else from the YAML, hands
    from ``<side>_ee``. No-op when there is nothing to stage.
    """
    if not init_state and right_arm_hold is None:
        return
    state = init_state or {}
    left = np.asarray(state.get("left_arm", {}).get("qpos", [0.0] * 7), dtype=np.float64)
    right = (
        np.asarray(right_arm_hold, dtype=np.float64)
        if right_arm_hold is not None
        else np.asarray(state.get("right_arm", {}).get("qpos", [0.0] * 7), dtype=np.float64)
    )
    if left.size != 7 or right.size != 7:
        raise ValueError(f"init arm dims wrong (left={left.size}, right={right.size}); expected 7 each.")
    hands: dict[str, Sequence[float]] = {}
    for side in HANDS:
        q = state.get(f"{side}_ee", {}).get("qpos")
        if q and len(q) == HAND_DOF:
            hands[side] = q
    logger.info("Going to init posture (left=avg, right=fixed; hold %ds)...", hold_s)
    robot.go_to_init(left, right, hands)
    countdown(hold_s, "holding init posture")


# ── the real robot ───────────────────────────────────────────────────────────


class H2Robot:
    """:class:`RobotIO` over the patched teleop checkout.

    ``connect()`` does all the I/O, in the order the old client did it: DDS
    factory, hand clients (which send a neutral command: the hands move), the
    arm controller (whose constructor blocks until a ``rt/lowstate`` message
    arrives), then the cameras.
    """

    def __init__(self, options: RobotOptions) -> None:
        self.options = options
        self._arm: Any = None
        self._hands: dict[str, Any] = {}
        self._cameras: Any = None
        self._closed = False

    @property
    def has_arm(self) -> bool:
        return self._arm is not None

    @property
    def hands(self) -> tuple[str, ...]:
        return tuple(side for side in HANDS if side in self._hands)

    def connect(self) -> H2Robot:
        options = self.options
        ensure_teleop_importable()
        if options.arm or options.hands:
            init_unitree_dds(options.dds)
        if options.hands:
            from teleop.robot_control.robot_hand_sharpa_dds import SharpaHandDDSClient

            for side in options.hands:
                logger.info(
                    "[%s] creating DDS hand client (the bridge on the onboard computer must be running)...", side
                )
                client = SharpaHandDDSClient(side)
                client.set_joint_position([0.0] * HAND_DOF, False)
                self._hands[side] = client
            logger.info("Waiting for hands to reach neutral...")
            time.sleep(2.0)
        if options.arm:
            from teleop.robot_control.robot_arm import H2_ArmController

            arm_options = options.arm_options
            logger.info(
                "Initialising H2_ArmController (motion_mode=%s, head_pitch_home=%s)...",
                arm_options.motion,
                arm_options.head_pitch_home,
            )
            self._arm = H2_ArmController(
                motion_mode=arm_options.motion,
                head_pitch_home=arm_options.head_pitch_home,
                kp_low=arm_options.kp_low,
                kp_wrist=arm_options.kp_wrist,
                kd_low=arm_options.kd_low,
                kd_wrist=arm_options.kd_wrist,
                kp_head=arm_options.kp_head,
                kd_head=arm_options.kd_head,
            )
            self._arm.arm_velocity_limit = arm_options.velocity_limit
            logger.info("Arm velocity limit pinned to %s rad/s.", arm_options.velocity_limit)
            logger.info("H2_ArmController ready.")
        if options.cameras:
            self._open_cameras(options.camera_options)
        return self

    def _open_cameras(self, options: CameraOptions) -> None:
        from teleop.utils.ros_image_client import ROSImageClient

        # Pin CycloneDDS to the camera network like teleop does; without it the
        # client relies on ambient CYCLONEDDS_URI / multicast discovery and
        # typically receives nothing.
        cyclonedds_uri = None
        if options.network_address and options.network_peer:
            from teleop.utils.unitree_dds_network import build_address_config

            cyclonedds_uri = build_address_config(
                options.network_address, options.network_peer, max_message_size_bytes=1400
            )
            logger.info("Camera DDS pinned: address=%s peer=%s", options.network_address, options.network_peer)
        self._cameras = ROSImageClient(
            warmup_timeout=options.warmup_s, head_mode=options.head_mode, cyclonedds_uri=cyclonedds_uri
        )
        logger.info("Camera client: ROS 2 (CycloneDDS), head_mode=%s", options.head_mode)
        self._warm_cameras(options.warmup_s)

    def _warm_cameras(self, timeout_s: float) -> None:
        getters = self._frame_getters()
        deadline = time.perf_counter() + timeout_s
        warmed: set[str] = set()
        while len(warmed) < len(getters) and time.perf_counter() < deadline:
            for name, getter in getters.items():
                if name in warmed:
                    continue
                frame = getter()
                if frame is not None and getattr(frame, "bgr", None) is not None:
                    warmed.add(name)
            if len(warmed) < len(getters):
                time.sleep(0.05)
        logger.info("Cameras warmed: %s (%d/%d)", sorted(warmed), len(warmed), len(getters))

    def _frame_getters(self) -> dict[str, Any]:
        cameras = self._cameras
        return {
            "head": cameras.get_head_frame,
            "left_wrist": cameras.get_left_wrist_frame,
            "right_wrist": cameras.get_right_wrist_frame,
        }

    def arm_q(self) -> np.ndarray:
        if self._arm is None:
            return np.zeros(ARM_DOF, dtype=np.float64)
        return np.asarray(self._arm.get_current_dual_arm_q(), dtype=np.float64)

    def hand_q(self, side: str) -> np.ndarray:
        return read_hand_state(self._hands.get(side))

    def frames(self) -> dict[str, np.ndarray | None]:
        if self._cameras is None:
            return {name: None for name in FRAME_NAMES}
        out: dict[str, np.ndarray | None] = {}
        for name, getter in self._frame_getters().items():
            frame = getter()
            out[name] = None if frame is None else getattr(frame, "bgr", None)
        return out

    def snapshot(self, hands: Iterable[str]) -> RobotSnapshot:
        t_capture = time.monotonic()
        frames = self.frames()
        arm_q = self.arm_q() if self.has_arm else None
        hand_q = {side: self.hand_q(side) for side in hands if side in self.hands}
        return RobotSnapshot(frames=frames, arm_q=arm_q, hand_q=hand_q, t_capture=t_capture)

    def send_arm(self, q: np.ndarray) -> None:
        if self._arm is None:
            return
        self._arm.ctrl_dual_arm(np.asarray(q, dtype=np.float64), np.zeros(ARM_DOF, dtype=np.float64))

    def send_hand(self, side: str, q: np.ndarray, blocking: bool = False) -> None:
        client = self._hands.get(side)
        if client is None:
            return
        err = client.set_joint_position(list(map(float, q)), blocking)
        # The DDS client returns None; the vendor SDK returns an error object.
        if err is not None and getattr(err, "code", 0) != 0:
            logger.warning("%s hand error: %s", side, getattr(err, "message", err))

    def go_home(self) -> None:
        if self._arm is not None:
            self._arm.ctrl_dual_arm_go_home()
        for side in self.hands:
            self._hands[side].set_joint_position([0.0] * HAND_DOF, True)

    def go_to_init(
        self, left_arm_q: np.ndarray | None, right_arm_q: np.ndarray | None, hand_q: dict[str, Sequence[float]]
    ) -> None:
        if self._arm is not None and left_arm_q is not None and right_arm_q is not None:
            target = np.concatenate(
                [np.asarray(left_arm_q, dtype=np.float64), np.asarray(right_arm_q, dtype=np.float64)]
            )
            self._arm.ctrl_dual_arm(target, np.zeros(ARM_DOF, dtype=np.float64))
        for side, q in hand_q.items():
            client = self._hands.get(side)
            if client is not None and len(q) == HAND_DOF:
                client.set_joint_position(list(map(float, q)), True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for side, client in list(self._hands.items()):
            try:
                client.go_neutral()
                time.sleep(0.5)
            except Exception as exc:
                logger.warning("[%s] DDS go_neutral error: %s", side, exc)
            logger.info("[%s] DDS client released.", side)
        self._hands.clear()
        if self._cameras is not None:
            try:
                self._cameras.close()
            except Exception as exc:
                logger.warning("camera client close failed: %s", exc)
            self._cameras = None
        # The arm controller has no shutdown API (daemon threads); the last
        # latched target stays.


# ── the fake ─────────────────────────────────────────────────────────────────


class FakeRobot:
    """:class:`RobotIO` without hardware.

    Frames are constant fills that tell the streams apart (head left eye 30,
    right eye 40, left wrist 10, right wrist 20; BGR uint8) with a frame
    counter in pixel ``[0, 0, 0]``. Joints track every command first-order
    (``state += lag * (target - state)``). Every command is recorded.
    """

    HEAD_LEFT_FILL = 30
    HEAD_RIGHT_FILL = 40
    LEFT_WRIST_FILL = 10
    RIGHT_WRIST_FILL = 20

    def __init__(
        self,
        *,
        arm: bool = True,
        hands: Iterable[str] = HANDS,
        binocular: bool = True,
        frame_hw: tuple[int, int] = (480, 640),
        lag: float = 0.5,
        seed: int = 0,
    ) -> None:
        wanted = set(hands)
        self.has_arm = bool(arm)
        self.hands: tuple[str, ...] = tuple(side for side in HANDS if side in wanted)
        self.binocular = binocular
        self.frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
        self.lag = float(lag)
        self.seed = seed
        self.sent_arm: list[np.ndarray] = []
        self.sent_hands: dict[str, list[np.ndarray]] = {side: [] for side in self.hands}
        self.home_calls = 0
        self.init_calls = 0
        self.closed = False
        self.frame_counter = 0
        self._lock = threading.Lock()
        self._arm_q = np.zeros(ARM_DOF, dtype=np.float64)
        self._hand_q = {side: np.zeros(HAND_DOF, dtype=np.float32) for side in self.hands}

    def _frame(self, fill: int, width: int, counter: int) -> np.ndarray:
        frame = np.full((self.frame_hw[0], width, 3), fill, dtype=np.uint8)
        frame[0, 0, 0] = counter % 256
        return frame

    def frames(self) -> dict[str, np.ndarray | None]:
        with self._lock:
            self.frame_counter += 1
            counter = self.frame_counter
        height, width = self.frame_hw
        if self.binocular:
            head = np.concatenate(
                [
                    np.full((height, width, 3), self.HEAD_LEFT_FILL, dtype=np.uint8),
                    np.full((height, width, 3), self.HEAD_RIGHT_FILL, dtype=np.uint8),
                ],
                axis=1,
            )
            head[0, 0, 0] = counter % 256
        else:
            head = self._frame(self.HEAD_LEFT_FILL, width, counter)
        return {
            "head": head,
            "left_wrist": self._frame(self.LEFT_WRIST_FILL, width, counter),
            "right_wrist": self._frame(self.RIGHT_WRIST_FILL, width, counter),
        }

    def arm_q(self) -> np.ndarray:
        with self._lock:
            return self._arm_q.copy()

    def hand_q(self, side: str) -> np.ndarray:
        with self._lock:
            q = self._hand_q.get(side)
            return np.zeros(HAND_DOF, dtype=np.float32) if q is None else q.copy()

    def snapshot(self, hands: Iterable[str]) -> RobotSnapshot:
        t_capture = time.monotonic()
        frames = self.frames()
        arm_q = self.arm_q() if self.has_arm else None
        hand_q = {side: self.hand_q(side) for side in hands if side in self.hands}
        return RobotSnapshot(frames=frames, arm_q=arm_q, hand_q=hand_q, t_capture=t_capture)

    def send_arm(self, q: np.ndarray) -> None:
        if not self.has_arm:
            return
        target = np.asarray(q, dtype=np.float64).reshape(ARM_DOF)
        with self._lock:
            self._arm_q += self.lag * (target - self._arm_q)
            self.sent_arm.append(target.copy())

    def send_hand(self, side: str, q: np.ndarray, blocking: bool = False) -> None:
        if side not in self._hand_q:
            return
        target = np.asarray(q, dtype=np.float32).reshape(HAND_DOF)
        with self._lock:
            self._hand_q[side] += np.float32(self.lag) * (target - self._hand_q[side])
            self.sent_hands[side].append(target.copy())

    def go_home(self) -> None:
        with self._lock:
            self.home_calls += 1
            self._arm_q[:] = 0.0
            for q in self._hand_q.values():
                q[:] = 0.0

    def go_to_init(
        self, left_arm_q: np.ndarray | None, right_arm_q: np.ndarray | None, hand_q: dict[str, Sequence[float]]
    ) -> None:
        with self._lock:
            self.init_calls += 1
            if self.has_arm and left_arm_q is not None and right_arm_q is not None:
                self._arm_q[:] = np.concatenate(
                    [np.asarray(left_arm_q, dtype=np.float64), np.asarray(right_arm_q, dtype=np.float64)]
                )
            for side, q in hand_q.items():
                if side in self._hand_q and len(q) == HAND_DOF:
                    self._hand_q[side][:] = np.asarray(q, dtype=np.float32)

    def close(self) -> None:
        self.closed = True
