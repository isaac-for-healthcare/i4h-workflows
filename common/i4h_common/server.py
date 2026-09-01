# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend half of the two-halves task protocol.

The arena half is one generic proxy (``i4h_engine.remote.RemoteTask``); this
is its counterpart, and for the same reason: the wire protocol should exist in
exactly one place, not be re-derived by every policy stack.

A stack subclasses :class:`PolicyServer` and implements :meth:`load` and
:meth:`infer`. Everything else — session handshake, observation decoding, chunk
publishing, status reporting, error recovery — is here.

Lives in ``common`` rather than a shared ``tasks/`` project because the policy
stacks have mutually incompatible dependencies and cannot import each other;
``common`` is the only thing all of them already depend on.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np

from i4h_common.bus.base import Bus
from i4h_common.bus.keys import Keys
from i4h_common.bus.messages import ActionChunk, ObsFrame, TaskSpecMsg, TaskStatusMsg, decode, encode

logger = logging.getLogger("i4h.server")


@dataclass(frozen=True, slots=True)
class ActionContract:
    """What this backend's action vectors mean.

    Reported on the ready handshake because the backend is the only party
    that has actually loaded the checkpoint. A manifest records intent and
    goes stale the moment someone passes ``--checkpoint``; this does not.
    """

    space: str = "joint_position"
    layout: str = "joints"
    dof: int = 0
    robots: tuple[str, ...] = ()
    gripper: str = "none"


@dataclass
class Session:
    """One inference session, opened by a ``TaskSpecMsg`` from the arena side."""

    task_uid: str
    task_id: str
    run_id: str
    episode_index: int
    prompt: str
    checkpoint: str
    embodiment: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    #: Newest frame acted on, so a backlog is skipped rather than replayed.
    last_step: int = -1

    @property
    def action_horizon(self) -> int:
        return int(self.model.get("action_horizon", 1) or 1)

    @property
    def execution_steps(self) -> int:
        return int(self.model.get("execution_steps", self.action_horizon) or self.action_horizon)

    def images(self, frame: ObsFrame) -> dict[str, np.ndarray]:
        """Decode the frame's raw RGB buffers using the declared shapes."""
        decoded: dict[str, np.ndarray] = {}
        for name, payload in frame.images.items():
            shape = frame.image_shapes.get(name)
            if not shape:
                continue
            decoded[name] = np.frombuffer(payload, dtype=np.uint8).reshape(*shape)
        return decoded


class PolicyServer(ABC):
    """Serves one or more task ids over the bus.

    Subclasses implement :meth:`load` (once per session, on the spec) and
    :meth:`infer` (once per observation, returning an action chunk).
    """

    #: Task ids this server answers. Empty means "anything addressed to us",
    #: which is what a single-stack process wants.
    task_ids: tuple[str, ...] = ()

    #: Directory holding this stack's task manifests, resolved from the
    #: subclass's own module so a backend finds its own declarations.
    manifest_dir: Path | None = None

    def __init__(self, *, namespace: str, bus: Bus | None = None, keys: Keys | None = None) -> None:
        self.keys = keys or Keys(namespace)
        self._own_bus = bus is None
        if bus is None:
            from i4h_common.bus.zenoh_bus import open_zenoh_bus

            bus = open_zenoh_bus()
        self.bus = bus
        self.sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._subscriptions: list[Any] = []

    # -- subclass contract -----------------------------------------------
    @abstractmethod
    def load(self, session: Session) -> None:
        """Prepare the model for ``session``. Called once, before any inference.

        Raise to reject the session; the arena side reports the reason and fails
        the node rather than waiting out its handshake timeout.
        """

    @abstractmethod
    def infer(self, session: Session, frame: ObsFrame) -> np.ndarray | None:
        """Return actions shaped ``(horizon, dof)``, or ``None`` to skip this frame."""

    def action_contract(self, session: Session) -> ActionContract:
        """Describe what :meth:`infer` emits, for the ready handshake.

        The default reads the manifest's ``[task.action]`` block, which is a
        reasonable starting point. Override in a backend that can introspect the
        loaded checkpoint — that is strictly better, because it stays right when
        the manifest is wrong.
        """
        declared = dict(session.params.get("action", {}) or session.model.get("action", {}) or {})
        return ActionContract(
            space=str(declared.get("space", "joint_position")),
            layout=str(declared.get("layout", "joints")),
            dof=int(declared.get("dof", 0)),
            robots=tuple(declared.get("robots", ()) or ()),
            gripper=str(declared.get("gripper", "none")),
        )

    def preload(self, task_id: str, checkpoint: str = "") -> None:
        """Load a checkpoint before any spec arrives.

        Otherwise the first ``load`` happens while the simulator is already
        stepping, and the rollout spends its opening seconds holding position:
        with a 600-step budget that was two-thirds of the episode.
        """
        entry = self._declaration(task_id)
        session = Session(
            task_uid=f"preload-{task_id}",
            task_id=task_id,
            run_id="preload",
            episode_index=0,
            prompt=str(entry.get("prompt") or entry.get("summary", "")),
            checkpoint=checkpoint,
            embodiment=str(entry.get("embodiment", "")),
            model=dict(entry.get("model", {})),
            observation=dict(entry.get("observation", {})),
        )
        logger.info("preloading %s", task_id)
        self.load(session)
        logger.info("preloaded %s", task_id)

    def _declaration(self, task_id: str) -> dict[str, Any]:
        """This task's YAML declaration, from the stack's ``manifest/``."""
        import yaml

        directory = self.manifest_dir
        if directory is None:
            directory = Path(sys.modules[type(self).__module__].__file__).parent / "manifest"
        path = directory / f"{task_id.rpartition('/')[2]}.yaml"
        if not path.is_file():
            logger.warning("no declaration at %s", path)
            return {}
        return yaml.safe_load(path.read_text()) or {}

    def is_done(self, session: Session, frame: ObsFrame) -> bool:
        """Whether the policy considers the task complete.

        Default ``False``: a policy usually cannot judge its own success, so the
        workflow's ``until=`` predicate decides. Override when the model genuinely
        emits a termination signal.
        """
        return False

    def unload(self, session: Session) -> None:
        """Release per-session resources."""
        return None

    # -- protocol --------------------------------------------------------
    def start(self) -> None:
        """Subscribe to the spec channel. Wildcard: one process serves many nodes."""
        pattern = f"{self.keys.root}/task/*/spec"
        self._subscriptions.append(self.bus.subscribe(pattern, self._on_spec))
        logger.info("policy server listening on %s", pattern)

    def serve_forever(self, preload: tuple[str, ...] = (), *, preload_checkpoint: str = "") -> None:
        self.start()
        for task_id in preload:
            self.preload(task_id, checkpoint=preload_checkpoint)
        logger.info("ready for specs")
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError):
                signal.signal(sig, lambda *_: self._stop.set())
        try:
            while not self._stop.is_set():
                time.sleep(0.1)
        finally:
            self.close()

    def preload_only(self, task_ids: tuple[str, ...], *, checkpoint: str = "") -> None:
        """Load requested checkpoints, then exit without opening an inference session."""
        try:
            for task_id in task_ids:
                self.preload(task_id, checkpoint=checkpoint)
        finally:
            self.close()

    def _on_spec(self, key: str, payload: bytes) -> None:
        try:
            message = decode(payload, TaskSpecMsg)
        except Exception:
            logger.warning("undecodable spec on %s", key, exc_info=True)
            return
        if self.task_ids and message.task_id not in self.task_ids:
            logger.debug("ignoring %s (not ours)", message.task_id)
            return

        session = Session(
            task_uid=message.task_uid,
            task_id=message.task_id,
            run_id=message.run_id,
            episode_index=message.episode_index,
            prompt=message.prompt,
            checkpoint=message.checkpoint,
            params=dict(message.params),
            observation=dict(message.observation),
            model=dict(message.model),
        )
        # Fill from this stack's own manifest: the arena side sends none of it,
        # because only this process reads a checkpoint's configuration.
        entry = self._declaration(session.task_id)
        session.embodiment = str(entry.get("embodiment", session.embodiment))
        session.model = {**entry.get("model", {}), **session.model}
        session.observation = {**entry.get("observation", {}), **session.observation}
        if not session.prompt:
            session.prompt = str(entry.get("prompt") or entry.get("summary", ""))
        logger.info("session %s for %s (%r)", session.task_uid, session.task_id, session.prompt)
        try:
            self.load(session)
        except Exception as exc:
            logger.exception("failed to load %s", session.task_id)
            self._publish_status(session.task_uid, "error", f"{type(exc).__name__}: {exc}")
            return

        with self._lock:
            previous = self.sessions.pop(session.task_uid, None)
            if previous is not None:
                self.unload(previous)
            self.sessions[session.task_uid] = session
            self._subscriptions.append(self.bus.subscribe(self.keys.task_obs(session.task_uid), self._on_obs))
        contract = self.action_contract(session)
        logger.info(
            "session %s ready: %s/%s dof=%s robots=%s",
            session.task_uid,
            contract.space,
            contract.layout,
            contract.dof,
            list(contract.robots) or "default",
        )
        self._publish_status(session.task_uid, "ready", contract=contract)

    def _on_obs(self, key: str, payload: bytes) -> None:
        try:
            frame = decode(payload, ObsFrame)
        except Exception:
            logger.warning("undecodable observation on %s", key, exc_info=True)
            return
        with self._lock:
            session = self.sessions.get(frame.task_uid)
        if session is None:
            return

        session.steps += 1
        try:
            actions = self.infer(session, frame)
        except Exception as exc:
            logger.exception("inference failed for %s", session.task_id)
            self._publish_status(session.task_uid, "failure", f"{type(exc).__name__}: {exc}")
            return

        if frame.step < session.last_step:
            logger.debug("session %s: dropping stale frame %s", session.task_uid, frame.step)
            return
        session.last_step = frame.step

        if session.steps <= 1 or session.steps % 100 == 0:
            logger.info(
                "session %s frame %s: images=%s state=%s -> %s actions",
                session.task_uid,
                frame.step,
                sorted(frame.images),
                len(frame.state),
                0 if actions is None else len(actions),
            )
        if actions is not None:
            array = np.asarray(actions, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            horizon, dof = array.shape
            self.bus.publish(
                self.keys.task_action(session.task_uid),
                encode(
                    ActionChunk(
                        task_uid=session.task_uid,
                        for_step=frame.step,
                        horizon=horizon,
                        dof=dof,
                        actions=[float(v) for v in array.reshape(-1)],
                    )
                ),
            )

        if self.is_done(session, frame):
            self._publish_status(session.task_uid, "success")

    def _publish_status(
        self, task_uid: str, status: str, detail: str = "", contract: ActionContract | None = None
    ) -> None:
        message = TaskStatusMsg(task_uid=task_uid, status=status, detail=detail)
        if contract is not None:
            message.action_space = contract.space
            message.action_layout = contract.layout
            message.action_dof = contract.dof
            message.action_robots = list(contract.robots)
            message.action_gripper = contract.gripper
        self.bus.publish(self.keys.task_status(task_uid), encode(message))

    def close(self) -> None:
        with self._lock:
            for session in self.sessions.values():
                self.unload(session)
            self.sessions.clear()
        for subscription in self._subscriptions:
            close = getattr(subscription, "close", None)
            if callable(close):
                close()
        self._subscriptions.clear()
        if self._own_bus:
            self.bus.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
