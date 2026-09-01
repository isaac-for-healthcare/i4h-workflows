# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local process bridge between GR00T/RLinf and the Isaac Sim 6 runtime."""

from __future__ import annotations

import os
from contextlib import suppress
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

_AUTHKEY_ENV = "I4H_RL_SIM_AUTHKEY"


def bridge_authkey() -> bytes:
    value = os.environ.get(_AUTHKEY_ENV)
    if not value:
        raise RuntimeError(f"{_AUTHKEY_ENV} is not set")
    return value.encode("utf-8")


def to_numpy_tree(value: Any) -> Any:
    """Move simulator tensors to transport-safe NumPy containers."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(value, np.ndarray) or value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {key: to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_numpy_tree(item) for item in value)
    if isinstance(value, list):
        return [to_numpy_tree(item) for item in value]
    torch_view = getattr(value, "torch", None)
    if torch_view is not None:
        return to_numpy_tree(torch_view)
    raise TypeError(f"cannot transport simulator value of type {type(value).__name__}")


def to_torch_tree(value: Any, *, device: str) -> Any:
    """Restore transported arrays as tensors in the RLinf worker."""
    import torch

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device=device)
    if isinstance(value, dict):
        return {key: to_torch_tree(item, device=device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_torch_tree(item, device=device) for item in value)
    if isinstance(value, list):
        return [to_torch_tree(item, device=device) for item in value]
    return value


class RemoteIsaacEnv:
    """IsaacLab-shaped client for one local Isaac Sim server."""

    def __init__(self, socket_path: str | Path, *, device: str = "cuda:0"):
        self.socket_path = str(Path(socket_path).resolve())
        self._device = device
        self._connection = Client(self.socket_path, family="AF_UNIX", authkey=bridge_authkey())
        self._closed = False

    @classmethod
    def from_environment(cls) -> RemoteIsaacEnv:
        socket_path = os.environ.get("I4H_RL_SIM_SOCKET")
        if not socket_path:
            raise RuntimeError("I4H_RL_SIM_SOCKET is not set for the RLinf environment worker")
        return cls(socket_path)

    def _request(self, command: str, payload: Any = None) -> Any:
        if self._closed:
            raise RuntimeError("Isaac Sim bridge is closed")
        self._connection.send((command, payload))
        status, response = self._connection.recv()
        if status != "ok":
            raise RuntimeError(f"Isaac Sim server failed during {command}: {response}")
        return to_torch_tree(response, device=self._device)

    def reset(self, seed: int | None = None, env_ids: Any = None):
        ids = None if env_ids is None else env_ids.detach().cpu().numpy()
        return self._request("reset", {"seed": seed, "env_ids": ids})

    def step(self, actions):
        return self._request("step", actions.detach().cpu().numpy())

    def close(self) -> None:
        if self._closed:
            return
        try:
            with suppress(BrokenPipeError, ConnectionError, EOFError, OSError):
                self._request("close")
            # The CLI also owns the server process lifecycle, so a server that
            # has already exited is equivalent to a closed bridge.
        finally:
            self._closed = True
            self._connection.close()

    def device(self) -> str:
        return self._device
