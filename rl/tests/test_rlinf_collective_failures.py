# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed RLinf receive must wake waiters, including chained/nested work."""

import ast
import logging
import multiprocessing
import time
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/RLinf-a4b6abe/rlinf/scheduler/collective"


def async_work_type():
    path = SOURCE / "async_work.py"
    classes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name in ("AsyncWork", "AsyncFuncWork")
    ]
    namespace = dict(
        vars(typing),
        Future=torch.futures.Future,
        time=time,
        Worker=SimpleNamespace(current_worker=SimpleNamespace(has_accelerator=False)),
    )
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["AsyncFuncWork"]


def fail():
    raise EOFError("receive failed")


def test_failed_receive_completes_future():
    work = async_work_type()(fail)
    work(None)
    assert work.done(), "the receive queue must not leave a dead thread's future pending"
    with pytest.raises(EOFError, match="receive failed"):
        work.wait()


def test_failed_dependency_does_not_run_success_callback():
    called = []
    head = async_work_type()(fail)
    tail = head.then(lambda: called.append(True))
    head(None)
    assert tail.done() and not called
    with pytest.raises(EOFError):
        tail.wait()


@pytest.mark.parametrize("failure", [True, False])
def test_nested_work_completes_parent(failure):
    kind = async_work_type()
    child = kind(fail if failure else lambda: 42)
    parent = kind(lambda: child)
    parent(None)
    child(None)
    assert parent.done()
    if failure:
        with pytest.raises(EOFError):
            parent.wait()
    else:
        assert parent.wait() == 42


def delayed_broadcast(rank, rendezvous, results):
    dist = torch.distributed
    from datetime import timedelta

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=2))
    try:
        # First epoch/group works. A later receive on the same group expires.
        dist.broadcast(torch.ones(1), src=0)
        dist.barrier()
        if rank == 0:
            time.sleep(4)
            return
        path = SOURCE / "multi_channel_pg.py"
        node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
        method = next(node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == "_broadcast")
        namespace = dict(torch=torch, dist=dist)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        owner = SimpleNamespace(_cur_rank=rank, _logger=logging.getLogger("collective-test"))
        work = async_work_type()(lambda: namespace["_broadcast"](owner, torch.zeros(1), 0, dist.group.WORLD))
        work(None)
        try:
            work.wait()
        except Exception as exc:
            results.put((work.done(), str(exc)))
        else:
            results.put((False, "collective timeout was swallowed"))
    finally:
        dist.destroy_process_group()


def test_gloo_timeout_reaches_waiter(tmp_path):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    processes = [context.Process(target=delayed_broadcast, args=(rank, rendezvous, results)) for rank in (0, 1)]
    for process in processes:
        process.start()
    try:
        done, error = results.get(timeout=20)
        assert done and ("Timed out" in error or "Connection closed" in error), error
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
