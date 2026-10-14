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
    if not path.is_file():
        pytest.skip(f"optional patched RLinf source is absent: {path}")
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


def delayed_broadcast(rank, rendezvous, results, ready, complete, fail_startup=False):
    from datetime import timedelta

    dist = torch.distributed
    initialized = False
    try:
        if fail_startup and rank == 0:
            raise RuntimeError("injected startup failure")
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
        initialized = True
        tensor = torch.tensor([42.0 if rank == 0 else 0.0])
        dist.broadcast(tensor, src=0)
        assert tensor.item() == 42
        failure_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
        dist.barrier()
        if rank == 0:
            ready.set()
            if not complete.wait(90):
                raise TimeoutError("rank 1 did not complete the failure test")
            return
        if not ready.wait(60):
            raise TimeoutError("rank 0 did not become ready")
        path = SOURCE / "multi_channel_pg.py"
        node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
        method = next(node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == "_broadcast")
        namespace = dict(torch=torch, dist=dist)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        owner = SimpleNamespace(_cur_rank=rank, _logger=logging.getLogger("collective-test"))
        work = async_work_type()(lambda: namespace["_broadcast"](owner, torch.zeros(1), 0, failure_group))
        work(None)
        try:
            work.wait()
        except Exception as exc:
            results.put(("timeout", work.done(), str(exc)))
        else:
            results.put(("timeout", False, "collective timeout was swallowed"))
    except Exception as exc:
        results.put(("startup", False, str(exc)))
    finally:
        if rank == 1:
            complete.set()
        if initialized:
            dist.destroy_process_group()


def run_gloo(tmp_path, *, fail_startup=False):
    # Check the optional source before spawning children, where a skip is not useful.
    async_work_type()
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    ready, complete = context.Event(), context.Event()
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    processes = [
        context.Process(target=delayed_broadcast, args=(rank, rendezvous, results, ready, complete, fail_startup))
        for rank in (0, 1)
    ]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        result = results.get(timeout=120)
        if not fail_startup:
            for process in started:
                process.join(timeout=20)
                assert process.exitcode == 0
        return result, processes
    finally:
        complete.set()
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        results.close()
        results.join_thread()


def test_gloo_timeout_reaches_waiter(tmp_path):
    kind, done, error = run_gloo(tmp_path)[0]
    assert kind == "timeout" and done and "timed out" in error.lower(), error


def test_gloo_startup_failure_cleans_up_children(tmp_path):
    (kind, done, error), processes = run_gloo(tmp_path, fail_startup=True)
    assert kind == "startup" and not done and "injected startup failure" in error
    assert all(not process.is_alive() for process in processes)
