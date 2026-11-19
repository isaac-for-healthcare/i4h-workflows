# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed RLinf receive must wake waiters, including chained/nested work."""

import ast
import asyncio
import logging
import multiprocessing
import threading
import time
import typing
from concurrent.futures import Future as ConcurrentFuture
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/RLinf-a4b6abe/rlinf/scheduler/collective"
CLASSES = ("AsyncWork", "AsyncFuncWork", "AsyncChannelWork", "AsyncChannelCommWork")


def async_work_type(name="AsyncFuncWork"):
    path = SOURCE / "async_work.py"
    if not path.is_file():
        pytest.skip(f"optional patched RLinf source is absent: {path}")
    classes = [
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name in CLASSES
    ]
    namespace = dict(
        vars(typing),
        asyncio=asyncio,
        threading=threading,
        Future=torch.futures.Future,
        ConcurrentFuture=ConcurrentFuture,
        time=time,
        ray=SimpleNamespace(actor=SimpleNamespace(ActorHandle=object)),
        Worker=SimpleNamespace(current_worker=SimpleNamespace(has_accelerator=False)),
    )
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


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


@pytest.mark.parametrize("receive_first", [True, False])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_failed_channel_receive_fails_get(receive_first, asynchronous):
    # Channel.get(async_op=True) waits on a per-query future that only a successful
    # receive resolved, so a failed receive left the trainer waiting forever (NVBug 6955045).
    receive = async_work_type()(fail)
    if receive_first:
        receive(None)
    get = async_work_type("AsyncChannelCommWork")(receive, query_id=1, channel_actor=None)
    if not receive_first:
        receive(None)
    assert get.done(), "a failed receive must wake the channel get"
    with pytest.raises(EOFError, match="receive failed"):
        if asynchronous:
            asyncio.run(get.async_wait())
        else:
            get.wait()


def test_channel_get_keeps_data_received_for_another_query():
    # The channel may answer two gets out of order, so each receive stores its data by query.
    kind, get_type = async_work_type(), async_work_type("AsyncChannelCommWork")
    first, second = kind(lambda: ("second", 2)), kind(lambda: ("first", 1))
    gets = [get_type(first, query_id=1, channel_actor=None), get_type(second, query_id=2, channel_actor=None)]
    first(None)
    second(None)
    assert [get.wait() for get in gets] == ["first", "second"]


def test_failed_receive_keeps_data_already_stored_for_its_get():
    kind, get_type = async_work_type(), async_work_type("AsyncChannelCommWork")
    other, receive = kind(lambda: ("mine", 2)), kind(fail)
    get_type(other, query_id=1, channel_actor=None)
    get = get_type(receive, query_id=2, channel_actor=None)
    other(None)
    receive(None)
    assert get.wait() == "mine"


def channel_actor(*replies):
    """Stub a channel actor whose put calls return the given futures or raise the given errors."""
    replies = list(replies)

    def remote(*args, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(future=lambda: reply)

    return SimpleNamespace(put=SimpleNamespace(remote=remote))


def resolved(value=None, error=None):
    future = ConcurrentFuture()
    if error is None:
        future.set_result(value)
    else:
        future.set_exception(error)
    return future


def test_failed_channel_put_fails_waiter_and_runs_next_put():
    kind = async_work_type("AsyncChannelWork")
    actor = channel_actor(resolved(error=EOFError("channel actor failed")), resolved("stored"))
    failed = kind("channel", "key", actor, "put")
    assert failed.done(), "a failed channel actor call must wake the put"
    with pytest.raises(EOFError, match="channel actor failed"):
        failed.wait()
    # Puts to one key run in order, so a failed put must not block the next one.
    following = kind("channel", "key", actor, "put")
    assert following.done() and following.wait() == "stored"


def test_queued_channel_put_that_cannot_call_actor_fails_waiter():
    kind = async_work_type("AsyncChannelWork")
    first = ConcurrentFuture()
    actor = channel_actor(first, EOFError("actor call failed"))
    kind("channel", "key", actor, "put")
    queued = kind("channel", "key", actor, "put")
    first.set_result("stored")
    assert queued.done(), "a queued put whose actor call raises must not stay pending"
    with pytest.raises(EOFError, match="actor call failed"):
        queued.wait()


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
        # Wrap the receive the way Channel.get(async_op=True) does; that is what the trainer waits on.
        get = async_work_type("AsyncChannelCommWork")(work, query_id=1, channel_actor=None)
        work(None)
        try:
            get.wait()
        except Exception as exc:
            results.put(("timeout", work.done() and get.done(), str(exc)))
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
