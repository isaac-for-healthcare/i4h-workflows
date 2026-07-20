# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local scene-edit bridge for an already-running Isaac Sim scene."""

from __future__ import annotations

import io
import json
import logging
import re
import runpy
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("arena.bridge")


@dataclass
class _BridgeJob:
    script_path: Path | None = None
    inspect_target: str | None = None
    list_objects: bool = False
    list_cameras: bool = False
    capture_output_dir: Path | None = None
    capture_camera_names: set[str] | None = None
    capture_viewport: bool = True
    inspect_env_index: int = 0
    teleport: dict[str, Any] | None = None
    bake: dict[str, Any] | None = None
    timeout_s: float = 30.0
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    cancel_requested: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


def _job_label(job: _BridgeJob) -> str:
    if job.script_path is not None:
        return f"script:{job.script_path.name}"
    if job.list_objects:
        return "objects"
    if job.list_cameras:
        return "cameras"
    if job.capture_output_dir is not None:
        return "capture"
    if job.inspect_target is not None:
        return f"object:{job.inspect_target}"
    if job.teleport is not None:
        return f"teleport:{job.teleport.get('name', '<unknown>')}"
    if job.bake is not None:
        return "bake"
    return "unknown"


def _snapshot_registered_rigid_bodies(stage: Any) -> set[str]:
    """Prim paths carrying a UsdPhysics rigid body at the moment of capture.

    Captured before the first bridge job runs, this is the set of bodies in the
    GPU physics pipeline at sim init. Bodies spawned live (after init) are
    absent, and the move helpers refuse tensor-API access outside this set —
    querying a non-registered body triggers a fatal, unrecoverable CUDA
    illegal-memory access that kills the whole process.
    """
    paths: set[str] = set()
    if stage is None:
        return paths
    try:
        from pxr import UsdPhysics  # noqa: PLC0415

        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                paths.add(str(prim.GetPath()))
    except Exception:  # noqa: BLE001
        pass
    return paths


class BridgeHelpers:
    """Generic context helpers exposed to bridge scripts as ``helpers``."""

    def __init__(self, ctx: SimpleNamespace):
        self.ctx = ctx

    @property
    def stage(self):
        return self.get_stage()

    def get_stage(self):
        import omni.usd  # noqa: PLC0415

        return omni.usd.get_context().get_stage()

    def env_root_path(self, env_index: int = 0) -> str:
        return f"/World/envs/env_{env_index}"

    def env_path(self, relative_path: str = "", env_index: int = 0) -> str:
        relative = relative_path.strip("/")
        root = self.env_root_path(env_index=env_index)
        return f"{root}/{relative}" if relative else root

    def env_live_path(self, relative_path: str, env_index: int = 0) -> str:
        return self.env_path(f"LiveEdit/{relative_path.strip('/')}", env_index=env_index)

    def scene_prim_path(self, name: str, env_index: int = 0) -> str:
        """Resolve an IsaacLab scene key like ``scissors`` to an env prim path.

        Tries, in order:
          1. the runtime asset's ``prim_path``
          2. the runtime asset's ``cfg.prim_path``
          3. the ``InteractiveSceneCfg`` field ``<name>.prim_path`` — required for
             unmanaged ``AssetBaseCfg`` members (lights, ground plane, USD-only
             decorations) that IsaacLab does not wrap in a Python asset object.
        """
        scene = self.ctx.env.scene
        prim_path: str | None = None

        try:
            asset = scene[name]
        except (KeyError, IndexError, TypeError):
            asset = None

        if asset is not None:
            prim_path = getattr(asset, "prim_path", None)
            if prim_path is None:
                cfg = getattr(asset, "cfg", None)
                if cfg is not None:
                    prim_path = getattr(cfg, "prim_path", None)

        if prim_path is None:
            scene_cfg = getattr(scene, "cfg", None)
            if scene_cfg is not None:
                field = getattr(scene_cfg, name, None)
                if field is not None:
                    prim_path = getattr(field, "prim_path", None)

        if prim_path is None:
            raise KeyError(f"scene asset {name!r} does not expose a prim path")

        path = str(prim_path).replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_index}")
        return re.sub(r"/World/envs/env_[^/]*", f"/World/envs/env_{env_index}", path, count=1)

    def resolve_path(self, path_or_scene_key: str, env_index: int = 0) -> str:
        if path_or_scene_key.startswith("/"):
            return path_or_scene_key
        return self.scene_prim_path(path_or_scene_key, env_index=env_index)

    def get_prim(self, path_or_scene_key: str, env_index: int = 0):
        path = self.resolve_path(path_or_scene_key, env_index=env_index)
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"prim does not exist: {path}")
        return prim

    def world_bbox(self, path_or_scene_key: str, env_index: int = 0) -> dict[str, tuple[float, float, float]]:
        """Return the world-aligned bbox of a prim as ``{min, max, size}``.

        Use this for placement math (e.g. "spawn a cube 1 cm above the table top")
        instead of recomputing ``UsdGeom.BBoxCache`` in every script.
        """
        from pxr import UsdGeom  # noqa: PLC0415

        prim = self.get_prim(path_or_scene_key, env_index=env_index)
        bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        bbox_min = tuple(float(value) for value in bbox.GetMin())
        bbox_max = tuple(float(value) for value in bbox.GetMax())
        bbox_size = tuple(hi - lo for lo, hi in zip(bbox_min, bbox_max, strict=True))
        return {"min": bbox_min, "max": bbox_max, "size": bbox_size}

    def surface_top_z(self, path_or_scene_key: str, env_index: int = 0) -> float:
        """World-aligned top z of a prim — convenience for placing new objects on a surface."""
        return self.world_bbox(path_or_scene_key, env_index=env_index)["max"][2]

    def env_origin(self, env_index: int = 0) -> tuple[float, float, float]:
        """World-space origin of env_<index>.

        Bridge ``world_bbox`` returns world coordinates, but ``init_state.pos`` in
        ``arena/assets/<env>.py`` is **env-local**. Subtract this origin from a
        world position before baking it into a config.
        """
        try:
            origins = self.ctx.env.scene.env_origins
        except AttributeError:
            return (0.0, 0.0, 0.0)
        try:
            row = origins[env_index]
        except (IndexError, TypeError):
            return (0.0, 0.0, 0.0)
        try:
            values = row.detach().cpu().tolist()
        except AttributeError:
            values = list(row)
        return tuple(float(values[i]) for i in range(3))

    # ---- Physics-safe moves -------------------------------------------------
    # Driving a PhysX body that is NOT in the GPU pipeline (e.g. a prim spawned
    # live, after sim init) through the tensor API triggers a fatal CUDA
    # illegal-memory access that kills the bridge. The helpers below only touch
    # the tensor API for bodies present at init (`registered_rigid_bodies()`) and
    # fall back to USD otherwise, so a script can't crash the bridge by moving the
    # wrong thing. Prefer `helpers.move(...)` over raw `create_rigid_body_view`.

    def registered_rigid_bodies(self) -> set[str]:
        """Prim paths of rigid bodies present at bridge init (safe for the tensor API)."""
        return set(getattr(self.ctx, "_registered_rb_paths", set()))

    def physics_sim_view(self):
        """Cached PhysX tensor sim view."""
        view = getattr(self.ctx, "_physics_sim_view", None)
        if view is None:
            from isaacsim.core.simulation_manager import SimulationManager  # noqa: PLC0415

            view = SimulationManager.get_physics_sim_view()
            self.ctx._physics_sim_view = view
        return view

    def find_rigid_body(self, path_or_scene_key: str, env_index: int = 0) -> str | None:
        """Prim path of the rigid body at/under a target — the root for a
        ``RigidObjectCfg``, or a deep child mesh for a catalog ``AssetBaseCfg``
        prop (e.g. ``SCISSOR_TRAY_USD``). ``None`` if the target has no body."""
        from pxr import Usd, UsdPhysics  # noqa: PLC0415

        try:
            root = self.get_prim(path_or_scene_key, env_index=env_index)
        except ValueError:
            return None
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
        return None

    def rigid_body_view(self, body_path: str):
        """Guarded PhysX rigid-body view. Refuses bodies not registered at init
        (querying those faults CUDA) — raises ``ValueError`` instead of crashing."""
        if body_path not in self.registered_rigid_bodies():
            raise ValueError(
                f"refusing PhysX tensor view for {body_path!r}: not registered at bridge init "
                "(live-added or non-physics). Querying it risks a fatal CUDA fault — move it with "
                "helpers.move(...) or relaunch so it spawns at init."
            )
        return self.physics_sim_view().create_rigid_body_view(body_path)

    def move(
        self, target: str, pos=None, dpos=None, rot_wxyz=None, *, env_index: int = 0, zero_velocity: bool = True
    ) -> dict[str, Any]:
        """Safely move a scene entity (scene key or prim path) so the pose holds.

        ``pos=(x,y,z)`` sets an absolute position (a ``None`` axis is kept);
        ``dpos=(dx,dy,dz)`` adds a delta; ``rot_wxyz`` optionally sets orientation.
        Registered PhysX bodies (a root ``RigidObject`` or a catalog prop's
        kinematic child mesh) are driven through the tensor view so the pose holds
        across steps; anything else (live-added, non-physics) is moved via its USD
        xform — never through the tensor API, so this cannot fault CUDA. Returns
        ``{mode: "physx"|"usd", ...}``.
        """
        import torch  # noqa: PLC0415

        body = self.find_rigid_body(target, env_index=env_index)
        if body is not None and body in self.registered_rigid_bodies():
            view = self.physics_sim_view().create_rigid_body_view(body)
            t = view.get_transforms().clone()  # [N,7] x,y,z, qx,qy,qz,qw (XYZW)
            if pos is not None:
                for i, v in enumerate(pos):
                    if v is not None:
                        t[:, i] = float(v)
            if dpos is not None:
                for i, d in enumerate(dpos):
                    t[:, i] += float(d)
            if rot_wxyz is not None:
                w, x, y, z = (float(c) for c in rot_wxyz)
                t[:, 3], t[:, 4], t[:, 5], t[:, 6] = x, y, z, w
            idx = torch.arange(view.count, dtype=torch.int32, device=t.device)
            view.set_transforms(t, idx)
            if zero_velocity:
                try:
                    view.set_velocities(torch.zeros((view.count, 6), dtype=torch.float32, device=t.device), idx)
                except Exception:  # noqa: BLE001
                    pass
            return {
                "mode": "physx",
                "body": body,
                "pose": [round(float(v), 4) for v in view.get_transforms()[0].tolist()],
            }

        # USD fallback — live-added / non-physics prim. Never touch the tensor API.
        from pxr import Gf, UsdGeom  # noqa: PLC0415

        prim = self.get_prim(target, env_index=env_index)
        attr = prim.GetAttribute("xformOp:translate")
        has_attr = bool(attr and attr.IsValid())
        cur = attr.Get() if has_attr and attr.HasAuthoredValue() else None
        nx, ny, nz = (float(cur[0]), float(cur[1]), float(cur[2])) if cur is not None else (0.0, 0.0, 0.0)
        if pos is not None:
            if pos[0] is not None:
                nx = float(pos[0])
            if pos[1] is not None:
                ny = float(pos[1])
            if pos[2] is not None:
                nz = float(pos[2])
        if dpos is not None:
            nx += float(dpos[0])
            ny += float(dpos[1])
            nz += float(dpos[2])
        if not has_attr:
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(nx, ny, nz))
        return {
            "mode": "usd",
            "body": body,
            "translate": [round(nx, 4), round(ny, 4), round(nz, 4)],
            "warning": "target not registered at bridge init — moved via USD only; PhysX does "
            "not drive it (a dynamic body won't fall/persist). Relaunch to simulate.",
        }

    def camera_images(self, camera_names: set[str] | None = None) -> dict[str, Any]:
        """Return currently available RGB camera arrays from the live scene."""
        _render_once(self.ctx)
        from arena.dump import camera_images  # noqa: PLC0415

        images = camera_images(None, self.ctx.env)
        if camera_names is not None:
            images = {name: image for name, image in images.items() if name in camera_names}
        return images

    def capture(
        self,
        output_dir: str | Path | None = None,
        *,
        camera_names: set[str] | None = None,
        viewport: bool = True,
    ) -> dict[str, Any]:
        """Save current camera frames and optionally the active viewport."""
        return _capture_scene(
            self, output_dir=_capture_output_dir(output_dir), camera_names=camera_names, viewport=viewport
        )


class _LazyStage:
    def __init__(self, helpers: BridgeHelpers):
        self._helpers = helpers
        self._stage = None

    def _resolve(self):
        if self._stage is None:
            self._stage = self._helpers.get_stage()
        return self._stage

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        return repr(self._resolve())


class BridgeServer:
    """Threaded HTTP bridge that queues work for main-thread execution."""

    def __init__(self, ctx: SimpleNamespace, *, host: str = "127.0.0.1", port: int = 8765):
        self.ctx = ctx
        self.host = host
        self.port = port
        self._jobs: Queue[_BridgeJob] = Queue()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._current_job: _BridgeJob | None = None
        self._last_pump_at: float | None = None
        self._last_job_finished_at: float | None = None
        self._completed_jobs = 0
        self._rejected_jobs = 0

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self.host}:{self.port}"
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        handler = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="arena-bridge", daemon=True)
        self._thread.start()
        logger.info("scene-edit HTTP bridge listening at %s", self.url)

    def shutdown(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        logger.info("scene-edit HTTP bridge stopped")

    def submit(self, job: _BridgeJob) -> tuple[bool, str | None]:
        """Submit a main-loop job unless another one is already pending/running."""
        with self._state_lock:
            if self._current_job is not None:
                current = self._current_job
                age = time.monotonic() - (current.started_at or current.submitted_at)
                self._rejected_jobs += 1
                return False, f"main loop busy running {_job_label(current)!r} for {age:.1f}s"
            if not self._jobs.empty():
                self._rejected_jobs += 1
                return False, "main loop already has a queued job"
            self._jobs.put(job)
        return True, None

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._state_lock:
            current = self._current_job
            status: dict[str, Any] = {
                "queue_depth": self._jobs.qsize(),
                "completed_jobs": self._completed_jobs,
                "rejected_jobs": self._rejected_jobs,
                "last_pump_age_s": None if self._last_pump_at is None else round(now - self._last_pump_at, 3),
                "last_job_finished_age_s": (
                    None if self._last_job_finished_at is None else round(now - self._last_job_finished_at, 3)
                ),
                "busy": current is not None,
            }
            if current is not None:
                status["current_job"] = _job_label(current)
                status["current_job_age_s"] = round(now - (current.started_at or current.submitted_at), 3)
                status["current_job_timeout_s"] = current.timeout_s
            return status

    def pump(self) -> int:
        """Run all pending bridge jobs. Call from the Isaac Sim main thread."""
        with self._state_lock:
            self._last_pump_at = time.monotonic()
        processed = 0
        while True:
            try:
                job = self._jobs.get_nowait()
            except Empty:
                return processed
            processed += 1
            now = time.monotonic()
            if job.done.is_set() or job.cancel_requested or now - job.submitted_at > job.timeout_s:
                if not job.done.is_set():
                    job.response = {
                        "ok": False,
                        "error": f"{_job_label(job)} expired before the main loop started",
                    }
                    job.done.set()
                continue
            with self._state_lock:
                job.started_at = now
                self._current_job = job
            try:
                self._run_job(job)
            finally:
                with self._state_lock:
                    self._current_job = None
                    self._completed_jobs += 1
                    self._last_job_finished_at = time.monotonic()

    def _run_job(self, job: _BridgeJob) -> None:
        helpers = BridgeHelpers(self.ctx)
        if not hasattr(self.ctx, "_registered_rb_paths"):
            # Snapshot init-time rigid bodies before the first job can mutate the
            # scene, so the move helpers refuse tensor access to live-added bodies
            # (querying those faults CUDA and kills the bridge).
            self.ctx._registered_rb_paths = _snapshot_registered_rigid_bodies(helpers.get_stage())
        stdout = io.StringIO()
        stderr = io.StringIO()
        namespace: dict[str, Any] = {
            "app": self.ctx.app,
            "args": self.ctx.args,
            "ctx": self.ctx,
            "env": self.ctx.env,
            "helpers": helpers,
            "get_stage": helpers.get_stage,
            "script_path": str(job.script_path),
            "stage": _LazyStage(helpers),
        }
        try:
            if job.list_objects:
                result = _list_objects(helpers, env_index=job.inspect_env_index)
            elif job.list_cameras:
                result = _list_cameras(helpers)
            elif job.capture_output_dir is not None:
                result = _capture_scene(
                    helpers,
                    output_dir=job.capture_output_dir,
                    camera_names=job.capture_camera_names,
                    viewport=job.capture_viewport,
                )
            elif job.inspect_target is not None:
                result = _inspect_object(helpers, job.inspect_target, env_index=job.inspect_env_index)
            elif job.teleport is not None:
                result = _teleport_asset(helpers, job.teleport)
            elif job.bake is not None:
                result = _bake_snippets(helpers, job.bake)
            elif job.script_path is not None:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    namespace = runpy.run_path(
                        str(job.script_path),
                        init_globals=namespace,
                        run_name="__arena_bridge__",
                    )
                result = namespace.get("result")
            else:
                raise ValueError("bridge job has neither script_path nor inspect_target")
            job.response = {
                "ok": True,
                "result": _json_safe(result),
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
            }
        except BaseException as exc:  # noqa: BLE001
            job.response = {
                "ok": False,
                "error": repr(exc),
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "traceback": traceback.format_exc(),
            }
        finally:
            job.done.set()

    def _make_handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._send_json(
                        {
                            "ok": True,
                            "endpoint": "arena-scene-edit-bridge",
                            "env_id": bridge.ctx.env_id,
                            "script_url": f"{bridge.url}/script",
                            "object_url": f"{bridge.url}/object?name=<scene_key>",
                            "objects_url": f"{bridge.url}/objects",
                            "cameras_url": f"{bridge.url}/cameras",
                            "capture_url": f"{bridge.url}/capture",
                            "teleport_url": f"{bridge.url}/object/teleport",
                            "bake_url": f"{bridge.url}/bake",
                            "main_loop": bridge.status(),
                        }
                    )
                    return
                if parsed.path == "/context":
                    self._send_json(
                        {
                            "ok": True,
                            "endpoints": {
                                "GET /health": "Server readiness and endpoint discovery.",
                                "GET /context": "Execution globals, helper names, and endpoint inventory.",
                                "GET /objects": "List scene objects with kind (articulation/rigid/camera/xform).",
                                "GET /object?name=<scene_key>": "USD xform + bbox + LIVE PhysX pose for the entity (the 'live' field is authoritative for articulations and rigid objects; the 'bbox' is computed from stale USD attributes and should NOT be used to verify a teleport).",
                                "GET /object?path=<prim_path>": "Same payload, looked up by raw USD path.",
                                "GET /cameras": "List live RGB camera outputs.",
                                "POST /capture": 'Save live camera frames and optionally the active viewport to JPEG files. Body: {"output_dir": <abs path>, "viewport": true, "cameras": [<names>?]}.',
                                "POST /script": 'Run a trusted absolute Python script path on Isaac\'s main loop. Body: {"path": <abs path>}.',
                                "POST /object/teleport": 'Live-teleport an articulation or rigid scene asset. Body: {"name": <scene key>, "translation": [x, y, z], "rotation_wxyz": [w, x, y, z], "zero_velocity": true, "env_index": 0}. Returns \'live\' pose for verification.',
                                "POST /bake": 'Return Python code snippets for the current live state of named entities (does not write files). Body: {"names": [<scene keys>]}.',
                            },
                            "globals": ["app", "args", "ctx", "env", "helpers", "script_path", "stage", "get_stage"],
                            "helpers": [
                                "env_root_path",
                                "env_path",
                                "env_live_path",
                                "scene_prim_path",
                                "resolve_path",
                                "get_prim",
                                "world_bbox",
                                "surface_top_z",
                                "env_origin",
                                "camera_images",
                                "capture",
                            ],
                        }
                    )
                    return
                if parsed.path == "/objects":
                    try:
                        params = parse_qs(parsed.query)
                        timeout_s = float(_single_query_value(params, "timeout") or 5.0)
                        env_index = int(_single_query_value(params, "env_index") or 0)
                        job = _BridgeJob(list_objects=True, inspect_env_index=env_index, timeout_s=timeout_s)
                        accepted, error = bridge.submit(job)
                        if not accepted:
                            self._send_json({"ok": False, "error": error, "main_loop": bridge.status()}, status=409)
                            return
                        if not job.done.wait(timeout_s):
                            job.cancel_requested = True
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": f"object list timed out after {timeout_s:.1f}s waiting for main loop",
                                    "main_loop": bridge.status(),
                                },
                                status=504,
                            )
                            return
                        self._send_json(job.response or {"ok": False, "error": "object list produced no response"})
                    except BaseException as exc:  # noqa: BLE001
                        self._send_json({"ok": False, "error": repr(exc)}, status=400)
                    return
                if parsed.path == "/object":
                    try:
                        params = parse_qs(parsed.query)
                        target = _single_query_value(params, "name") or _single_query_value(params, "path")
                        if target is None:
                            raise ValueError("expected query parameter 'name' or 'path'")
                        timeout_s = float(_single_query_value(params, "timeout") or 5.0)
                        env_index = int(_single_query_value(params, "env_index") or 0)
                        job = _BridgeJob(inspect_target=target, inspect_env_index=env_index, timeout_s=timeout_s)
                        accepted, error = bridge.submit(job)
                        if not accepted:
                            self._send_json({"ok": False, "error": error, "main_loop": bridge.status()}, status=409)
                            return
                        if not job.done.wait(timeout_s):
                            job.cancel_requested = True
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": f"object query timed out after {timeout_s:.1f}s waiting for main loop",
                                    "main_loop": bridge.status(),
                                },
                                status=504,
                            )
                            return
                        self._send_json(job.response or {"ok": False, "error": "object query produced no response"})
                    except BaseException as exc:  # noqa: BLE001
                        self._send_json({"ok": False, "error": repr(exc)}, status=400)
                    return
                if parsed.path == "/cameras":
                    try:
                        params = parse_qs(parsed.query)
                        timeout_s = float(_single_query_value(params, "timeout") or 5.0)
                        job = _BridgeJob(list_cameras=True, timeout_s=timeout_s)
                        accepted, error = bridge.submit(job)
                        if not accepted:
                            self._send_json({"ok": False, "error": error, "main_loop": bridge.status()}, status=409)
                            return
                        if not job.done.wait(timeout_s):
                            job.cancel_requested = True
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": f"camera list timed out after {timeout_s:.1f}s waiting for main loop",
                                    "main_loop": bridge.status(),
                                },
                                status=504,
                            )
                            return
                        self._send_json(job.response or {"ok": False, "error": "camera list produced no response"})
                    except BaseException as exc:  # noqa: BLE001
                        self._send_json({"ok": False, "error": repr(exc)}, status=400)
                    return
                self.send_error(404, "expected /health, /context, /objects, /object, or /cameras")

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/capture":
                    try:
                        payload = self._read_json()
                        timeout_s = float(payload.get("timeout", 30.0))
                        output_dir = _capture_output_dir(payload.get("output_dir"))
                        camera_names = _csv_names(payload.get("cameras"))
                        viewport = bool(payload.get("viewport", True))
                        job = _BridgeJob(
                            capture_output_dir=output_dir,
                            capture_camera_names=camera_names,
                            capture_viewport=viewport,
                            timeout_s=timeout_s,
                        )
                        accepted, error = bridge.submit(job)
                        if not accepted:
                            self._send_json({"ok": False, "error": error, "main_loop": bridge.status()}, status=409)
                            return
                        if not job.done.wait(timeout_s):
                            job.cancel_requested = True
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": f"capture timed out after {timeout_s:.1f}s waiting for main loop",
                                    "main_loop": bridge.status(),
                                },
                                status=504,
                            )
                            return
                        self._send_json(job.response or {"ok": False, "error": "capture produced no response"})
                    except BaseException as exc:  # noqa: BLE001
                        self._send_json({"ok": False, "error": repr(exc)}, status=400)
                    return
                if parsed.path == "/object/teleport":
                    try:
                        payload = self._read_json()
                        timeout_s = float(payload.get("timeout", 30.0))
                        job = _BridgeJob(teleport=payload, timeout_s=timeout_s)
                        accepted, error = bridge.submit(job)
                        if not accepted:
                            self._send_json({"ok": False, "error": error, "main_loop": bridge.status()}, status=409)
                            return
                        if not job.done.wait(timeout_s):
                            job.cancel_requested = True
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": f"teleport timed out after {timeout_s:.1f}s waiting for main loop",
                                    "main_loop": bridge.status(),
                                },
                                status=504,
                            )
                            return
                        self._send_json(job.response or {"ok": False, "error": "teleport produced no response"})
                    except BaseException as exc:  # noqa: BLE001
                        self._send_json({"ok": False, "error": repr(exc)}, status=400)
                    return
                if parsed.path == "/bake":
                    try:
                        payload = self._read_json()
                        timeout_s = float(payload.get("timeout", 30.0))
                        job = _BridgeJob(bake=payload, timeout_s=timeout_s)
                        accepted, error = bridge.submit(job)
                        if not accepted:
                            self._send_json({"ok": False, "error": error, "main_loop": bridge.status()}, status=409)
                            return
                        if not job.done.wait(timeout_s):
                            job.cancel_requested = True
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": f"bake timed out after {timeout_s:.1f}s waiting for main loop",
                                    "main_loop": bridge.status(),
                                },
                                status=504,
                            )
                            return
                        self._send_json(job.response or {"ok": False, "error": "bake produced no response"})
                    except BaseException as exc:  # noqa: BLE001
                        self._send_json({"ok": False, "error": repr(exc)}, status=400)
                    return
                if parsed.path not in {"/script", "/execute"}:
                    self.send_error(404, "expected /script, /capture, /object/teleport, or /bake")
                    return
                try:
                    payload = self._read_json()
                    script_path = _script_path_from_payload(payload)
                    if script_path is None:
                        raise ValueError("JSON body must contain absolute script 'path'")
                    timeout_s = float(payload.get("timeout", 30.0))
                    job = _BridgeJob(script_path=script_path, timeout_s=timeout_s)
                    accepted, error = bridge.submit(job)
                    if not accepted:
                        self._send_json({"ok": False, "error": error, "main_loop": bridge.status()}, status=409)
                        return
                    if not job.done.wait(timeout_s):
                        job.cancel_requested = True
                        self._send_json(
                            {
                                "ok": False,
                                "error": f"script timed out after {timeout_s:.1f}s waiting for main loop",
                                "main_loop": bridge.status(),
                            },
                            status=504,
                        )
                        return
                    self._send_json(job.response or {"ok": False, "error": "script produced no response"})
                except BaseException as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": repr(exc)}, status=400)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.info("http %s - %s", self.address_string(), fmt % args)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("missing JSON request body")
                data = self.rfile.read(length)
                value = json.loads(data.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("request body must be a JSON object")
                return value

            def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
                body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _classify_scene_asset(scene: Any, name: str) -> tuple[str, Any | None]:
    """Classify a scene entity as articulation / rigid / camera / xform.

    Returns ``(kind, asset)``. ``asset`` is the runtime object if the name
    resolves through ``scene[name]``, otherwise ``None``. ``kind`` is one of:

    - ``"articulation"`` — has ``data.root_link_pose_w`` (e.g. robot)
    - ``"rigid"``        — has ``data.root_pos_w`` but no articulation root link
    - ``"camera"``       — sensor with ``data.output`` (RGB / depth)
    - ``"xform"``        — anything else (lights, ground plane, static USD decorations)
    """
    try:
        asset = scene[name]
    except (KeyError, IndexError, TypeError):
        return ("xform", None)

    # Camera ``data`` is a lazy property that renders/updates buffers on access.
    # Classification must stay passive; otherwise cheap endpoints such as
    # /objects can unexpectedly run camera work or block on a partially-added
    # live sensor.
    cfg = getattr(asset, "cfg", None)
    if "camera" in type(asset).__name__.lower() or (
        cfg is not None and hasattr(cfg, "data_types") and hasattr(cfg, "height") and hasattr(cfg, "width")
    ):
        return ("camera", asset)

    # Discriminate by the strongest available signal. IsaacLab's RigidObjectData
    # also exposes ``root_link_pose_w``, so that alone is not enough to tell
    # articulations apart from rigid bodies. Articulations are the only ones
    # with joints, so ``joint_pos`` (or ``joint_names``) is the discriminator.
    data = getattr(asset, "data", None)
    if data is not None:
        if hasattr(data, "joint_pos") or hasattr(data, "joint_names"):
            return ("articulation", asset)
        if hasattr(data, "root_pos_w") or hasattr(data, "root_link_pose_w"):
            return ("rigid", asset)
        if hasattr(data, "output"):
            return ("camera", asset)
    return ("xform", asset)


def _live_state(asset: Any, kind: str) -> dict[str, Any] | None:
    """Return the live PhysX/Articulation state for a scene asset, or None.

    USD xform attributes are stale after ``write_root_pose_to_sim``; this
    function returns the authoritative live pose / velocity tensors instead.
    """
    if asset is None:
        return None
    data = getattr(asset, "data", None)
    if data is None:
        return None
    state: dict[str, Any] = {}
    if kind == "articulation":
        pose = getattr(data, "root_link_pose_w", None)
        if pose is not None:
            try:
                state["root_pose_w"] = pose[0].detach().cpu().tolist()
            except Exception:  # noqa: BLE001
                pass
        lin = getattr(data, "root_lin_vel_w", None)
        if lin is not None:
            try:
                state["root_lin_vel_w"] = lin[0].detach().cpu().tolist()
            except Exception:  # noqa: BLE001
                pass
        ang = getattr(data, "root_ang_vel_w", None)
        if ang is not None:
            try:
                state["root_ang_vel_w"] = ang[0].detach().cpu().tolist()
            except Exception:  # noqa: BLE001
                pass
    elif kind == "rigid":
        pos = getattr(data, "root_pos_w", None)
        if pos is not None:
            try:
                state["root_pos_w"] = pos[0].detach().cpu().tolist()
            except Exception:  # noqa: BLE001
                pass
        quat = getattr(data, "root_quat_w", None)
        if quat is not None:
            try:
                state["root_quat_w"] = quat[0].detach().cpu().tolist()
            except Exception:  # noqa: BLE001
                pass
    return state or None


def _env_origin_offset(scene: Any, env_index: int = 0) -> tuple[float, float, float]:
    origins = getattr(scene, "env_origins", None)
    if origins is None:
        return (0.0, 0.0, 0.0)
    try:
        row = origins[env_index]
    except (IndexError, TypeError):
        return (0.0, 0.0, 0.0)
    try:
        values = row.detach().cpu().tolist()
    except AttributeError:
        values = list(row)
    return (float(values[0]), float(values[1]), float(values[2]))


def _teleport_asset(helpers: BridgeHelpers, params: dict[str, Any]) -> dict[str, Any]:
    """Live-teleport an articulation or rigid scene asset.

    Encapsulates the inference-mode wrap, ``env_origins`` offset, dispatch
    between ``Articulation`` and ``RigidObject`` write methods, and zero-velocity
    reset. Returns the live pose after the write so callers can verify without a
    separate /script callback.
    """
    import torch  # noqa: PLC0415

    name = params.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("'name' must be a non-empty scene-entity key")

    translation = params.get("translation")
    if not isinstance(translation, (list, tuple)) or len(translation) != 3:
        raise ValueError("'translation' must be a 3-element list [x, y, z]")
    rotation = params.get("rotation_wxyz")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 4:
        raise ValueError("'rotation_wxyz' must be a 4-element list [w, x, y, z]")

    zero_velocity = bool(params.get("zero_velocity", True))
    env_index = int(params.get("env_index", 0))

    scene = helpers.ctx.env.scene
    kind, asset = _classify_scene_asset(scene, name)
    if kind not in ("articulation", "rigid") or asset is None:
        raise ValueError(
            f"cannot teleport scene entity {name!r} of kind {kind!r}; "
            "only articulations and rigid objects support live teleport"
        )

    device = getattr(asset, "device", None) or scene.device
    unwrapped = getattr(helpers.ctx.env, "unwrapped", helpers.ctx.env)
    num_envs = unwrapped.scene.num_envs

    pos = torch.tensor([list(translation)], device=device, dtype=torch.float32).repeat(num_envs, 1)
    quat = torch.tensor([list(rotation)], device=device, dtype=torch.float32).repeat(num_envs, 1)

    env_origins = getattr(unwrapped.scene, "env_origins", None)
    if env_origins is not None:
        pos = pos + env_origins.to(device=device, dtype=torch.float32)

    # Normalize quaternion to prevent zero-norm / non-unit crashes inside WBC.
    quat_norms = torch.linalg.norm(quat, dim=-1, keepdim=True)
    if torch.any(quat_norms <= 1e-6):
        raise ValueError(f"'rotation_wxyz' has zero norm: {rotation}")
    quat = quat / quat_norms

    root_pose = torch.cat([pos, quat], dim=-1)

    with torch.inference_mode():
        if kind == "articulation":
            asset.write_root_pose_to_sim(root_pose)
            if zero_velocity:
                asset.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=device))
        else:
            # RigidObject exposes the same write methods (or equivalents).
            writer = getattr(asset, "write_root_pose_to_sim", None) or getattr(asset, "write_root_link_pose_to_sim")
            writer(root_pose)
            if zero_velocity:
                vel_writer = getattr(asset, "write_root_velocity_to_sim", None) or getattr(
                    asset, "write_root_com_velocity_to_sim", None
                )
                if vel_writer is not None:
                    vel_writer(torch.zeros((num_envs, 6), device=device))

    return {
        "name": name,
        "kind": kind,
        "env_index": env_index,
        "wrote_world_pose": root_pose[env_index].cpu().tolist(),
        "live": _live_state(asset, kind),
    }


def _bake_snippets(helpers: BridgeHelpers, params: dict[str, Any]) -> dict[str, Any]:
    """Return Python code snippets that persist the named entities' live state.

    For each requested name, the response includes:
      - ``kind``: articulation / rigid / xform / camera
      - ``live``: the live PhysX state (None for xform/camera)
      - ``env_local.pos`` / ``env_local.rot_wxyz``: live state with ``env_origins``
        subtracted, ready to paste into an ``InitialStateCfg``
      - ``code``: a single-line Python snippet you can apply
      - ``hint``: which file the snippet typically belongs in

    The bridge does *not* write the snippet to a repo file; the calling agent
    applies the edit. This keeps the durable code path under the agent's
    control (git, review, etc.).
    """
    names = params.get("names")
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if not isinstance(names, list) or not names:
        raise ValueError("'names' must be a non-empty list of scene-entity keys")
    env_index = int(params.get("env_index", 0))

    scene = helpers.ctx.env.scene
    origin = _env_origin_offset(scene, env_index=env_index)
    out: dict[str, Any] = {}
    for raw in names:
        name = str(raw)
        kind, asset = _classify_scene_asset(scene, name)
        live = _live_state(asset, kind)
        entry: dict[str, Any] = {"kind": kind, "live": live}
        if kind == "articulation" and live and "root_pose_w" in live:
            pose = live["root_pose_w"]
            pos = (pose[0] - origin[0], pose[1] - origin[1], pose[2] - origin[2])
            rot = (pose[3], pose[4], pose[5], pose[6])
            entry["env_local"] = {"pos": list(pos), "rot_wxyz": list(rot)}
            entry["code"] = (
                f"embodiment.set_initial_pose(Pose("
                f"position_xyz=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
                f"rotation_wxyz=({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f})))"
            )
            entry["hint"] = (
                "Place inside the env class's get_env, after constructing the embodiment. "
                "File: arena/environments/<env>_environment.py"
            )
        elif kind == "rigid" and live and "root_pos_w" in live:
            wp = live["root_pos_w"]
            wq = live.get("root_quat_w") or [1.0, 0.0, 0.0, 0.0]
            pos = (wp[0] - origin[0], wp[1] - origin[1], wp[2] - origin[2])
            rot = (wq[0], wq[1], wq[2], wq[3])
            entry["env_local"] = {"pos": list(pos), "rot_wxyz": list(rot)}
            entry["code"] = (
                f"init_state=RigidObjectCfg.InitialStateCfg("
                f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
                f"rot=({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f}))"
            )
            entry["hint"] = (
                "Replace the init_state= line for this rigid object in "
                f"arena/assets/<env>.py:<TaskName>SceneCfg.{name}"
            )
        elif kind == "xform":
            # Static xform — read the USD xform translate/orient as the best
            # available "pose" (no PhysX state). The /object inspector covers
            # this; for bake just return what we can read from the prim.
            try:
                prim_path = helpers.scene_prim_path(name, env_index=env_index)
                stage = helpers.get_stage()
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    from pxr import UsdGeom  # noqa: PLC0415

                    xform = UsdGeom.Xformable(prim)
                    pos = (0.0, 0.0, 0.0)
                    rot = (1.0, 0.0, 0.0, 0.0)
                    for op in xform.GetOrderedXformOps():
                        op_name = op.GetOpName()
                        if "translate" in op_name.lower():
                            value = op.Get()
                            pos = (
                                float(value[0]) - origin[0],
                                float(value[1]) - origin[1],
                                float(value[2]) - origin[2],
                            )
                        elif "orient" in op_name.lower():
                            value = op.Get()
                            rot = (float(value.GetReal()), *[float(v) for v in value.GetImaginary()])
                    entry["env_local"] = {"pos": list(pos), "rot_wxyz": list(rot)}
                    entry["code"] = (
                        f"init_state=AssetBaseCfg.InitialStateCfg("
                        f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
                        f"rot=({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f}))"
                    )
                    entry["hint"] = (
                        f"Replace the init_state= line for this static asset in "
                        f"arena/assets/<env>.py:<TaskName>SceneCfg.{name}"
                    )
            except BaseException as exc:  # noqa: BLE001
                entry["error"] = f"could not read xform: {exc!r}"
        elif kind == "camera":
            entry["hint"] = "Camera entities are baked via TiledCameraCfg.OffsetCfg in arena/assets/<env>.py"
        out[name] = entry
    return {"env_index": env_index, "snippets": out}


def _list_cameras(helpers: BridgeHelpers) -> dict[str, Any]:
    images = helpers.camera_images()
    return {
        "camera_count": len(images),
        "cameras": [
            {
                "name": name,
                "shape": _image_shape(image),
            }
            for name, image in sorted(images.items())
        ],
    }


def _capture_scene(
    helpers: BridgeHelpers,
    *,
    output_dir: Path,
    camera_names: set[str] | None = None,
    viewport: bool = True,
) -> dict[str, Any]:
    from arena.dump import rgb_uint8  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cameras = helpers.camera_images(camera_names=camera_names)
    camera_records = []
    for name, image in sorted(cameras.items()):
        filename = f"{stamp}_{_safe_name(name)}.jpg"
        path = output_dir / filename
        Image.fromarray(rgb_uint8(image)).save(path, format="JPEG", quality=90)
        camera_records.append({"name": name, "path": str(path), "shape": _image_shape(image)})

    viewport_record = (
        _capture_viewport(helpers.ctx, output_dir=output_dir, stamp=stamp)
        if viewport
        else {"ok": False, "skipped": True}
    )
    manifest = {
        "ok": True,
        "env_id": helpers.ctx.env_id,
        "output_dir": str(output_dir),
        "camera_count": len(camera_records),
        "cameras": camera_records,
        "viewport": viewport_record,
    }
    manifest_path = output_dir / f"{stamp}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _capture_output_dir(value: Any = None) -> Path:
    if value is None or value == "":
        return Path(__file__).resolve().parents[2] / "runs" / "arena_bridge" / "captures"
    if not isinstance(value, str):
        raise ValueError("'output_dir' must be a string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("'output_dir' must be absolute")
    return path


def _capture_viewport(ctx: SimpleNamespace, *, output_dir: Path, stamp: str) -> dict[str, Any]:
    path = output_dir / f"{stamp}_viewport.jpg"
    try:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: PLC0415
    except BaseException as exc:  # noqa: BLE001
        return {"ok": False, "path": str(path), "error": f"viewport capture API unavailable: {exc!r}"}

    try:
        viewport_api = get_active_viewport()
        if viewport_api is None:
            return {"ok": False, "path": str(path), "error": "no active viewport"}
        try:
            capture = capture_viewport_to_file(viewport_api, str(path))
        except TypeError:
            capture = capture_viewport_to_file(str(path), viewport_api)
        landed = _wait_for_capture(ctx, capture, path)
        if not path.is_file():
            return {
                "ok": False,
                "path": str(path),
                "error": "viewport capture did not produce a file after timeout",
            }
        if not landed:
            # File exists but our wait loop timed out before we could confirm.
            # The capture API is asynchronous and Kit sometimes drops the file
            # in *just* after our timeout, so return success but include a hint.
            return {"ok": True, "path": str(path), "warning": "file landed after wait timeout"}
        return {"ok": True, "path": str(path)}
    except BaseException as exc:  # noqa: BLE001
        return {"ok": False, "path": str(path), "error": repr(exc)}


def _wait_for_capture(
    ctx: SimpleNamespace, capture: Any, output_path: Path | None = None, *, max_iters: int = 120
) -> bool:
    """Wait for an asynchronous viewport capture to land.

    Returns True if the capture completes within ``max_iters`` render/update
    cycles (each ~16 ms at 60 Hz, so ~2 s total). Returns False if the timer
    expires; the caller should also check the file system because Kit may
    drop the file just after the loop exits.

    Kit's ``capture_viewport_to_file`` returns a Future-like that resolves
    *after* its CompletedCallback fires, which is several app.update() cycles
    after the Kit renderer dispatches the screenshot. Three iterations
    (the previous default) were not enough on heavier scenes — Kit would
    eventually write the file but our wait loop already returned, so we
    falsely reported ``"viewport capture did not produce a file"``.
    """
    for _ in range(max_iters):
        _render_once(ctx)
        update = getattr(ctx.app, "update", None)
        if callable(update):
            update()
        if output_path is not None and output_path.is_file():
            return True
    return output_path is not None and output_path.is_file()


def _render_once(ctx: SimpleNamespace) -> None:
    sim = getattr(getattr(ctx.env, "unwrapped", ctx.env), "sim", None)
    render = getattr(sim, "render", None)
    if callable(render):
        render()


def _image_shape(image: Any) -> list[int]:
    shape = getattr(image, "shape", None)
    if shape is None:
        return []
    return [int(value) for value in shape]


def _list_objects(helpers: BridgeHelpers, *, env_index: int = 0) -> dict[str, Any]:
    stage = helpers.get_stage()
    scene = helpers.ctx.env.scene
    scene_keys = []
    for name in sorted(_scene_names(scene)):
        try:
            prim_path = helpers.scene_prim_path(name, env_index=env_index)
        except BaseException as exc:  # noqa: BLE001
            scene_keys.append({"name": name, "error": repr(exc)})
            continue
        prim = stage.GetPrimAtPath(prim_path)
        kind, _ = _classify_scene_asset(scene, name)
        scene_keys.append(
            {
                "name": name,
                "prim_path": prim_path,
                "type": prim.GetTypeName() if prim and prim.IsValid() else None,
                "kind": kind,
                "valid": bool(prim and prim.IsValid()),
                "active": bool(prim and prim.IsValid() and prim.IsActive()),
            }
        )

    live_root = helpers.env_live_path("", env_index=env_index)
    bridge_prims = []
    root_prim = stage.GetPrimAtPath(live_root)
    if root_prim and root_prim.IsValid():
        for prim in root_prim.GetAllChildren():
            bridge_prims.append(
                {
                    "prim_path": prim.GetPath().pathString,
                    "type": prim.GetTypeName(),
                    "active": prim.IsActive(),
                }
            )

    return {
        "env_index": env_index,
        "env_root": helpers.env_root_path(env_index=env_index),
        "scene_objects": scene_keys,
        "bridge_objects_root": live_root,
        "bridge_objects": bridge_prims,
    }


def _scene_names(scene: Any) -> list[str]:
    """Union of runtime scene keys and InteractiveSceneCfg fields with a prim_path.

    The runtime ``scene.keys()`` only lists managed assets (articulations, rigid
    objects, sensors). Unmanaged ``AssetBaseCfg`` members — lights, the ground
    plane, USD decorations — only show up as fields on ``scene.cfg``. We union
    both so ``/objects`` returns the full picture.
    """
    names: set[str] = set()
    if hasattr(scene, "keys"):
        names.update(str(name) for name in scene.keys())
    entities = getattr(scene, "_entities", None)
    if isinstance(entities, dict):
        names.update(str(name) for name in entities)
    scene_cfg = getattr(scene, "cfg", None)
    if scene_cfg is not None:
        for attr in dir(scene_cfg):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(scene_cfg, attr)
            except BaseException:  # noqa: BLE001
                continue
            if hasattr(value, "prim_path"):
                names.add(attr)
    return sorted(names)


def _inspect_object(helpers: BridgeHelpers, target: str, *, env_index: int = 0) -> dict[str, Any]:
    from pxr import UsdGeom  # noqa: PLC0415

    stage = helpers.get_stage()
    prim_path = helpers.resolve_path(target, env_index=env_index)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"prim does not exist: {prim_path}")

    xform_ops = []
    if prim.IsA(UsdGeom.Xformable):
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            xform_ops.append(
                {
                    "name": op.GetOpName(),
                    "type": str(op.GetOpType()),
                    "value": _sequence_or_repr(op.Get()),
                }
            )

    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    bbox_min = tuple(float(value) for value in bbox.GetMin())
    bbox_max = tuple(float(value) for value in bbox.GetMax())
    bbox_size = tuple(max_value - min_value for min_value, max_value in zip(bbox_min, bbox_max, strict=True))

    children = []
    for child in prim.GetAllChildren():
        children.append(
            {
                "path": child.GetPath().pathString,
                "type": child.GetTypeName(),
                "active": child.IsActive(),
            }
        )

    # Live state lookup: try to resolve target as a scene-entity key. If the
    # caller passed a raw prim path, the scene lookup will fall through.
    scene = helpers.ctx.env.scene
    kind, asset = _classify_scene_asset(scene, target)
    live = _live_state(asset, kind)

    return {
        "target": target,
        "prim_path": prim_path,
        "type": prim.GetTypeName(),
        "kind": kind,
        "active": prim.IsActive(),
        "valid": prim.IsValid(),
        "xform_ops": xform_ops,
        "bbox": {
            "min": bbox_min,
            "max": bbox_max,
            "size": bbox_size,
        },
        "live": live,
        "children": children[:50],
        "child_count": len(children),
    }


def _sequence_or_repr(value: Any) -> Any:
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return repr(value)


def _csv_names(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        names = {item.strip() for item in value.split(",") if item.strip()}
        return names or None
    if isinstance(value, (list, tuple, set)):
        names = {str(item).strip() for item in value if str(item).strip()}
        return names or None
    raise ValueError("'cameras' must be a comma-separated string or list")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "camera"


def _single_query_value(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    if not values:
        return None
    return values[-1]


def _script_path_from_payload(payload: dict[str, Any]) -> Path | None:
    value = payload.get("path", payload.get("script_path"))
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'path' must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("'path' must be absolute")
    if not path.is_file():
        raise ValueError(f"script path does not exist: {path}")
    return path
