# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from aiohttp import web
from aiohttp.web import middleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

webrtc_pcs: set[RTCPeerConnection] = set()


@middleware
async def cors_middleware(request, handler):
    """Add CORS headers to all responses."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


class CameraFrameTrack(VideoStreamTrack):
    """A video track that streams the latest pushed camera frame at a given framerate."""

    def __init__(self, fps: int = 30):
        super().__init__()
        self.fps = fps
        self.frame_count = 0
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def update_frame(self, frame: np.ndarray) -> None:
        """Update the latest frame to be sent to the client."""
        if frame is None:
            return
        # Ensure we always have an RGB image
        if frame.ndim == 3 and frame.shape[-1] >= 3:
            frame = frame[..., :3]
        with self._lock:
            # Copy to decouple from the simulation buffers
            self._frame = np.ascontiguousarray(frame)

    async def recv(self):
        """Generate video frames from the latest camera image."""
        pts, time_base = await self.next_timestamp()

        with self._lock:
            if self._frame is None:
                # Lazy-create a black frame until the first real frame arrives
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                img = self._frame

        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base

        self.frame_count += 1
        if self.frame_count % 100 == 0:
            self._logger.debug("Sent %d frames on WebRTC cam stream", self.frame_count)

        return frame


active_video_tracks: set[CameraFrameTrack] = set()


async def ice_servers(request):
    """Return ICE server configuration."""
    return web.Response(
        content_type="application/json",
        text=json.dumps([{"urls": ["stun:stun.l.google.com:19302"]}]),
    )


async def offer(request):
    """Handle WebRTC offer from client."""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    webrtc_pcs.add(pc)
    logger = logging.getLogger(__name__)
    logger.info("Created WebRTC peer connection (total: %d)", len(webrtc_pcs))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("WebRTC connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            webrtc_pcs.discard(pc)
            video_track = getattr(pc, "_video_track", None)
            if video_track:
                active_video_tracks.discard(video_track)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info("WebRTC ICE connection state: %s", pc.iceConnectionState)

    fps = request.app["fps"]

    # Add video track
    video_track = CameraFrameTrack(fps=fps)
    pc.addTrack(video_track)
    pc._video_track = video_track  # type: ignore[attr-defined]
    active_video_tracks.add(video_track)
    logger.info("Added WebRTC video track (total active: %d)", len(active_video_tracks))

    # Handle the offer
    await pc.setRemoteDescription(offer)
    logger.info("Set remote description (offer)")

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("Created and set local description (answer)")

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        ),
    )


async def on_webrtc_shutdown(app):
    """Clean up peer connections on shutdown."""
    coros = [pc.close() for pc in webrtc_pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    webrtc_pcs.clear()
    active_video_tracks.clear()


async def _run_webrtc_server(image_fps: int, host: str, port: int):
    """Run the WebRTC server (internal helper, used from a background thread)."""
    logger = logging.getLogger(__name__)

    app = web.Application(middlewares=[cors_middleware])
    app["fps"] = image_fps
    app.on_shutdown.append(on_webrtc_shutdown)
    app.router.add_get("/iceServers", ice_servers)
    app.router.add_post("/offer", offer)

    logger.info("Starting WebRTC cam server on %s:%d", host, port)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def start_webrtc_server(host: str = "0.0.0.0", port: int = 8080, fps: int = 30) -> None:
    """Start the WebRTC server in a background daemon thread.

    This is intentionally minimal: the thread exits automatically when the main
    process exits, so we do not currently expose explicit shutdown hooks.
    """

    def _run():
        asyncio.run(_run_webrtc_server(image_fps=fps, host=host, port=port))

    thread = threading.Thread(target=_run, name="webrtc-cam-server", daemon=True)
    thread.start()


def publish_webrtc_frame(frame: np.ndarray) -> None:
    """Push a new camera frame to all active WebRTC tracks."""
    for track in list(active_video_tracks):
        track.update_frame(frame)


def setup_webrtc_cam(args: Any, camera_name: str = "robot_head_cam_rgb") -> Callable[[dict[str, Any]], None]:
    """Configure WebRTC camera streaming and return a publisher callback.

    The returned function ``_maybe_publish_cam(observation)`` can be called
    on each environment step; it will publish frames when streaming is enabled
    via CLI args and otherwise becomes a no-op.

    The camera observation is looked up in the following order:
    1. ``observation["camera_obs"][camera_name]``
    2. ``observation["policy"][camera_name]``

    Args:
        args: Parsed CLI arguments (expects ``webrtc_cam``, ``webrtc_host``,
              ``webrtc_port``, ``webrtc_fps`` attributes).
        camera_name: The key identifying the camera in the observation dict.
                     Defaults to ``"robot_head_cam_rgb"``.

    Returns:
        A callback that publishes frames to WebRTC when called with an observation dict.
    """
    webrtc_enabled = getattr(args, "webrtc_cam", False)
    if not webrtc_enabled:
        return lambda *_args, **_kwargs: None  # type: ignore[return-value]

    start_webrtc_server(
        host=getattr(args, "webrtc_host", "0.0.0.0"),
        port=getattr(args, "webrtc_port", 8080),
        fps=getattr(args, "webrtc_fps", 30),
    )

    webrtc_error_logged = False

    def _maybe_publish_cam(observation: dict[str, Any]) -> None:
        nonlocal webrtc_error_logged
        try:
            # Camera observations may be stored either under the aggregated
            # camera_obs group or directly in the policy observation group.
            if "camera_obs" in observation and camera_name in observation["camera_obs"]:
                camera_obs = observation["camera_obs"][camera_name]
            elif "policy" in observation and camera_name in observation["policy"]:
                camera_obs = observation["policy"][camera_name]
            else:
                raise KeyError(
                    f"Camera '{camera_name}' not found in observation. Top-level keys: {list(observation.keys())}"
                )

            # Support torch tensors and numpy arrays
            if hasattr(camera_obs, "detach"):
                frame = camera_obs[0].detach().cpu().numpy()
            else:
                frame = np.array(camera_obs[0])

            # Convert from CHW to HWC if needed
            if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (3, 4):
                frame = np.moveaxis(frame, 0, -1)

            # Ensure uint8 for WebRTC; handle both [0, 1] and [0, 255] float ranges.
            if frame.dtype != np.uint8:
                max_val = float(frame.max()) if frame.size else 0.0
                if max_val <= 1.0 + 1e-3:
                    frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

            publish_webrtc_frame(frame)
        except Exception as e:  # noqa: BLE001
            if not webrtc_error_logged:
                print(f"[WARNING] Failed to publish WebRTC frame for '{camera_name}': {e}")
                webrtc_error_logged = True

    return _maybe_publish_cam
