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

import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch


def set_viewport_camera(camera_prim_path: str) -> None:
    """Set the active viewport camera to a specific camera prim."""
    import omni.kit.commands
    from omni.kit.viewport.utility import get_active_viewport

    viewport_api = get_active_viewport()
    if viewport_api is None:
        return

    omni.kit.commands.execute(
        "SetViewportCamera",
        camera_path=camera_prim_path,
        viewport_api=viewport_api,
    )


class _MultiViewConcatWriter:
    """Stream-write one MP4 per view, concatenating all episodes in order."""

    def __init__(self, video_dir: str, *, base_name: str, fps: int = 30) -> None:
        self._dir = Path(video_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._base_name = base_name
        self._fps = int(fps)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._sizes: dict[str, tuple[int, int]] = {}
        self._frame_counts: dict[str, int] = {}
        self._warned_missing: set[str] = set()

    def _ensure_writer(self, stream_key: str, frame_bgr: np.ndarray) -> cv2.VideoWriter:
        h, w = frame_bgr.shape[:2]
        if stream_key in self._writers:
            # Enforce constant size per view to keep the writer happy.
            vw, vh = self._sizes[stream_key]
            if (w, h) != (vw, vh):
                frame_bgr = cv2.resize(frame_bgr, (vw, vh))
            return self._writers[stream_key]

        out_path = self._dir / f"{self._base_name}_{stream_key}.mp4"
        writer = cv2.VideoWriter(str(out_path), self._fourcc, self._fps, (w, h))
        self._writers[stream_key] = writer
        self._sizes[stream_key] = (w, h)
        self._frame_counts[stream_key] = 0
        print(f"Video: Recording stream '{stream_key}' -> {out_path.name}")
        return writer

    @staticmethod
    def _to_uint8_rgb(frame: Any) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach()
            if frame.dtype != torch.uint8:
                frame = (frame * 255).to(torch.uint8)
            frame = frame.cpu().numpy()
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        return frame

    def write_title_card(self, stream_key: str, text: str, *, frames: int = 10) -> None:
        if stream_key not in self._writers:
            # Defer until we see the first real frame for this view.
            return
        w, h = self._sizes[stream_key]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, text, (20, max(40, h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for _ in range(max(1, int(frames))):
            self._writers[stream_key].write(img)
            self._frame_counts[stream_key] += 1

    def _timecode_for_view(self, stream_key: str) -> str:
        """Return cumulative timecode HH:MM:SS:FF based on frames written for a view."""
        frames = int(self._frame_counts.get(stream_key, 0))
        fps = max(1, int(self._fps))
        total_seconds = frames // fps
        ff = frames % fps
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    @staticmethod
    def _draw_center_hud(
        frame_bgr: np.ndarray,
        *,
        top_text: str,
        bottom_text: str,
        green: tuple[int, int, int] = (0, 200, 0),
        scale: float = 1.0,
    ) -> None:
        """Draw a centered HUD like:

        [white box: top_text]
        [green divider]
        [black box: bottom_text]
        """
        h, w = frame_bgr.shape[:2]
        s = float(scale) if scale is not None else 1.0
        if not np.isfinite(s) or s <= 0:
            s = 1.0

        pad_x = int(round(14 * s))
        pad_y = int(round(6 * s))
        font = cv2.FONT_HERSHEY_SIMPLEX
        top_scale = 0.9 * s
        bot_scale = 0.7 * s
        thickness = max(1, int(round(2 * s)))

        (tw, th), _ = cv2.getTextSize(top_text, font, top_scale, thickness)
        (bw, bh), _ = cv2.getTextSize(bottom_text, font, bot_scale, thickness)
        box_w = max(tw, bw) + pad_x * 2

        top_h = th + pad_y * 2
        div_h = max(1, int(round(5 * s)))
        bot_h = bh + pad_y * 2

        x1 = max(0, (w - box_w) // 2)
        x2 = min(w, x1 + box_w)
        y1 = max(0, int(round(8 * s)))
        y2 = min(h, y1 + top_h + div_h + bot_h)

        # White top box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y1 + top_h), (255, 255, 255), -1)
        # Green divider
        cv2.rectangle(frame_bgr, (x1, y1 + top_h), (x2, y1 + top_h + div_h), green, -1)
        # Black bottom box
        cv2.rectangle(frame_bgr, (x1, y1 + top_h + div_h), (x2, y2), (0, 0, 0), -1)

        # Centered text
        top_x = x1 + (box_w - tw) // 2
        top_y = y1 + pad_y + th
        cv2.putText(frame_bgr, top_text, (top_x, top_y), font, top_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        bot_x = x1 + (box_w - bw) // 2
        bot_y = y1 + top_h + div_h + pad_y + bh
        cv2.putText(frame_bgr, bottom_text, (bot_x, bot_y), font, bot_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def write_from_scene_camera(
        self,
        env,
        *,
        view: str,
        camera_key: str,
        env_index: int = 0,
        stream_key: str | None = None,
        overlay_text: str | None = None,
        overlay_scale: float = 1.0,
    ) -> None:
        if stream_key is None:
            stream_key = view
        cam = env.scene[camera_key]
        rgb_out = cam.data.output["rgb"]
        # IsaacLab camera outputs are typically (num_envs, H, W, 3). Allow single-view too.
        if isinstance(rgb_out, torch.Tensor) and rgb_out.ndim >= 4:
            idx = int(max(0, env_index))
            idx = min(idx, int(rgb_out.shape[0]) - 1)
            frame = rgb_out[idx]
        else:
            frame = rgb_out

        rgb = self._to_uint8_rgb(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if overlay_text:
            top_text = self._timecode_for_view(stream_key)
            self._draw_center_hud(bgr, top_text=top_text, bottom_text=overlay_text, scale=overlay_scale)

        writer = self._ensure_writer(stream_key, bgr)
        vw, vh = self._sizes[stream_key]
        if bgr.shape[1] != vw or bgr.shape[0] != vh:
            bgr = cv2.resize(bgr, (vw, vh))
        writer.write(bgr)
        self._frame_counts[stream_key] += 1

    def close(self) -> None:
        for w in self._writers.values():
            w.release()


# Camera sensor -> observation key mapping
_CAMERA_MAP = {
    "front_camera": "video.room_view",
    "left_wrist_camera": "video.left_wrist_view",
    "right_wrist_camera": "video.right_wrist_view",
}

# Camera view -> sensor key mapping for video recording
_VIDEO_CAMERAS = [
    ("head", "front_camera"),
    ("left_wrist", "left_wrist_camera"),
    ("right_wrist", "right_wrist_camera"),
]


def process_observation(obs: Dict[str, Any], env, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Convert IsaacLab observations to Gr00t policy input format."""
    processed_obs: Dict[str, torch.Tensor] = {}

    # Get camera images directly from scene sensors
    sensors_dict = getattr(env.scene, "sensors", None)
    if sensors_dict:
        for cam_key, obs_key in _CAMERA_MAP.items():
            if cam_key in sensors_dict:
                rgb = sensors_dict[cam_key].data.output["rgb"]  # (env_num, H, W, 3)
                if rgb.dtype != torch.uint8:
                    rgb = (rgb * 255).to(torch.uint8)
                if rgb.ndim == 3:
                    rgb = rgb.unsqueeze(0)
                processed_obs[obs_key] = rgb.to(device)

    # Process robot joint state from observation
    dex3_states = obs["policy"]["robot_dex3_joint_state"]  # (bs, 14)
    g129_shoulder_states = obs["policy"]["robot_joint_state"][:, 15:29]  # (bs, 14) - arms only

    processed_obs["state.left_arm"] = g129_shoulder_states[:, :7].to(device)
    processed_obs["state.right_arm"] = g129_shoulder_states[:, 7:14].to(device)
    processed_obs["state.left_hand"] = dex3_states[:, :7].to(device)
    processed_obs["state.right_hand"] = dex3_states[:, 7:14].to(device)

    return processed_obs


def check_success(env, success_stage: int = 4) -> torch.Tensor:
    """Vectorized success check per env. Returns bool tensor of shape (num_envs,)."""
    if hasattr(env, "_task_stage"):
        stage = env._task_stage
        if isinstance(stage, torch.Tensor):
            return stage >= success_stage
    return torch.zeros(int(getattr(env, "num_envs", 1) or 1), dtype=torch.bool, device=env.device)


def _record_cameras(
    writer: _MultiViewConcatWriter,
    env,
    *,
    env_index: int = 0,
    stream_prefix: str = "",
    overlay_text: str | None = None,
) -> None:
    """Record all cameras for a given env to the video writer."""
    for view, cam_key in _VIDEO_CAMERAS:
        overlay_text = overlay_text if view == "head" else None
        stream_key = f"{stream_prefix}{view}" if stream_prefix else view
        writer.write_from_scene_camera(
            env,
            view=view,
            camera_key=cam_key,
            env_index=env_index,
            stream_key=stream_key,
            overlay_text=overlay_text,
        )


def evaluate_episode(
    env,
    policy,
    *,
    max_steps: int,
    num_episodes: int,
    action_chunk_size: int = 1,
    frequency_hz: float = 0.0,
    task_description: str = "assemble trocar from tray",
    success_stage: int = 4,
    save_video: bool = False,
    video_writer: _MultiViewConcatWriter | None = None,
    video_env_id: int = 0,
    save_video_all_envs: bool = False,
) -> List[Dict[str, Any]]:
    """Unified evaluation function supporting both single-env and multi-env scenarios.

    When num_envs == 1: runs episodes sequentially with detailed per-episode logging.
    When num_envs > 1: runs episodes in parallel across envs, resetting only done envs.
    """
    num_envs = int(getattr(env, "num_envs", 1) or 1)
    action_chunk_size = int(max(1, min(16, action_chunk_size)))
    is_single_env = num_envs == 1

    # Assign episode ids to env slots.
    next_ep = 0
    active = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    ep_id = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
    for i in range(num_envs):
        if next_ep < num_episodes:
            active[i] = True
            ep_id[i] = next_ep
            next_ep += 1

    # Reset all envs once.
    obs, _ = env.reset()

    step_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    reward_sum = torch.zeros(num_envs, dtype=torch.float32, device=env.device)

    results: List[Dict[str, Any]] = []
    action_buffer: List[Any] = []

    hz = float(frequency_hz or 0.0)
    period_s = 1.0 / hz if hz > 0 else 0.0

    # For single-env: track successes so far for overlay text.
    def _successes_so_far() -> int:
        return sum(1 for r in results if r.get("success"))

    # Print episode header for single-env mode.
    if is_single_env and active[0]:
        print(f"\n{'=' * 60}")
        print(f"Episode {int(ep_id[0].item()) + 1}")
        print(f"{'=' * 60}")

    while len(results) < num_episodes:
        loop_start_time = time.perf_counter()

        if not action_buffer:
            processed_obs = process_observation(obs, env, device=policy.device)
            for key in processed_obs:
                processed_obs[key] = processed_obs[key].unsqueeze(1)

            processed_obs["annotation.human.task_description"] = [task_description] * num_envs
            processed_obs_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in processed_obs.items()}

            action_dict = policy.get_action(processed_obs_cpu)
            action_chunk = np.concatenate(
                [np.atleast_1d(action_dict[key]) for key in action_dict.keys()],
                axis=-1,
            )
            # Handle shape: ensure (num_envs, 16, action_dim)
            if action_chunk.ndim == 2:
                # Single env: (16, action_dim) -> (1, 16, action_dim)
                action_chunk = action_chunk[np.newaxis, :, :]
            if action_chunk.shape[-1] == 28:
                action_chunk = np.concatenate([np.zeros((action_chunk.shape[0], 16, 15)), action_chunk], axis=-1)
            action_buffer = [action_chunk[:, t, :] for t in range(action_chunk_size)]

        action_arr = action_buffer.pop(0)
        action_tensor = torch.as_tensor(action_arr, device=env.device, dtype=torch.float32)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0).repeat(num_envs, 1)
        obs, reward, terminated, truncated, _info = env.step(action_tensor)

        # Update per-env stats for active envs only.
        if isinstance(reward, torch.Tensor) and reward.ndim > 0:
            reward_sum[active] += reward[active].to(torch.float32)
        elif isinstance(reward, torch.Tensor):
            reward_sum[active] += reward.item()
        step_count[active] += 1

        # Optional pacing.
        if period_s > 0:
            elapsed = time.perf_counter() - loop_start_time
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        current_stage = env._task_stage[0].item() if hasattr(env, "_task_stage") else -1

        # Video recording.
        if save_video and video_writer is not None:
            env.sim.render()
            if bool(save_video_all_envs) and num_envs > 1:
                # Record all envs.
                for vid_i in range(num_envs):
                    if not bool(active[vid_i].item()):
                        continue
                    epn = int(ep_id[vid_i].item())
                    ep_str = f"{epn + 1}/{num_episodes}" if epn >= 0 else "-"
                    overlay = f"env{vid_i}  ep {ep_str}"
                    _record_cameras(
                        video_writer, env, env_index=vid_i, stream_prefix=f"env{vid_i}_", overlay_text=overlay
                    )
            else:
                # Record single env (either single-env mode or specified video_env_id).
                vid_i = 0 if is_single_env else int(max(0, min(int(video_env_id), num_envs - 1)))
                # Only record if target env is still active.
                if bool(active[vid_i].item()):
                    epn = int(ep_id[vid_i].item())
                    ep_str = f"{epn + 1}/{num_episodes}" if epn >= 0 else "-"
                    if is_single_env:
                        overlay = f"Success: {_successes_so_far()}   Ep: {ep_str}"
                        _record_cameras(video_writer, env, overlay_text=overlay)
                    else:
                        overlay = f"env{vid_i}  ep {ep_str}"
                        _record_cameras(video_writer, env, env_index=vid_i, overlay_text=overlay)

        # Determine per-env done.
        success = check_success(env, success_stage=success_stage)
        done = success.clone()
        if isinstance(terminated, torch.Tensor):
            done |= terminated.to(torch.bool)
        if isinstance(truncated, torch.Tensor):
            done |= truncated.to(torch.bool)
        # Enforce max_steps per environment (script-level timeout).
        if int(max_steps) > 0:
            done |= step_count >= int(max_steps)
        done &= active

        # Single-env progress logging.
        if is_single_env and int(step_count[0].item()) % 50 == 0 and not done[0]:
            step_now = int(step_count[0].item())
            reward_now = reward_sum[0].item()
            print(f"  Step {step_now}/{max_steps}, Stage: {current_stage}, Reward: {reward_now:.2f}")

        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.ndim == 0:
                done_ids = done_ids.unsqueeze(0)
            stage = getattr(env, "_task_stage", None)

            # Record finished episodes for these envs.
            for i in done_ids.tolist():
                ep_success = bool(success[i].item())
                ep_steps = int(step_count[i].item())
                ep_reward = float(reward_sum[i].item())
                ep_num = int(ep_id[i].item())
                ep_stage = int(stage[i].item()) if isinstance(stage, torch.Tensor) else -1

                results.append(
                    {
                        "success": ep_success,
                        "steps": ep_steps,
                        "total_reward": ep_reward,
                        "episode": ep_num,
                        "final_stage": ep_stage,
                        "env_id": int(i),
                    }
                )

                # Single-env: print episode result and success marker frames.
                if is_single_env:
                    if ep_success:
                        print(f"Task completed at step {ep_steps}! (Stage: {ep_stage})")
                        if save_video and video_writer is not None:
                            env.sim.render()
                            overlay_done = f"Success: {_successes_so_far()}   Ep: {ep_num + 1}/{num_episodes}"
                            for _ in range(12):
                                _record_cameras(video_writer, env, overlay_text=overlay_done)
                    else:
                        print(f"Episode terminated at step {ep_steps} (Stage: {ep_stage})")

                    print(f"\nEpisode {ep_num + 1} Results:")
                    print(f"  Success: {'Yes' if ep_success else 'No'}")
                    print(f"  Steps: {ep_steps}")
                    print(f"  Total Reward: {ep_reward:.2f}")

            # Re-assign new episodes to finished envs (if remaining).
            reset_ids: List[int] = []
            for i in done_ids.tolist():
                if next_ep < num_episodes:
                    ep_id[i] = next_ep
                    next_ep += 1
                    reset_ids.append(i)
                    step_count[i] = 0
                    reward_sum[i] = 0.0
                else:
                    active[i] = False
                    ep_id[i] = -1
                    step_count[i] = 0
                    reward_sum[i] = 0.0

            if reset_ids:
                env_ids_t = torch.as_tensor(reset_ids, device=env.device, dtype=torch.int64)
                env.reset(env_ids=env_ids_t)
                # Multi-env: do NOT step the sim (would advance other envs).
                # Single-env: can step since there's only one env.
                _apply_cfg_default_pose(
                    env,
                    settle_steps=20 if is_single_env else 0,
                    env_ids=env_ids_t,
                    allow_sim_steps=is_single_env,
                )
                # Refresh observations for next loop.
                if hasattr(env, "get_observations"):
                    obs = env.get_observations()
                elif hasattr(env, "_get_observations"):
                    obs = env._get_observations()

                # Single-env: print next episode header.
                if is_single_env and len(reset_ids) > 0 and active[0]:
                    print(f"\n{'=' * 60}")
                    print(f"Episode {int(ep_id[0].item()) + 1}")
                    print(f"{'=' * 60}")

            # Clear action buffer so new episodes don't consume old chunk steps.
            action_buffer.clear()

    results.sort(key=lambda r: int(r.get("episode", 0)))
    return results


def _apply_cfg_default_pose(
    env,
    settle_steps: int = 8,
    env_ids: torch.Tensor | None = None,
    *,
    allow_sim_steps: bool = True,
) -> None:
    """Force robot to its configured default joint pose (from cfg) after env.reset().

    This is useful when actuator targets or external controllers can pull the robot away
    from the init pose immediately after reset.
    """
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.int64)

    # These defaults are populated from the asset init_state / cfg.
    q = robot.data.default_joint_pos[env_ids].clone()
    qd = robot.data.default_joint_vel[env_ids].clone()

    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    robot.set_joint_position_target(q[0] if env.num_envs == 1 else q)
    robot.set_joint_velocity_target(qd[0] if env.num_envs == 1 else qd)
    env.scene.write_data_to_sim()

    if allow_sim_steps and int(settle_steps) > 0:
        for _ in range(max(1, int(settle_steps))):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
    # Ensure a render/forward so sensors (cameras) can update on next capture.
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.render()
