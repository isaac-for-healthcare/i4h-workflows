# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arena teleop dispatcher.

Robot-agnostic: per-env helpers (under ``arena.teleop.helpers``) supply a
``make_teleop_interface(env, args_cli)`` factory and optional action
post-processing; this module runs the episode loop. Device modules under
``arena.teleop.devices`` are imported lazily so a workflow that doesn't
support a given device never imports its code.

Reserved keys (consumed by this loop, not the device):

* ``B`` — start the episode. The leisaac Se3Keyboard already gates ``advance()``
  on its internal ``started`` flag; until B is pressed, ``advance()`` returns
  None and the env idles.
* ``N`` — mark current episode as successful; save it (when recording is on)
  and advance to the next attempt.
* ``R`` — discard current episode, env reset, start a new attempt.

There is intentionally no in-viewer abort key. We tried ``P``, ``Y``, and
``ESCAPE`` — all were swallowed by Kit's viewport before they could reach
the leisaac callback. To end a teleop session, run
``workflows/agentic/stop.sh arena --env <env_id>`` from a terminal; the
loop unwinds, ``AppLauncher`` tears down, and the Isaac Sim window closes.
The ``abort`` path is still wired internally so the SIGTERM cascade
unwinds cleanly. ``F`` is reserved by Isaac Sim for Frame-Selected.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import torch
from arena.recording import close_recording, save_episode, setup_recording
from arena.runtimes.core.base import ready

logger = logging.getLogger("arena")

# Per-env hook signature: (actions, leader_reference) -> (sim_actions, leader_reference).
# SO-ARM passes a clamp+leader-sync impl; humanoid + Franka use identity.
ActionPostprocess = Callable[[torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor | None]]


def _identity_postprocess(
    actions: torch.Tensor, leader_reference: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return actions, leader_reference


class TeleopEvents:
    """Keypress flags consumed by the teleop loop.

    ``N`` saves the episode, ``R`` discards and restarts the attempt within
    the same teleop session. ``abort_requested`` is no longer driven by an
    in-viewer key (no Kit-safe binding survived testing); it stays in place
    so the SIGTERM-driven shutdown path can flip it if needed.
    """

    def __init__(self) -> None:
        self.reset_requested = False
        self.success_requested = False
        self.abort_requested = False

    def request_reset(self) -> None:
        self.reset_requested = True

    def request_success(self) -> None:
        self.success_requested = True

    def request_abort(self) -> None:
        self.abort_requested = True

    def clear(self) -> None:
        self.reset_requested = False
        self.success_requested = False
        # abort_requested is intentionally not cleared — it terminates the session.


def run_teleop_job(
    ctx,
    args_cli,
    *,
    make_teleop_interface: Callable[[object, object], object],
    action_postprocess: ActionPostprocess = _identity_postprocess,
    sync_first_action: bool = False,
    on_first_action: Callable[[object, torch.Tensor], None] | None = None,
) -> None:
    """Run a teleop session. Each completed episode is saved when recording is on.

    The teleop device wiring + any env-specific action post-processing
    (e.g. joint clamping, leader-arm remap) is supplied by the caller, so
    this loop stays env-agnostic.
    """
    if args_cli.record_to:
        setup_recording(ctx, args_cli.record_to)

    teleop_interface = make_teleop_interface(ctx.env, args_cli)
    events = TeleopEvents()
    _add_teleop_callbacks(teleop_interface, events)
    teleop_interface.reset()
    completed = 0
    try:
        while not ctx.controller.should_abort() and ctx.simulation_app.is_running():
            if args_cli.episodes > 0 and completed >= args_cli.episodes:
                break
            ctx.env.reset()
            # Reset the teleop adapter too so accumulated targets (e.g.
            # keyboard_23d hand/base/torso pose state) don't carry across
            # episodes — otherwise the WBC drives the robot back to its
            # pre-R pose right after env.reset() lands.
            teleop_interface.reset()
            events.clear()
            # Teleop runs untimed — only ``N`` (save) / ``R`` (reset) end
            # an episode from the viewer; the SIGTERM/stop.sh path triggers
            # the abort branch via ``ready(ctx)`` going False. The step cap
            # (meaningful for policy rollouts) is intentionally ignored here.
            status = run_teleop_based_episode(
                ctx,
                teleop=teleop_interface,
                events=events,
                max_timesteps=None,
                action_postprocess=action_postprocess,
                sync_first_action=sync_first_action,
                on_first_action=on_first_action,
            )
            if ctx.controller.should_abort() or not ctx.simulation_app.is_running():
                break
            # Save only on N (success). R (reset) discards and continues;
            # a sim shutdown (stop.sh / SIGTERM) discards and exits.
            if status == "completed":
                if args_cli.record_to:
                    save_episode(
                        ctx,
                        metadata={
                            "env_id": getattr(ctx, "env_id", None),
                            "run_id": f"teleop-{completed + 1:03d}",
                            "episode_index": completed + 1,
                        },
                    )
                completed += 1
                ctx.controller.episode_completed()
                logger.info("teleop episode %s saved", completed)
            else:
                logger.info("teleop episode discarded (status=%s)", status)
            if events.abort_requested or status == "abort":
                logger.info("teleop session aborted (sim shutdown / stop.sh)")
                break
    finally:
        if args_cli.record_to:
            close_recording(ctx)


@torch.no_grad()
def run_teleop_based_episode(
    ctx,
    *,
    teleop,
    events: TeleopEvents,
    max_timesteps: int | None,
    action_postprocess: ActionPostprocess = _identity_postprocess,
    sync_first_action: bool = False,
    on_first_action: Callable[[object, torch.Tensor], None] | None = None,
) -> str:
    """Step the env with teleop-driven actions.

    Returns:
        ``completed`` if the user pressed N or the step cap was reached;
        ``reset`` if R was pressed; ``abort`` if the sim shut down (e.g.
        SIGTERM via stop.sh).
    """
    step = 0
    leader_reference: torch.Tensor | None = None
    while max_timesteps is None or max_timesteps <= 0 or step < max_timesteps:
        if not ready(ctx):
            return "abort"
        if events.success_requested:
            return "completed"
        if events.abort_requested:
            return "abort"
        if events.reset_requested:
            return "reset"
        actions = teleop.advance()
        if actions is None:
            ctx.env.sim.render()
            time.sleep(0.001)  # brief yield while waiting on the device
            continue
        actions, leader_reference = action_postprocess(actions, leader_reference)
        if leader_reference is not None and sync_first_action and on_first_action is not None:
            on_first_action(ctx.env, actions)
            sync_first_action = False
        ctx.env.step(actions)
        step += 1
    # Falling out of the step cap counts as success — the user let the
    # episode run to completion. Matches soarm behaviour.
    return "completed"


def _add_teleop_callbacks(teleop_interface, events: TeleopEvents) -> None:
    add_callback = getattr(teleop_interface, "add_callback", None)
    if add_callback is None:
        return
    add_callback("R", events.request_reset)
    add_callback("N", events.request_success)
    # No in-viewer abort binding: P, Y, and ESCAPE were each tried and all
    # got swallowed by Kit's viewport before the leisaac callback fired.
    # The canonical abort path is `workflows/agentic/stop.sh arena --env <id>`
    # from a terminal — that SIGTERMs the process, the loop unwinds via
    # ``ready(ctx) == False``, and the AppLauncher closes the window.
