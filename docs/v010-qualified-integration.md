# v0.10 Workflows integration checkpoint

PR [292](https://github.com/isaac-for-healthcare/i4h-workflows/pull/292) remains the
primary Workflows v0.10 integration PR. This checkpoint combines its published
`e3cccfd2` source with the committed arm/display optimization from `ee953f7b` and
the qualified QA/recording branch through `1c0f2f26`.

The combined implementation checkpoint is `3fdef720`. The original dirty
`elastic-rod` worktree was preserved: its uncommitted liver-controller transfer
optimization, tests and shader cache are not included. Qualification worktrees and
historical evidence remain unchanged.

## Dependencies and review boundary

- Physics remains pinned to the qualified production implementation
  `5302e311332a830fe0eef9227a1fad63906aa89f`. Setup paths, lock metadata, exported-policy
  revision metadata and source-dependent checks agree.
- The pin is now reachable remotely through
  [physics PR 21](https://github.com/isaac-for-healthcare/i4h-physics-simulation-internal/pull/21).
  An isolated filtered fetch of the exact SHA succeeded. Later physics PR commits
  add only the 7011011 test fix and review documentation, so the runtime pin is retained.
- All package versions and the Isaac Lab/Arena/Newton/Warp pins remain unchanged.
  Loaded Warp 1.17.0 native SHA256 is
  `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`.
- Immutable simulator dependencies are reused by private lightweight environments;
  actual Workflows and Numerical Sim source imports point into this integration worktree.
- [Cosmos PR 7](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams/pull/7) already
  covers the local output-path/NumPy-format fixes and other QA items. No duplicate
  Cosmos PR or unreviewed submodule bump is included here.

## Fresh combined-source checks

| Check | Result |
| --- | --- |
| Repository/common | 475 passed, 3 skipped |
| Arena | 1,303 passed, 4 skipped |
| Isolated RL | 222 passed, 1 skipped |
| Recording tools | 46 passed |
| Arm/display focused regression selection | 200 passed |
| Pre-commit checks across the integration diff | Passed |
| Arena lock check | Passed |
| Hosted 1/8/16 liver reset/restore | Assertions passed; 1-env grasped probe hung at shutdown (240 s timeout); 8/16 exited normally |
| Visible liver cycle and viewport video | 1/1 full-cycle success; valid 146-frame 1280x720 H.264 video |
| Fresh-process liver replay with success validation | **Failed: 0/1**, despite completing all 146 actions |
| Native static-aorta capture/replay | 180 actions; replay observation and final particle errors both zero |

Evidence is under `runs/liver_reach_retract/20261123_182635/` and
`runs/endoluminal_navigation_arm/20261123_182635/` in the isolated integration worktree.
The visible liver recording and replay evidence is under
`runs/liver_reach_retract/20261123_183500/`.
New snapshots are generated for this checkout's strict source/asset identity; old
snapshot identities and prior qualification reports are not rewritten.

## New failures that block release

The fresh visible liver recording succeeded, but explicit replay success validation
returned **0/1**. A separate headless diagnostic restored the initial observation
exactly, then exceeded the existing target-offset bound at action 11
(0.1154 mm versus 0.1 mm). Its objective later failed at physics step 231 because
fewer than half of the originally acquired tissue nodes remained retained
(`retention_loss`, failure code 5). All 146 recorded actions were consumed.
The recording has no per-step particle states, so the first solver-level divergence
is not established. The cause is unresolved; replay completion and process exit 0
are not success qualification. No threshold, controller or physics change was made
to suppress this finding. This failure is additional to the historical accepted
encoded-normal mismatch and is not covered by that acceptance.

The 1-environment grasped isolation probe wrote a passing assertions report, then
hung during Kit shutdown and reached its 240-second timeout. Its orphaned child was
identified by this checkout's exact command and cleaned up. The 8/16-environment
probes exited normally. Assertion correctness and process lifecycle are reported
separately; the shutdown issue remains unresolved.

## Performance gate and accepted exceptions

The committed `ee953f7b` implementation retains its original measured performance
qualification and regression fixtures. Those measurements used the earlier dependency
identity; they are not fresh measurements of this combined source.

A fresh performance preflight observed an unrelated `cath-gdwr-validation` GPU worker
at 99% SM utilization. No timing trials were launched, no failed sample was replaced,
and the unrelated jobs were left running. A fresh matched, warmed performance campaign
on the final dependency pair remains a release gate. Throughput or FPS improvement
for this combined checkpoint is not claimed.

The earlier user-accepted transient catheter cap crossing, liver encoded-normal
trajectory mismatch and interrupted physics-suite observations remain accepted only
for local progression, with their original failed measurements preserved. See
[the original QA report](pr292-seven-qa-qualification.md) and
[the cleanup disposition](qualification-recording-cleanup.md).
GB300/aarch64 and real-device checks remain outstanding. Physical keyboard, leader
and XR devices were not validated by scripted controls or regression tests.

The main migration's earlier stopped-PPO result remains unchanged; a successful full
training campaign is not claimed. Fresh public setup remains a separate release gate:
current Numerical Sim source is in the private internal repository, and portable
patched-Warp distribution plus source-bound assets/checkpoints require their own handoff.
No merge, public numerical publication or NVBugs update is part of this checkpoint.
