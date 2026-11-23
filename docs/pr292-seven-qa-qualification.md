# PR 292 seven-fix source qualification

Status: **catheter and liver qualified for local progression with user-accepted exceptions**, following visual review on 2026-11-23. Original numerical failures and hardware limits remain recorded.

## Source boundary

Workflows starts at refreshed PR 292 head `e3cccfd2fb8888b8ee36358710516e4352324e16`.
Numerical Sim starts at refreshed internal main `caff73b12a77abd6157771678ee0a57206771ac4`
and is pinned at clean local commit `267f48a4356cd2bf4a6204501b9daa7a2ee5c15b`.

The physics branch `fix/seven-qa-sync` has four commits:

- `f14a39f`: containment geometry, insertion entry selection and capped-mesh continuation coverage.
- `bdefd71`: haptics test locking and pre-fire clipping references.
- `4d00ca7`: public-tag parity setup, missing-source diagnostics, output directory and recording key documentation.
- `267f48a`: GL camera conversion, both aorta tip views and explicit PyOpenGL dependency.

Public source commits are recorded in each commit message. The port retains the internal
Newton storage, manager, masked reset and continuation implementation. Caps belong to the
containment query mesh; the rendered vessel mesh stays authored. Both `cap_open_ends` and
`keep_open` remain available with geometry validation.

NVBug **6940614 is deferred entirely**. The centreline has 192 nodes, its reach and
576-value observation contract remain unchanged, and `guide_tip`, procedures and numerical
baselines are untouched. The only centreline YAML change is the approved tip-camera distance.

Workflows updates setup, all five numerical package sources, lock metadata, liver export
revision metadata, qualification builders and source-dependent tests together. All locked
package versions, the legacy solver pin, navigation backend default and Isaac Lab/Newton/Warp
pins remain unchanged. Existing worktrees, dirty liver optimizations and historical reports
and snapshots were preserved.

## Seven-bug disposition

| NVBug | Included public commits | Disposition |
| --- | --- | --- |
| 6940633 | `86a6f80`, `b06d3c4`, `6e88ec6` | Implemented. Existing rod suite and one-env 12/24 s reproduction pass. Extended eight-env 12 s check fails; 24 s returns inside the bound. Accepted by the user for progression after visual review; numerical failure retained. |
| 6940611 | `ce0e711` | 44 haptics/clipping tests pass in each of five runs. Three forced servo-race tests and two forced collapse tests also pass in each of five runs. |
| 7048521 | `6dd03a2`, `e5ffcf4` | Public `v0.9.0` parity reference, explicit public fetch and output directory documented. Fresh setup, missing-tag/source diagnostics and the complete documented parity path pass (20 bounded metrics, 12 informational rows). |
| 7048496 | `632095b` | Documentation states that a second R resumes recording with a new take from reset. Physical input verification remains separate. |
| 7048533 | `30d91c1` | Direction conversion tests and visible GL inspection pass; the screenshot shows yaw 90 degrees toward the aorta. |
| 7107434 | `0cd1e99` | Both visible tip views pass inspection and report exactly zero camera-to-tip error over the 90-frame probes. |
| 7107443 | `d6d962b` | Explicit GL-extra dependency and regenerated lock pass; the visible GL screenshot includes the restored UI. |

**7011011 remains separately tracked**: these test fixes do not assert a second-press mechanics fix.
Hardware-specific GB300 verification is unavailable on this RTX PRO 6000 host. Real haptics devices were not exercised.

## Evidence and acceptance

Raw logs, reproducer scripts, source identity and scope checks are retained under
`runs/endoluminal_navigation/20261123_123316/` in this worktree. Fresh liver artifacts are
kept under `runs/liver_reach_retract/20261123_124141/`. These are local run artifacts, not
replacements for historical qualification identities.

The isolated Arena environment installs this worktree and the clean local numerical clone
as editables. It reuses immutable simulator binaries from the existing elastic-rod environment
to avoid copying 33 GB on a disk with limited free space. `installed-source.json` records
actual imported paths and versions; `scope-and-pins.json` verifies unchanged package versions.
This establishes local qualification only. **Fresh remote setup is unavailable until the
physics commit is published**; no push, PR update, merge or NVBug change was performed.

Completed static and CPU checks:

| Check | Result |
| --- | --- |
| Repository/common CPU | 472 passed, 3 skipped. |
| Isolated RL CPU | 222 passed, 1 skipped after restoring required third-party paths. |
| Arena CPU | 1,143 passed, 14 skipped, 1 failed. The attenuation ray-march mismatch also fails at unchanged PR head `e3cccfd2`. |
| Physics runtime/package suite | 1,283 passed, 51 skipped, 9 failed in 1,250.64 seconds. |
| Physics focused rod/render/tip suite | 51 passed, 26 skipped before enabling rendering extras; capped continuation separately passed in eager and captured modes (2 passed). |
| Generated CPU synthetic example | Passed. |
| Six package builds and wheel/sdist boundary inspection | Passed. |
| Applicable source pre-commit checks in both repositories | Passed. |
| Workflows patched Warp binary identity | Passed; native library hash matches the required `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`. |

The physics suite used the locked standalone environment with its stock Warp 1.17.0 wheel.
Eight hash-grid capture cases returned wrong neighbor sets. The wall-clock test
`test_physics_rate_independent_of_render` observed 101 worker steps and 9 frames, below
its unchanged requirement of at least 10 frames. Its worker-throughput ratio did not fail.
These results are retained as failures; a later focused result does not erase them.
The Workflows environment uses its existing separately qualified patched Warp wheel.
The byte-identical Warp-only hash-grid test file from internal base `caff73b` reproduces
**8 failed, 1 passed** on the stock wheel; **9 passed** on the existing Workflows wheel.
This identifies the artifact boundary without changing either dependency pin or dropping
failed stock-wheel results.

## Fresh-checkout parity path

The documented public-tag fetch, reference installation, output-directory creation,
reference run, current-source run and comparison all finish with exit 0 from the fresh
parity checkout. **20 bounded metrics pass**, 12 rows are informational, and none fail.
Both engines execute all six scenarios at the original and refined substep settings.
The reference's first large Warp 1.13 module load took 312.6 seconds; that startup cost
is separate from a setup failure. Raw NPZs and `parity-report.json` remain with the run.
This utility/mechanics result does not waive native containment or replay failures.

## Runtime closeout

- Haptics/clipping: **44 passed in each of five runs**. The three affected servo tests also pass
  five times with a competing no-sample tick blocked by the lock. Forced first-fire complete
  collapse passes five times; the same probes require the old post-fire reference to fail.
- Rendering: **41 passed, 1 failed** with rendering enabled. The unchanged
  `test_physical_surface_without_appearance` fixture omits a particles integrator and is rejected
  by the existing loader. Visible GL/UI and both aorta tip views pass separately; their PNGs and
  zero-distance JSON measurements are retained in the navigation evidence directory.
- Containment: the original rod suite, including capped branch ends, open entry, masked reset
  and captured/eager agreement, passes. The added capped-mesh continuation tests pass in both modes.
  Rendered topology stays at 22,539 vertices and 44,688 triangles; seven query-only caps leave the entry open.
- One environment at full insertion speed passes at 12 and 24 seconds (maximum signed distances
  -0.01912184 and -0.01822048). The original eight-env 300-step sweep passes in the rod suite.
- The extended eight-env sweep at rates 1/8 through 8/8 fails at 12 seconds: environment 4's tip
  has signed distance -0.00913643, missing the unchanged -0.01 clearance bound; environment 6's
  tip has **positive signed distance 0.00278999** near the capped branch. A separate CPU
  geometry probe places that tip **0.00277127 m beyond the capped end plane** (ear-clipped cap),
  confirming the transient crossing rather than just a clearance-margin failure. At 24 seconds every row
  is inside the bound. The diagnostic continues solely to retain that previously unobserved
  endpoint; it preserves the 12-second failure and exits unsuccessfully. See
  `insertion-diagnostic.json` and the saved per-node NPZs. Zero escape is not established.
- Native static-aorta capture and fresh-process replay complete **180/180** actions of width 3.
  Maximum relative observation error and final particle error are both **0.0**. This is a scripted
  numerical-host recording, not a human teleop demonstration or a task-success claim.

## Liver qualification

Fresh source-bound snapshots, ordinary and snapshot reset isolation, fresh-process continuation,
and grasped-state isolation/restore pass in **1, 8 and 16 environments**. Grasped checks include
exact complete round trips, peer preservation, live acquired bindings and incompatible-bundle
rejection. Fresh-process initial observations are exact. Maximum particle transition errors are
4.33e-7, 4.77e-7 and 6.06e-7 m, below the unchanged 1e-5 m bound.

| Environments | Fresh settled snapshot identity |
| --- | --- |
| 1 | `ab9a174c4937afebadee6aa2966db3eb5b3d92dd5f4a957d58ba8e1e0c308f43` |
| 8 | `e46853618beb349975d1bc8b257fd94916c6dfd187d2b0f801270db97918111a` |
| 16 | `a4092abe998e514f4f3d30d2cd1bc61cb4abfaf3014c42ea1c3925fc5279b602` |

The normal visible launcher capture finishes **1/1 full-cycle episodes succeeded**, exit 0.
`verify.hdf5` contains `demo_0`, success true, **145 actions of width 4**, 57 observation
features, and complete numerical initial state. Direct HDF5 inspection confirms approach,
capture, qualify, retract, hold, release and relax phases. It contains no per-step camera or
particle-state dataset. The generic dataset inspector fails on its flat observation dataset
(`Dataset` has no `items`); native HDF5 inspection passes. The retired launcher `--attempts`
option was removed after preflight; each requested episode runs once on this branch.

Normal fresh-process launcher replay completes all 145 actions and passes success validation
(**1/1**, exit 0). A visible replay screen recording and extracted frame are retained.
The initial screenshot was taken before the first rendered scene; the observed replay confirms
that the liver is visible. An optional viewport-video CLI attempt is retained as unavailable
through the launcher's appended argument position; it did not execute an episode.

**Trajectory comparison fails separately from replay completion.** A fresh-process replay
starts with an exact initial observation and completes all actions, but at action 18 the
feature columns 3:6 differ by 0.00010587275 against the existing 0.0001 bound. Maximum encoded
feature discrepancy over the episode is 0.00554323. No tolerance was changed. Full per-step
particle-trajectory equality cannot be checked from this recording; the separate continuation
probes above do compare particles.

## User acceptance for progression — 2026-11-23

After inspecting the catheter and liver visual review, the user instructed:

> it looks stable, qualify both and lets move on

Both are therefore **qualified for local progression with explicitly accepted exceptions**:

- Catheter: the transient 12-second cap crossing is accepted for progression on the basis of
  the user's visual stability review. The original crossing measurement and failed assertion remain.
- Liver: successful full-cycle execution and replay completion are accepted for progression
  despite the small surface-normal comparison mismatch. The original failed bound remains.

The reviewed artifact is `runs/endoluminal_navigation/20261123_134710/index.html`.
It derives from the retained source-bound checkpoints and original failed comparison;
no simulation or acceptance threshold was changed for this decision.

## Retained failed checks and qualification scope

The extended eight-env containment failure, full-suite failures, render fixture failure,
Arena attenuation failure and liver encoded-trajectory mismatch remain recorded as failures.
The user's acceptance changes the progression decision for the reviewed catheter and liver
behaviors; it does not turn those test results into passes or certify a fully passing stack.
Other suite failures, fresh remote setup and GB300 qualification remain separately unresolved.

No all-passing PPO qualification manifest was generated; its source checks were repinned,
and failed gates were not marked true. No training campaign, numerical threshold change,
unrelated GPU-job cleanup, public release or tracker update was performed.
