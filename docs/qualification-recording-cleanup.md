# Qualification and recording cleanup

This is a historical qualification record. Current Workflows setup installs stock Warp and does not require the local wheel or native hashes recorded below. The upstream captured-HashGrid defect is tracked in [Warp #2426](https://github.com/NVIDIA/warp/issues/2426); its explicit diagnostics remain separate from normal setup and CPU CI.

Status: **qualified for local progression with user-accepted test exceptions**,
following user review on 2026-11-23. The SIGTERM interruption and two failed
observations remain recorded; an uninterrupted all-passing suite is not established.

The approved cleanup is implemented on isolated local branches. Catheter crossing and
liver normal-direction mismatch remain accepted exceptions for local progression.
The earlier [qualification report](pr292-seven-qa-qualification.md) and its failed
stock-Warp results are preserved verbatim. The test-suite acceptance below is a
separate progression decision.

## Source and artifact boundary

- Physics: `5302e311332a830fe0eef9227a1fad63906aa89f`, branch
  `fix/qualification-test-cleanup`, based on approved `267f48a`.
- Workflows implementation: `4c310ad6` and native-video compatibility `8bbdeb0d`, branch
  `fix/qualification-recording-cleanup`, based on approved `2442dac8`.
- Workflows installs its declared `third_party/i4h-numerical-sim-5302e31` clone. Actual
  imports were checked against that location. Standalone qualification uses a separate
  `.venv-qualification`. Existing worktrees and unrelated GPU jobs were preserved.
- All 384 Arena lockfile package names/versions remain unchanged. Standalone Newton
  remains `7f0c13af6e4dd75253709c31dc65c0edcaf22b44`; Workflows Newton remains
  `d31603ff5e8389d1d6b2a5daef6ea2cb074c26ff`. Isaac Lab remains `3939445` and
  Arena remains `196680c`, with the maintained integration patches.
- Warp remains **1.17.0**. Approved wheel SHA256:
  `dff334c1589510f4374807eb3fbdc85cfa5e20ab8a0e05548dc54e0b64990546`.
  Actual loaded native-library SHA256:
  `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`.
  Both standalone and hosted qualification verified this native identity.
- The qualified artifact is limited to Linux x86-64 with glibc >= 2.38; this host has
  glibc 2.39 and an RTX PRO 6000 Blackwell. Portable wheel production, GB300 and fresh
  remote setup remain unqualified. The local physics revision has not been published.

Run `python3 numerical_sim/scripts/setup_qualification.py --warp-wheel /absolute/path/to/warp_lang-1.17.0-py3-none-linux_x86_64.whl`
from the physics repository root. See `numerical_sim/docs/standalone-qualification.md`.
Use that environment's Python directly after setup; a later dependency sync may replace
its patched wheel. Normal installation and CPU CI were not changed.

For the full local runtime/package run, from the physics repository root:

```bash
numerical_sim/.venv-qualification/bin/python numerical_sim/scripts/fetch_assets.py --check
I4H_RENDER_TESTS=1 I4H_DATA="$PWD/numerical_sim/data" \
  numerical_sim/.venv-qualification/bin/python -m pytest \
  numerical_sim/tests/runtime numerical_sim/tests/packages -q
```

## Implemented corrections

The physical-surface fixture now includes its required particles integrator. Worker
cadence measures exactly ten frames after readiness, retaining the greater-than-5:1
step/frame ratio and maximum 100 ms interval assertion. An event-controlled blocked
draw test proves physics continues, with bounded waits and worker-error propagation.

The attenuation reference now sums each segment's sampled contribution with its own
coefficient. Explicit overlapping-bend and differing-material cases retain existing
numerical tolerances. Production attenuation kernels are unchanged. Documentation
now states that bent flat-ended spans can overlap and count material more than once.

Native recording inspection accepts flat policy-observation datasets and grouped
camera observations without rewriting HDF5. It labels policy features separately
from physical particle state. Video switches precede the Arena environment subcommand;
other passthrough arguments retain their positions.

Actual video verification exposed another pinned-stack integration issue: Lab 3.0's
`render_mode="rgb_array"` returns no frames. The added Arena patch registers Lab's
native video recorder in its normal step/close lifecycle and enables Replicator before
Kit startup. The legacy Gym recorder fallback and camera-recording path remain available.
The patch is applied by normal third-party setup without changing dependency versions.

## Checks

| Check | Result |
| --- | --- |
| Repository/common CPU, after repin | 475 passed, 3 skipped |
| Arena CPU before the native-video patch | 1,156 passed, 4 skipped |
| Isolated RL CPU | 222 passed, 1 skipped |
| Recording tools, including grouped/legacy and flat inspection | 46 passed |
| Final focused attenuation/launcher/episode tests | 194 passed |
| Native-video compatibility and setup tests | 19 passed |
| Qualification artifact rejection tests | 6 passed |
| Worker readiness and blocked-render tests | 2 passed |
| Corrected physical-surface fixture, rendering enabled | 1 passed |
| Production asset manifest | 82 files verified |
| Applicable pre-commit hooks and Arena lock check | Passed |
| Full physics runtime/package suite, rendering enabled | SIGTERM after 906 results; continuation: 432 passed, 2 skipped |

The full physics suite was started once, with production assets and verified patched
Warp. It ended with **SIGTERM (exit 143)** after **898 passed, 2 failed and 6 skipped**,
before pytest could print its summary or failure traces. The termination source is
not established. All nine captured hash-grid tests passed before interruption.
The failed nodes, retained as failed observations, are:

- `tests/runtime/test_sequence_render.py::test_gl_sequence_omits_markers_and_keeps_legacy`
- `tests/runtime/test_two_worlds.py::test_contact_events_capped_and_reset`

A single focused diagnostic invocation of those two nodes passed (2 passed in 2.98 s).
This does not erase their failures in the larger run; order/state dependence remains
unresolved, and no failing assertion traceback survived the interruption. There was
no full-suite rerun. The remaining **434** collected nodes were executed once in two separate
bounded continuation batches: **100 passed**, then **332 passed, 2 skipped**. Across
the interrupted attempt and continuation, all **1,340 collected nodes** are covered:
**1,330 passed, 2 retained failed observations, 8 skipped**. This is coverage accounting
across processes, not a passing uninterrupted full-suite result. Focused worker measurements were not repeated to obtain
a passing sample. An uninterrupted passing supported-environment suite is not established.

## User acceptance of test qualification — 2026-11-23

After reviewing the two failed observations, their passing focused diagnostic,
the interrupted full-suite attempt and the completed continuation, the user instructed:

> ok qualify the test and lets move on

The physics test campaign is therefore **qualified for local progression with
explicitly accepted exceptions**: the two observed full-run failures and the missing
uninterrupted full-suite pass. This acceptance closes the cleanup batch for progression.
It does not mark the failed observations as passes, establish their root causes, change
assertions or thresholds, or remove the original SIGTERM and diagnostic evidence.
The existing catheter/liver exceptions and unavailable-hardware limits remain unchanged.
No next-stage implementation, publishing or tracker action is authorized by this closeout.

## Liver and recording evidence

Main probe evidence: `runs/liver_reach_retract/20261123_141200/`.
Final video capture: `runs/liver_reach_retract/20261123_142000/`.
Navigation and standalone suite evidence: `runs/endoluminal_navigation/20261123_141000/`.
These paths are relative to the isolated Workflows checkout.

The original recording at the preserved `pr292-seven-qa-sync` checkout's
`runs/liver_reach_retract/20261123_124141/verify.hdf5` now inspects successfully:
145 frames, actions `(145, 4)`, policy observations `(145, 57)`, success true, no
cameras. SHA256 before and after inspection is identical.

Strict snapshot reuse correctly rejected the old checkout's absolute asset paths.
Original snapshots and the rejection evidence were retained. Fresh snapshots passed
ordinary/masked reset, complete restore, fresh-process continuation and live-grasp
isolation checks in **1, 8 and 16 environments**. Complete round trips, peer preservation
and incompatible-bundle rejection passed. Maximum fresh-process particle error was
`7.60e-7 m`, below the unchanged `1e-5 m` bound; initial observations were exact.

| Environments | New strict snapshot hash |
| --- | --- |
| 1 | `203d618a7da714569d556c32459973d6cbd2d5746b8752239d5f3d594346857b` |
| 8 | `b2f9126a77abcd06a02326a676cb1f1de3256b62b9a483d0039cb3de99ae38b0` |
| 16 | `90a7e03a3466a0908777e1f410c5ac08c4d7212a8feb632ad51cbb5d1a68b60f` |

The normal visible launcher records **1/1 successful full-cycle liver episodes**, seed
7341, exit 0, with 145 actions of width 4 and 57 policy features. HDF5 inspection and
finite/shape/success checks pass. The viewport MP4 decodes fully: **145 frames,
1280 x 720, H.264, 30 fps, 4.833 seconds**. Extracted frames show the liver and grasper.
Fresh-process normal-launcher replay of episode 0 finishes successfully, exit 0.
The recording contains initial numerical state, but no per-step particle states;
replay completion is not a claim of exact per-step trajectory equality. The earlier
accepted normal-direction mismatch remains recorded in the original report.

The first video attempt's successful HDF5 and zero-frame Gym output remain in
`20261123_141200`. The native-recorder attempt before enabling Replicator failed with
`No module named 'omni.replicator'`; it is retained in `20261123_141900`. The final
`20261123_142000` capture uses the corrected native-recorder bootstrap.

Native static-aorta capture and fresh-process replay complete **180/180** actions of
width 3. Maximum relative observation error and final particle error are both **0.0**.
This is numerical recording qualification, not a human demonstration or task-success claim.

NVBug **6940614 remains deferred**; **7011011 remains separate**. No mechanics,
training, tracker, PR, publication or unrelated process cleanup was performed.
