# PR 292: repin to committed physics main

## Source boundary

On 2026-11-23, physics local and remote `main` both resolve to
`4f49fcf34e038b848355b60f1d14ff89772249a4`. Workflows starts at PR 292 head
`335240ca5ded6133927c9e4933b58b812bb8331c` and updates the setup revision/directory,
five package sources, Arena lock metadata, exported liver policy revision metadata,
qualification builder and source-dependent tests together.

The dependency was checked out separately under
`third_party/i4h-numerical-sim-4f49fcf`; the exact SHA was fetched from the internal
remote. Actual module imports were verified there. Missing local asset files reuse
read-only symlinks into the production asset tree; tracked new dVRK assets come from
the new checkout. All 87 production manifest entries verify. Both source worktrees
and unrelated GPU jobs were preserved.

All 384 locked package names and versions are unchanged. Isaac Lab, Arena, Newton,
Isaac Sim and Warp pins are unchanged. Loaded Warp 1.17.0 native SHA256 remains
`eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`.
This verifies the reused local environment, rather than a fresh public installation.

## Important upstream differences

This pin uses physics main exactly as committed. Relative to `5302e31`, it adds dVRK
USD instruments, licensing/metadata, changes several chole instrument configurations
and adds the standalone panel factory option. It also removes the containment-cap
implementation/coverage, qualification-wheel setup and some earlier haptics,
camera/UI and test changes. No removed patch is silently reapplied by this repin.
Earlier seven-fix qualification does not establish those removed behaviors on this
source. In particular, capped containment is not a property of this checkpoint.

The hosted `liver_pinned_grasp_1x` fixture is unchanged and still references
`instruments/pgrasp.usdc`; the new dVRK instrument configurations do not replace that
fixture's grasper. Fresh strict snapshots were generated for the new source identity.
Old snapshots, reports and failed measurements were retained unchanged.

## Fresh checks

| Check | Result |
| --- | --- |
| Repository/common | 475 passed, 3 skipped |
| Arena CPU | 1,303 passed, 4 skipped |
| Isolated RL CPU | 222 passed, 1 skipped |
| Dataset tools | 46 passed |
| Arena lock check and affected-source pre-commit | Passed |
| Hosted liver reset/restore and fresh continuation, 1/8/16 environments | All 15 probe processes passed and exited normally; live-grasp isolation passed |
| Visible native liver capture with viewport video | Kit startup crash, exit 139; no recording produced |
| Visible native liver capture without video, diagnostic configuration | 1/1 full-cycle success, exit 0; 145 finite four-channel actions, 145x57 observations, complete numerical initial state |
| Fresh-process native liver replay with explicit success validation | **0/1 successful**, failed demo 0; all actions consumed, process exit 0 |
| Native static-aorta capture/replay | 180 actions, width 3; zero relative observation and final particle error, both processes exit 0 |
| Focused upstream rod/haptics/clipping/render/worker selection, using the hosted dependency stack | **118 passed, 2 failed**, exit 1 |

Raw evidence is in the isolated integration worktree under
`runs/liver_reach_retract/20261123_223953/` and
`runs/endoluminal_navigation_arm/20261123_223953/`.
The first video-enabled launch crashed in Kit before environment construction.
Its log includes a native crash backtrace ending in Kit/Carb startup/shutdown code;
the underlying cause is not established. The separate no-video diagnostic retains
the same seed, policy and episode budget and succeeds, but does not qualify video.
No repeated video attempt was used to replace the failure.

The first aorta harness launch omitted `I4H_DATA` and failed before scene loading.
After providing the explicit production asset directory, capture and replay both
passed. The original configuration-error log is retained. This is an unscored
numerical diagnostic, rather than a task-success or camera-direction qualification.

The focused upstream failures are:

- `test_clipping.py::test_tracker_and_clip`: after the second press, clip count
  remains 1 rather than the expected 2. This source still uses the old second-press
  test sequence; the separate 7011011 correction that repositions onto a live,
  eligible node is not part of this physics main. The failing test is not evidence
  of a new second-press mechanics diagnosis. Its endpoint NPZ is retained.
- `test_render.py::test_physical_surface_without_appearance`: its fixture declares
  tet and surface systems without a particles integrator. The existing loader
  rejects that invalid fixture before rendering, exactly as its error states.
  The earlier particles-fixture cleanup is absent from the new main.

No failed upstream test was rerun or edited to obtain a pass. This selection uses
the patched Workflows Warp/Newton stack; it is not the full standalone qualification
campaign or a claim about the stock standalone wheel.

## Disposition

The new pin is tested with the results above; it is not release-qualified. Liver
replay success is still failing. The earlier shutdown hang did not recur in the
one-environment probes, but one passing invocation does not establish its cause or
a fix. No success threshold, numerical tolerance, controller or episode budget was
changed to make the checks pass.

Final-pair performance was not requalified. Earlier accepted transient catheter
crossing, encoded-normal mismatch and interrupted-suite observations remain tied
to their original source/report boundaries; they do not waive new failures or
qualify upstream behaviors removed by the new main. A full standalone physics suite,
full training campaign, real devices, GB300/aarch64 and fresh public setup were not
repeated by this Workflows repin. Public source availability and portable patched-Warp
delivery remain separate release work. No merge or public numerical publication is
part of this update.
