# Numerical catheter migration qualification, 2026-11-25

The capability and migration implementations are published as drafts. **Default
promotion is blocked** by the frozen mechanics failures and production arm
continuation divergence. The retained chole and vascular checks are additional
release/compatibility blockers. Legacy remains the fresh-run default.

## Source and stack

- Physics [PR #19](https://github.com/isaac-for-healthcare/i4h-physics-simulation/pull/19)
  is stacked on #17 at `f2799ce998204495a4e629deb81fe3c59de114b3`.
- Workflows [PR #309](https://github.com/isaac-for-healthcare/i4h-workflows/pull/309)
  is stacked on #292 at `68307fca8a1d18b74ba5e15b5ab4a1ce4936bd4e` and depends
  on physics #19. Physics #8 and Workflows #252 remain unchanged source proposals.
- All six operational physics references select the immutable published
  `827dc426888b614d4e75d4cd309e66f1ae9ce753` candidate. Its package sources are
  identical to `d970d8707739b396184e7accc651bb96bb91da89`; the follow-up is tests only.
- Production native, mechanics and PPO runtime source:
  Workflows `c878bc41dfb5778c26252d17611c8b7bcffd4452`. Later qualification-driver
  fixes `06babf0ee3b72667d0d801fcb7c8acd4de6ae708` and
  `364449295090cefee009a6cb1d1e035c57add20b` change pass/fail accounting and fast-shutdown
  exit status only, not task or solver behavior.
  Each artifact records the actual imported physics revision, Workflows revision,
  tracked changes and driver hash. Subsequent commits contain reports only.
- Approved native host: Isaac Sim 6.2.0.0, Isaac Lab 3.0,
  Newton `d31603ff`, stock Warp 1.17.0, Torch 2.11, torchvision 0.26,
  MuJoCo/MuJoCo-Warp 3.17, Arena `196680c` with the five existing approved
  patches, RLinf `a4b6abe`. Physics uses its separate unchanged locked dev stack
  (Newton `7f0c13af6`, stock Warp 1.17.0, Torch 2.14).

Dependencies, locks, assets, success predicates, 600-control-step task budgets,
120 Hz physics / 30 Hz control, material conversion and numerical limits are
unchanged. No unrelated GPU jobs were stopped. The isolated environments reuse
approved host dependencies; original worktrees and dirty work are preserved.

## Patient provenance

The first four production attempts failed because the historical manifest pointed
to a missing anatomy bundle. Those failures remain in `native-patient-{1,16}.*`.
The original s0011 bundle was subsequently located in the main checkout's
`data/TotalSegmentator/s0011`. Its CT volume is bitwise equal to the prior fixture,
and patient ID, voxel affine and world affine match. A separate schema-2 manifest
maps the existing artifacts without editing the original manifest or dataset.
`production-patient/provenance.json` contains every artifact SHA and the equality
checks. All final production native, PPO, profile and recording runs use this
manifest. No different patient or generated anatomy was substituted.

Synthetic native runs (`patient=None`) at 1/16 worlds cover free and mounted host
lifecycle without production anatomy. They passed exact snapshot/observation and
trajectory continuation at physics `300573a4` / Workflows `6a67ea0f`. The frozen
synthetic mechanics fixtures separately include curved/branched and deformable
lumen cases. An additional analytic patient manifest exists in the evidence, but
is not presented as production qualification.

## Mechanics: 47/50, promotion failed

Frozen geometry SHA256:
`daaf6f03ba5ee7042f5c2cf8975248fa08ee36a5dfce8dd7fd277d06983f72a3`.
The fresh reference uses the unchanged public legacy source at
`50056196775fc0ed8cca03bf16144fe71d183ca7` and the current Workflow control
sources. The old reference is retained but rejected for mismatched source hashes.

`zero_response`, `arrival_positive` and `arrival_negative` each fail **force
relative L2 = 1.3897889852523804**, above the unchanged 0.1 bound. Both force
magnitudes are about 5.04 microN and the peak vector discrepancy is about
1.80 microN; no absolute force tolerance was introduced. Their moment absolute
error is `7.900025078555473e-11 N m`, below the previously authorized `1e-9` policy.
Loaded force/moment, trajectory, displacement and arrival gates pass. The policy's
absolute exception applies only to those unloaded moments. All original limits
and failed observations remain intact.

The mechanics comparator's `unqualified` list still names the wrong-branch
transition classifier, full MuJoCo-Warp coupling and native lifecycle. Separate
host observations below do not turn that list into a full task-success claim.

## Native production lifecycle and continuation

| Check | Free | Arm |
|---|---|---|
| Exact State and observation snapshot restore, E=1/16 | Pass | Pass |
| Configuration mismatch rejection | Pass | Pass |
| Selected reset/restore with peers unchanged | Pass | Pass |
| Rate/bend limits and terminal capture before reset | Pass | Pass |
| Hard reconstruction and rebinding | Pass | Pass |
| Subsequent same-process trajectory | Exact | **Fail** |
| Fresh-process snapshot and observation restore | Exact | Exact |
| Subsequent fresh-process trajectory | Exact | **Fail** |
| Visible Kit and simulated keyboard press/release | Pass | Pass |

Arm same-process maximum rod discrepancy is **0.13271260261535645 m at E=16**
and **0.07189977169036865 m at E=1** over four control steps; corresponding
maximum joint discrepancies are 0.0002943277359008789 and
0.00024259090423583984 rad. Initial fresh-process discrepancy is
0.06510709226131439 m / 0.00011649727821350098 rad. Exact snapshot equality
therefore does not establish continuation. The arm approach is not parked after
the six-step warmup; no full approach/navigation cycle or trained success is claimed.

The initial driver recorded these trajectory failures while incorrectly setting
`passed=true`. Those raw records are preserved;
`native-continuation-adjudication.json` applies the corrected logic to each raw
result without overwriting it. `06babf0e` makes rod or robot
trajectory mismatch fail qualification and moves nonzero exit into `finally`,
so the fresh-resume return also records failure. `36444929` passes that status
to `app.close(exit_code=...)`, since fast shutdown exits inside close. Seven focused
CPU regressions cover this accounting and explicit fast-shutdown status. Final fresh-process exit checks:

- free: passed=True, trajectory error=0.0 m, joint error=0 rad.
- arm: passed=False, trajectory error=0.08989790081977844 m, joint error=6.365776062011719e-05 rad.
- `continuation-final-fresh-1`: process exit 1, passed=False.
- `missing-production-final-exit-status`: process exit 1, passed=False.

Physical keyboard hardware was not operated. Keyboard tests send Kit press and
release events and verify zero output after release; input polling timing is
reported separately. Sensor and viewport PNGs show the production CT/scene.

## Finite PPO and normal recording/replay

Both production workflows completed six actual PPO optimizer updates at 16 worlds,
24 steps per update, with finite observations, rewards, losses and parameters.
Existing profile hyperparameters, native manager stepping and the 600-step task
budget were retained. Free value loss ranges from about 0.0024 to 0.0168; arm
value loss is about 4044 to 16982. This is finite-update validation, not convergence
or task-success validation.

- `endoluminal_navigation`: record-demo exit 0, replay-demo exit 0, replay-conflicting-backend exit 2.
- `endoluminal_navigation_arm`: record-demo exit 0, replay-demo exit 0, replay-conflicting-backend exit 2.

Both recordings contain one 192-sample demo and the complete navigation initial
state; the arm also records its articulation state. Both demo success flags are
false. Numerical recording metadata includes the backend, immutable physics revision,
continuation schema, patient manifest hash and native dt/decimation. Replay selects
the recorded backend with no override; explicit `legacy` conflicts must reject.
Headless arm replay also emits an ignored upstream `Se3Keyboard` destructor
warning, retained in its log; it completes the episode and exits 0.
Existing CPU compatibility tests retain the explicit-legacy requirement for old
recordings without backend metadata. Demo episodes are not scored by the task's
success predicate. Recording is still opt-in.

## Trajectory, rendering and input diagnostics

The profiler records 30 measured control steps after five warmup steps. The
`qualify_diagnostics.py` evidence wrapper uses the normal external-environment
entry point (including the UI startup event) and saves all rod positions outside
the measured loop into compressed NPZ. Its hash is recorded in each JSON.
E=1 is visible and verifies actual `omni.ui.Workspace` windows and registered
operator controls; E=16/64 are headless. The earlier headless E=1 measurements are
retained separately. Values are mean ms per phase invocation; there are four
physics ticks per control loop. Fluoroscopy and viewport rendering are distinct.
Input at E=1 is keyboard polling; other inputs are a scripted tensor clone.

| Workflow | E | Physics tick ms | Fluoroscopy ms | Viewport ms | Input ms | Control loop ms | Trajectory finite |
|---|---:|---:|---:|---:|---:|---:|---|
| free | 1 | 41.991 | 14.229 | 13.610 | 0.052 | 220.589 | True |
| free | 16 | 51.702 | 532.881 | 0.000 | 0.007 | 750.369 | True |
| free | 64 | 67.691 | 2136.673 | 0.000 | 0.008 | 2420.933 | True |
| arm | 1 | 38.516 | 13.886 | 14.271 | 0.051 | 215.018 | True |
| arm | 16 | 46.755 | 524.804 | 0.000 | 0.007 | 731.139 | True |
| arm | 64 | 80.734 | 2209.257 | 0.000 | 0.008 | 2562.509 | True |

These synchronized diagnostics ran alongside existing user GPU jobs. They are
contended measurements, impose no new gate and do not establish performance
parity or sustained real-time rates. Shared-host/rendering time is not attributed
to the physics solver without a matched isolation experiment.

## CPU, packaging and compatibility

| Check | Result |
|---|---|
| Workflows root/common/Arena CPU | 1742 passed, 69 skipped |
| RL CPU with the approved RLinf source present | 217 passed, 2 skipped |
| Discovery, integration and native-video checks | 145 passed |
| Six-reference setup/pin checks | 16 passed |
| Qualification result regression | 7 passed |
| Physics full runtime/packages | 1207 passed, 23 skipped, **2 failed** |
| Physics final compatibility/parity group | 78 passed |
| Physics registry/body/constructor correction | 20 passed |
| Builds / artifacts / private reaches / assets | Six packages, 12 artifacts, import boundaries, 87 asset hashes passed |
| Both repositories' pre-commit, host-stack verification, workflow list/show | Passed |

The physics full run at d970d870 loaded the old registry expectation and failed
because the new live `gravity_scale` was absent from its expected set. The
827dc426 test-only correction and focused group pass. The broad suite was not
repeated after that correction and is not called green. Its other failure is
`test_chole_batched_matches_single`: `others=0.0030141472816467285` fails the
unchanged strict `>0.005` phase-separation check. Original/interrupted failures
and source-mismatch attempts are retained. Stock-Warp HashGrid opt-in failures
remain skipped; no patched Warp was installed.

The existing vascular checkout is dirty and was preserved. Its Newton 1.6.1
environment cannot collect the candidate's `ObservableKind` integration; other
tests import `i4h_isaaclab.composition`, already retired by parent #17. A separate
candidate-environment CPU subset produced 5 passed, 4 GPU skips and 1 retired
composition import failure. Full vascular regression remains blocked. No alias,
case change or dependency upgrade was used to conceal that incompatibility.
`run.sh lint` is not a supported command; repository pre-commit and discovery
checks supply the applicable lint/discovery validation.

## Evidence and disposition

Raw free-navigation evidence and reproducible drivers:

`/home/pkorzeniowsk/Projects/i4h/i4h-workflows-worktrees/numerical-catheter-runtime/runs/endoluminal_navigation/20261125_134900`

Arm evidence:

`/home/pkorzeniowsk/Projects/i4h/i4h-workflows-worktrees/numerical-catheter-runtime/runs/endoluminal_navigation_arm/20261125_134900`

Physics evidence:

`/home/pkorzeniowsk/Projects/i4h/qualification-evidence/catheter-runtime-20261125/physics`

Normal recording/replay directories are recorded in `native-final-queue.json`.
`navigation-report.json`, `production-patient/provenance.json`, PPO JSON, native
NPZ/JSON/PNGs, diagnostic traces and process logs are retained locally, not
attached to the PR. Historic failures were not overwritten by successful results.

No default-switch PR is opened. No PR is merged. This implementation and these
draft publications do not waive the mechanics, arm continuation, chole,
vascular or unrelated release blockers. Legacy package removal and converged
policy training remain outside scope.
