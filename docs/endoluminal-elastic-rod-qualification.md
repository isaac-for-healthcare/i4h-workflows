# Endoluminal elastic-rod migration qualification

Both navigation workflows now have an opt-in Numerical Sim backend, native
continuation recording and backend-aware replay. **Qualification and promotion
remain incomplete. Legacy remains the fresh-run default.** Dependency declarations
and locks retain the approved baseline until the qualification gates close.

See [native integration status](#native-integration-status) for the current
implementation, evidence and failures. The mechanics campaign below is retained
as the historical acceptance record.

## Historical mechanics source boundaries

The earlier mechanics campaign used separate `migration/elastic-rod` worktrees based on Workflows
`dfa5b83f`, Numerical Sim `caff73b1` and legacy physics `5005619`.

| Change | Local commit |
| --- | --- |
| Current-pin parity harness and frozen fixtures | `4676b89` in Numerical Sim |
| Opt-in mechanics candidate | `b07bba59a59eb5a1347583806c73e834a1a53ac6` in Numerical Sim |
| Workflow qualification driver | `88303728a57a7d45992c3881545641f0aae73092` in Workflows |

Both historical mechanics navigation archives identify clean source checkouts, the same harness
hash, NumPy 2.4.6, Torch 2.14.0 and the verified patched Warp 1.17 build.

The original Workflows checkout and existing worktrees are preserved. The numerical baseline
is a separate pristine `caff73b1` checkout; it does not import the mechanics
candidate's packages.

The upstream candidate adds transported node frames for curved initial placement,
absolute proximal-pose and bend commands in Newton Control, interior pull that
neither moves the wall nor reports contact, and opt-in wall prediction and colored
projection matching the reference. Existing mechanics remain the defaults.
There is no legacy solver underneath the numerical candidate.

Source changes in this checkout add `scripts/qualify_catheter_migration.py` and
its tests; this report documents the results. The utility freezes geometry, calls the existing route/tangent drive code and
uses the actual Workflow arrival predicate at 30 Hz. It records all physics ticks
at 120 Hz and retains separate attempted runs. It does not provide a runtime
backend switch or modify the workflow declarations.

At mechanics closeout, native adapters, backend selection and portable
continuation were still outstanding. The native integration below implements
those features. The dependency repin and default promotion remain gated.

## Fixed comparison contract

The complete original six-scenario rod campaign and its half-substep refinement
passed all 20 bounded rows at the approved source pins. The scenarios cover sag,
insertion, tip bend, twist and insertion into straight and bent rigid tubes.
The published bounds have not changed. These eager baseline runs used stock
Warp 1.17 and are not final native-host qualification.

Additional fixture inputs and thresholds were frozen before replacement mechanics:
curved initialization, route insertion, tangent feed on a prescribed flange,
deformable wall, interior pull, prescribed-flange reaction, zero input and positive
and negative arrival. The numerical candidate continues to use physical rotational
inertia and polyline tip mapping.

The new fixture contract requires tip RMS error at most 2% of rod length, force,
moment and nonzero wall-displacement relative errors at most 10%, no additional
wall escape or wrong-branch frames, and matching arrival decisions. Force and
moment are checked separately. The comparer rejects post-freeze settings or
threshold changes, mismatched commands or sample times, and differing Workflow
control-source hashes. Stateful tangent root targets may differ between engines.
Escape comparison preserves particle identity;
equal total escape counts cannot hide a newly escaped particle.

## Current acceptance result: revision 2

After the user authorized loosening the thresholds, the revised comparison passes
**50 of 50 mechanics checks**. Native workflow integration and promotion remain
outstanding.

The versioned policy is `scripts/catheter-migration-acceptance.json`. Only
`zero_response`, `arrival_positive` and `arrival_negative` use an absolute moment
limit of **1e-9 N m**. Both engines' peak moment magnitudes and their peak vector
difference must satisfy that limit. The largest observed value is
`1.1337971533853519e-10 N m`. The policy rejects loaded, moving or curved fixtures;
loaded force/moment limits remain 10%, and every other gate remains unchanged.

This is a reanalysis of the original immutable traces, with no physics rerun or
mechanics parameter changes. The new result and policy copy are retained under:

```text
runs/endoluminal_navigation/20261120_150900/
```

The physical fixture config and its frozen hash remain unchanged. The policy is
bound to that hash and is included in the new comparison report. The original
failed report below remains available in its original run directory.

## Original acceptance result: revision 1

The original strict comparison produced **47 passing rows and 3 failing rows**.
It is preserved as historical evidence; revision 2 supersedes its zero-load moment criterion.

| Gate | Measured result | Bound | Result |
| --- | --- | --- | --- |
| Maximum fixture tip RMS / length | 0.01095% | 2% | Pass |
| Deformable-wall displacement relative L2 | 0% | 10% | Pass |
| Interior-pull wall displacement relative L2 | 0% | 10% | Pass |
| Additional wall escapes or wrong-branch frames | 0 | 0 | Pass |
| Positive and negative arrival, both engines | Correct | Correct | Pass |
| Unloaded root moment relative L2, three fixtures | 200.02% | 10% | Fail |

The failed rows are `zero_response`, `arrival_positive` and `arrival_negative`.
The unloaded moment reverses sign: reference RMS is `5.6683845e-11 N m`, numerical
RMS is `5.669586e-11 N m`, and RMS difference is `1.133797e-10 N m`. The original
relative bound had no numerical-noise exemption. That run retained the failures.
The later user-authorized policy revision changes only the zero-load moment test;
material settings, iteration counts and response parameters remain unchanged.

The original Jacobi wall path failed displacement parity. Matching the reference
required colored projection and rest-strain construction from double-precision
frames before device upload. With those opt-in behaviors, both complete wall
trajectories match exactly. Earlier failed archives are retained.

The prescribed-flange fixture checks loads through commanded kinematics; it does
not establish two-way integration with an actual arm.

## Evidence

The canonical local run directory is:

```text
runs/endoluminal_navigation/20261120_135300/
```

It contains the approved plan, frozen inputs, complete source-pin archives,
per-tick navigation archives, comparison JSON, source snapshots, environment and
package hashes, pre-commit output, CPU coverage output, the NV-BASE report log and
eval schema results. Earlier failures and interrupted attempts are retained.
The historical upstream parity report remains unchanged.

At mechanics closeout, the native-host gates had not run: patient-backed and synthetic workflows,
headless and visible fluoroscopy, keyboard teleoperation, recorded demonstrations,
backend-aware initial-state restoration and replay, automatic conversion, native
terminal/reset ordering and full MuJoCo-Warp arm integration. Separate physics,
rendering and control-loop performance qualification also remains outstanding.
A prescribed-flange fixture does not establish arm integration or trained-policy
performance transfer.

## Validation results

| Check | Result |
| --- | --- |
| Numerical Sim full runtime/package suite, final source | 1,297 passed, 52 skipped, no failures |
| Workflows CPU/coverage command | 603 passed, 3 skipped; unavailable RL suite reported separately |
| Workflows qualification tests with Torch installed | 6 passed |
| Command integration, selected resets and continuation checks | 28 passed |
| Changed-file pinned pre-commit checks | Passed in both repositories |
| Workspace wheels and source distributions | All 12 built and inspected |
| Built wheel Python sources versus candidate | All 212 matched |
| Generated core CPU example | Passed |
| Installed numerical dependency consistency | Passed |
| NV-BASE local scan | Passed; 0 errors, 16 warnings |
| Skill eval schemas | All 14 passed |

The Workflows CPU invocation reported 39.4% aggregate coverage; it does not include
native simulator coverage. Its three individual skips lack Torch; the additional
qualification invocation above exercises the arrival adapter with Torch installed.
The full numerical suite retains unavailable Isaac Lab/XR and disabled rendering
skips. The NV-BASE findings include robot-coordinate tuples misclassified as GPS;
no skill source or eval content changed.

An earlier numerical full-suite attempt failed an unconditional `slangpy` import.
Installing the pinned renderer dependency fixed its focused check, and the final
full-suite rerun passed. The earlier failure and the stock-Warp hash-grid capture
failures remain in their original logs. No assertion or tolerance was weakened.

Other GPU jobs were left running. Test durations and trace-collection elapsed times
are not separate measurements of physics, rendering or control-loop performance.

## Reproducing the numerical test environment

From the Numerical Sim workspace, use the pinned dependencies and renderer extra.
Then install the operator-supplied patched Warp wheel and preserve it for testing:

```bash
uv sync --locked --group dev --extra render
uv pip install --python .venv/bin/python --no-deps "$I4H_WARP_WHEEL"
uv pip install --python .parity/reference/venv/bin/python --no-deps "$I4H_WARP_WHEEL"
UV_NO_SYNC=1 I4H_DATA="$PRODUCTION_DATA" uv run pytest tests/runtime tests/packages -q -ra
```

The renderer extra supplies `slangpy==0.43.1`, needed by an unconditional XR unit
test import. Tests without Isaac Lab, XR hardware dependencies or enabled rendering
remain explicitly skipped; fake XR checks do not qualify physical hardware.

| Artifact | SHA-256 |
| --- | --- |
| Patched Warp wheel | `dff334c1589510f4374807eb3fbdc85cfa5e20ab8a0e05548dc54e0b64990546` |
| Loaded Warp native library | `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb` |

Use a fresh canonical run directory for another qualification attempt, or preserve
its existing archives and logs first. The utility checkpoints completed scenarios
into the named output archive.

## Native integration status

Work continues in the existing `migration/elastic-rod` worktrees. The approved
starting revisions were Workflows `606e19df` and Numerical Sim `b07bba59`.
Upstream support is committed separately as `a426c2c` (portable world rows and
curved proximal guidance). Workflow integration starts at `f3a54cdb`, with
subsequent focused continuation and provenance corrections retained as local
commits. Nothing was pushed or published.

### Implemented behavior

- Both navigation workflows accept `--catheter-backend numerical|legacy` through
  every declared and built-in run mode. Fresh runs still select legacy.
- Numerical scenes use Arena's custom environment entry point, `I4hHostMixin`
  and the merged component. Physics is 120 Hz with decimation 4 and control at
  30 Hz. Existing action terms integrate feed, twist, bend and C-arm rates once
  per physics tick; numerical controls receive absolute pose/shape commands.
- The patient supplies the same route, anatomy transforms and deformable vessel.
  Synthetic operation remains available without patient data. The arm retains
  its approach, IK, grip, tangent feed, transport and lumen clamp, with supported
  two-way MuJoCo-Warp mount coupling in the shared Newton model.
- Native HDF5 metadata and `run.json` carry backend, numerical revision and
  continuation schema. Replay selects the recorded backend and patient, rejects
  an explicit conflicting backend, and requires explicit legacy for older
  metadata-less recordings. Conflicting capture provenance is rejected before
  publishing a sidecar.
- `initial_state/navigation` contains independent episode rows: declared
  numerical State/Control, particle histories, pending mount reactions, drive
  integrals, arm/MuJoCo continuation, C-arm state, task/reward/progress history
  and fluoroscopy caches, including per-term episodic reward accounting.
  Restoration validates identity, layouts and dtypes,
  restores host assets and flushes host caches before numerical history, and
  runs no history initialization afterward.
- Raw columns remain four actions and four catheter/C-arm state columns, plus
  seven arm joints for the arm workflow. Patient conversion retains the five
  derived goal columns and existing modality layout. Derived datasets also
  retain source backend/patient provenance in `meta/source_recordings.json`.

### Qualification evidence and gates

The canonical native evidence directories are:

```text
runs/endoluminal_navigation/20261120_152900/
runs/endoluminal_navigation_arm/20261120_155000/
runs/endoluminal_navigation/20261120_160200/
```

Every attempted run retains its own named output. These are staged results,
not a claim that every final-source qualification gate passed. New native driver outputs
include source revisions and driver hashes; snapshot identity checks additionally
bind numerical and owning Workflow source contents and anatomy.

- The isolated Arena environment was provisioned with existing pins and the
  verified patched Warp wheel. Numerical and legacy native coexistence smokes
  passed in opposite import orders. All five numerical packages in the candidate
  environment now resolve to the local Numerical Sim worktree.
- The final candidate's mechanics traces were collected on the pinned Arena
  stack and compared against the frozen reference: **50/50 measured gates pass**.
  Revision 2 remains unchanged; the 1e-9 N m exemption applies only to its three
  eligible unloaded fixtures. The trace-producing process returned signal status
  143 after writing its complete archive; the independent comparison completed.
- Synthetic numerical lifecycle checks reached 1/8/16 environments for both
  workflows. State round trips and selected reset/restore checks passed.
  An eight-environment hosted episode also restored into a fresh single
  environment with exact numerical state and rod trajectory agreement.
- Patient catheter checks reached 1/8/16 environments with exact saved-state
  round trips and immediate peer isolation. Resumed rod trajectory differences
  were measured, without assigning a new acceptance threshold: up to
  2.27e-6 m in the staged 16-environment check.
- A visible patient arm check exercised 180 control steps before restoration.
  Saved-state and reconstruction checks passed in subsequent attempts. Resumed
  rod and arm trajectories can differ at floating-point scale; exact state
  restoration is distinct from deterministic trajectory continuation.
- Native 24-step HDF5 capture and standard-launcher replay passed at the staged
  source. The first recording attempt exposed dropped dynamic recorder fields;
  declared recorder configurations fixed that issue. Final-source capture and replay evidence is retained separately below.
- The complete CPU/coverage command passed 614 tests with 3 individual skips.
  Its RL suite was reported unavailable because its component checkout is absent.
  The broad Arena run reported 1,061 passed, 5 skipped and one failure below;
  subsequent focused continuation/provenance tests are retained separately.
- Required NV-BASE and eval-schema validation passed: zero errors, 16 understood
  warnings, installed security scanner, and all 14 skill datasets accepted.
  Pinned pre-commit checks passed for all changed files.

### Final-source lifecycle checks

The `closeout-*.json` artifacts qualify the stricter source and live fluoroscopy
configuration identity (`49f963bb`), with the same Numerical Sim `a426c2c` physics.
All **12 lifecycle runs** passed: both workflows, synthetic and patient scenes,
and 1/8/16 environments. Each checks exact saved-state and observation restoration,
selected reset/restore with unchanged peer rows, terminal results before automatic
reset, rate/bend limits and rejection of changed fluoroscopy configuration before
mutation. All four 8-environment cases also passed hard reconstruction.

All **four fresh-process checks** passed exact state and first observations after
exporting a selected world from eight environments into one. The table records
maximum differences over four resumed control steps, not acceptance tolerances.
The arm comparison includes all native articulation joint columns.

| Workflow / scene | Same-process rod (m) | Same-process arm (rad) | Fresh rod (m) | Fresh arm (rad) |
| --- | ---: | ---: | ---: | ---: |
| Catheter / synthetic | 0 | n/a | 0 | n/a |
| Catheter / patient | 2.09e-6 | n/a | 7.75e-7 | n/a |
| Arm / synthetic | 0 | 1.81e-5 | 0 | 0 |
| Arm / patient | 5.43e-6 | 3.36e-5 | 4.77e-7 | 7.75e-6 |

These measurements establish exact serialization of the covered state. Resumed
trajectories differ; the cause of those differences is not established, and exact
trajectory equivalence remains unqualified. No numerical bound was invented to
turn those differences into a mechanics pass. The frozen 50-row mechanics
comparison remains the separate physical acceptance evidence.

Both visible patient checks used an active Kit visualizer, saved viewport and
real DRR images, and exercised press/release on all four keyboard channels through
Kit's event fallback. Physical keyboard hardware and an operator-driven teleop
recording were not tested. `lifecycle-closeout-summary.json` contains the aggregate.

### Recording, replay and conversion closeout

All eight workflow/backend/scene combinations below completed capture and standard
launcher replay. Synthetic scenes used the full 192-control-step sweep, visibly;
patient recordings used a 48-step headless capture smoke. These demos are
unscored and are not successful target-navigation demonstrations.

| Workflow | Backend | Scene | Capture run | Replay | Conversion |
| --- | --- | --- | --- | --- | --- |
| Catheter | legacy | patient | `20261120_165244` | pass | pass |
| Catheter | legacy | synthetic | `20261120_165435` | pass | not applicable |
| Arm | legacy | patient | `20261120_170113` | pass | pass |
| Arm | legacy | synthetic | `20261120_170351` | pass | not applicable |
| Catheter | numerical | patient | `20261120_171353` | pass | pass |
| Catheter | numerical | synthetic | `20261120_171450` | pass | not applicable |
| Arm | numerical | patient | `20261120_171540` | pass | pass |
| Arm | numerical | synthetic | `20261120_171642` | pass | not applicable |

`capture-matrix-qualified.json` lists full paths, commands and per-input exit
statuses. `recording-conversion-audit-final.json` verifies all eight native
recordings and four converted patient datasets. Raw actions remain four columns;
raw states remain four or eleven columns, and converted states are nine or
sixteen columns. Every converted raw state/action value matches its source,
including the arm joints. Each dataset has 48 H.264 frames at declared 30 Hz,
valid modality/stats metadata and backend/patient/source-episode provenance.
No anatomy override was supplied to conversion.

Both numerical workflows also recorded **eight independent 48-step episodes**
from eight environments, then replayed episode 7 through the normal launcher in
one environment. Every continuation dataset has one independent episode row.
`batched-capture-matrix-rerun.json` and `batched-recording-audit.json` retain the
commands and audit. The first two batch launches placed the runner's global
`--num_envs` argument after its scene subcommand; those argument-order failures
are retained. The corrected qualification harness places it before the scene;
no runtime or acceptance bound was changed for that correction.

An additional visible patient arm run warmed up for **180 control steps**. The
approach reported parked, and exact state/observation restoration, selected reset,
limits, terminal-before-reset and hard reconstruction all passed. The measured
four-step resumed rod difference was 2.57e-6 m and arm difference 7.75e-6 rad.
Viewport and DRR images are retained as `closeout-patient-late.*` in the arm
lifecycle evidence directory. This verifies the later controller phase without
claiming a trained-policy or target-arrival success result.

### Matched phase timings

Each final profile uses the same `s0011` scene, seed 42, one environment, five
warmup control steps, the same four-channel action and 30 measured control steps.
All four runs used an active Kit viewport and measured 120 physics calls, 30
viewport renders, 30 DRR updates and 30 keyboard polls. Durations below are
CUDA-synchronized call latencies. Viewport graphics may overlap other GPU work;
these are not isolated GPU frame timings.

| Workflow | Backend | Physics ms/tick | Viewport ms/call | DRR ms/update | Input ms/poll | Total ms/control | Wall control Hz |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Catheter | legacy | 122.92 | 11.33 | 54.96 | 0.054 | 569.85 | 1.75 |
| Catheter | numerical | 42.29 | 10.41 | 56.08 | 0.062 | 245.83 | 4.07 |
| Arm | legacy | 131.00 | 11.88 | 42.55 | 0.049 | 599.98 | 1.67 |
| Arm | numerical | 42.52 | 11.03 | 55.24 | 0.050 | 255.69 | 3.91 |

Physics still advances at 120 Hz in simulation time, with decimation 4 and 30 Hz
control. Measured whole-loop physics throughput is four times the wall control
rate above; no viewport FPS claim follows from a render-call count. This is one
warmed sample on a shared GPU with the observed CPU powersave profile, not a new
performance acceptance threshold. Unrelated GPU jobs were left running.
Several legacy first loads spent minutes in CUDA driver module loading before
rollout; that startup time is excluded from the warmed samples. The initial
terminated timing attempt and its diagnostic traceback remain retained.

Early direct-driver windowed attempts did not select a Kit visualizer. Those
results establish physics/DRR continuation only. The final driver explicitly
selects Kit, asserts an active `KitVisualizer`, and the timing driver requires
nonzero viewport render calls. The standard launcher already selected Kit through
its existing `--viz kit` path; that launcher behavior remains intact.

The patient used for native checks is the existing schema-2 `s0011` bundle at
`pr292-merged-numerical/runs/endoluminal_navigation/20261120_115300/patient/patient_twin.yaml`,
manifest SHA-256 `9a4943540dfe481b40a8c755562169f36dbb121dcf5731d87e809d736b605af0`.
It references the original rc5 `patient-s0011` anatomy, centerline, radii and mask,
with the native scan volume metadata and original world transform. Earlier
schema-1 manifest rejection is retained in the attempted-run logs.

**Promotion blocker:** the unchanged bent-catheter attenuation test fails:
`test_catheter_attenuation.py::test_a_bent_catheter_matches_a_sampled_ray_march`.
The frozen Workflows `606e19df` renderer reproduces the same failure in isolation:
5.96167 versus sampled 5.63704, outside the existing 0.06 bound. Neither the
renderer nor its bound was changed to pass this migration. The original and
baseline-reproduction logs are retained.

**Additional upstream failure:** the complete Numerical Sim runtime/package suite
finished with **1 failed, 1,299 passed and 52 skipped**. The unchanged
`tests/runtime/test_clipping.py::test_tracker_and_clip` failed its ring-pinching
bound: before 4.76837e-7 and after 5.36442e-7, versus the existing requirement
`after < 0.35 * before`. Its endpoint arrays are retained as
`upstream-clipping-failure.npz` beside `upstream-full.log`. The clipping source and
test were unchanged by this migration; baseline reproduction of this particular
failure has not been established. No clipping mechanics or bounds were changed.
A failed observation is not waived by a passing mechanics comparison.

**Native startup failure retained:** the staged synthetic arm capture completed,
but its replay at `runs/endoluminal_navigation_arm/20261120_165239/` exited 139 in
Kit startup, before environment construction. The retained stack includes
`libomni.platforminfo.plugin.so` and telemetry. This observation is not classified
as a continuation failure or silently discarded; startup reliability remains an
open qualification concern even when later independent starts pass.

Dependency repinning and promotion remain withheld. The isolated candidate
installation is explicit and does not upgrade the declared dependency stack:

```bash
# From this Workflow worktree, after provisioning its isolated Arena environment.
I4H_NUMERICAL_CHECKOUT=/home/pkorzeniowsk/Projects/i4h/i4h-physics-simulation-internal-worktrees/elastic-rod
for component in physics isaaclab surgical assessment endoluminal; do
  uv pip install --python arena/.venv/bin/python --no-deps \
    -e "$I4H_NUMERICAL_CHECKOUT/numerical_sim/$component"
done
arena/.venv/bin/python scripts/arena_stack_artifacts.py verify-runtime \
  --numerical-root "$I4H_NUMERICAL_CHECKOUT"
```

A normal locked setup restores the declared baseline. Candidate qualification
must use the explicit candidate source until the source pin, lock and source
verification can advance together. Policy interface compatibility remains a
structural contract check; trained-checkpoint performance has not been evaluated.
No training campaign, mechanics tuning, new acceptance thresholds, pushes or PR
updates are included.
