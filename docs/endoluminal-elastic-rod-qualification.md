# Endoluminal elastic-rod migration qualification

The migration is incomplete. Neither navigation workflow has changed backend,
recording behavior, runtime dependencies or defaults. Workflow integration and
promotion require a qualified upstream mechanics revision.

## Source boundaries

Work uses separate `migration/elastic-rod` worktrees based on Workflows
`dfa5b83f`, Numerical Sim `caff73b1` and legacy physics `5005619`.

| Change | Local commit |
| --- | --- |
| Current-pin parity harness and frozen fixtures | `4676b89` in Numerical Sim |
| Opt-in mechanics candidate | `b07bba59a59eb5a1347583806c73e834a1a53ac6` in Numerical Sim |
| Workflow qualification driver | `88303728a57a7d45992c3881545641f0aae73092` in Workflows |

Both final navigation archives identify clean source checkouts, the same harness
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

Native numerical environment adapters, `--catheter-backend`, backend/revision
recording metadata, replay dispatch, portable continuation and the dependency
repin have not been implemented. They remain Stage 3 work after mechanics
acceptance.

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

## Current acceptance result

The nine navigation fixtures produced **47 passing rows and 3 failing rows**.
Numerical remains unpromoted.

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
RMS is `5.669586e-11 N m`, and RMS difference is `1.133797e-10 N m`. The frozen
relative bound has no numerical-noise exemption. No threshold, material setting,
iteration count or response parameter was changed to accept it.

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

The final native-host gates have not run: patient-backed and synthetic workflows,
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
| Workflows CPU/coverage command | 600 passed, 3 skipped; unavailable RL suite reported separately |
| Workflows qualification tests with Torch installed | 4 passed |
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
