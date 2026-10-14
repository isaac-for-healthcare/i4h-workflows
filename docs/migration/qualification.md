# Internal endoluminal migration: implementation checkpoint

Upstream dependency: [runtime capability draft #14](https://github.com/isaac-for-healthcare/i4h-physics-simulation-internal/pull/14). This proposal was tested against upstream commit `addb3f7`; the six production source pins are deliberately unchanged.

2026-10-12. **Partial implementation; not a qualified internal replacement and
not deprecated.** The upstream branch is `migration/endoluminal-internal`; the public workflow
proposal branch is `proposal/numerical-catheter-migration`.
Baselines and fixture limits are recorded in `internal-endoluminal.md`.

## Implemented upstream

- Independent startup positions/xyzw frames and material rest lengths/Darboux,
  validated before body allocation; reset restores startup placement.
- Authored external root ownership, batched NumPy/Warp/Torch command API and
  I4hComponent forwarding; persistent buffers, Torch stream ordering, quaternion
  roll, selective command reset and mount ownership rejection.
- Immutable curved guide and independent distal-free-node count, with straight
  guide defaults retained.
- Per-environment linear-containment interior pull with no wall reaction or
  contact reports; real wall contact keeps two-way response and exact candidate
  overflow fallback. Existing spline behavior retained.

## Workflow preparation

`arena/i4h_arena/medical/numerical_catheter.py` in the workflow worktree converts
wire geometry/material/startup settings to the new explicit systems, rejects
obsolete overrides, and applies the authored cosine tip taper through public
regional tables. Five focused tests cover conversion, seeded versus straight
material, unsupported overrides, stepping and material persistence across reset.

This is a wire-case converter, **not an activated workflow migration**. It does
not add the patient vessel, replace managers/sensors, configure rigid mounts,
implement world/component conversion, or change existing workflow entry points.

## Verification

| Check | Result |
| --- | --- |
| Locked standalone install, plus locked render extra | Passed; Newton 1.6.1, Warp 1.17.0 |
| Final complete upstream runtime/package command | **522 passed, 226 skipped**, 3 deprecation warnings |
| New root/startup/guide/interior tests | **38 passed**; E=1,2,16; eager/captured, device inputs, reset isolation |
| Existing catheter port and mount/reaction tests | **13 passed**, including the unchanged bent-lumen bound |
| Core synthetic CPU example | Passed |
| Six wheels and six source distributions | Built; all artifact boundary inspections passed |
| Workflow converter + existing drive and diagnostics | **115 passed** on public main base; original internal base: 108 passed |
| Public workflow discovery, navigation lint and show | Passed |
| Vascular regression against changed upstream source | **53 passed**, using the user's existing vascular working tree and its declared data directory |
| Changed-file Ruff and git diff whitespace checks | Passed |

Skipped upstream checks include Isaac Lab, MuJoCo-Warp, production assets and
opt-in native/render tests. They are not passes. The vascular data directory
contains the vascular fixtures but not the required aorta mesh/centreline.

Failed attempts, retained as evidence:

- Initial upstream core-only run: 516 passed, 226 skipped, one XR test failed
  importing absent `slangpy`. Installing the locked render extra resolved it;
  no test or tolerance was changed.
- Broader workflow selection could not collect `test_catheter_initialization`
  (missing `scipy`) or `test_catheter_attenuation` (missing `xray_simulator`).
  The available selection does not stand in for these checks.

Raw logs are retained outside the repositories. Reproduction commands are below.
No benchmark or native UI result is inferred from unit tests.

## Outstanding release gates

1. Validate the proposal consumer contract against patient fixtures, then activate I4hComponent in both
   workflow-main and navigation consumers, reusing the f6 component integration.
2. Patient graph construction/contact, world/component transforms, complete
   action/observation/recording preservation, rebuild rebinding and manager-only
   stepping. Declare and label the main workflow's whole-shaft kinematic mode.
3. Lab 6864dd1 qualification: mounted two-way coupling, deterministic task
   arrival and isolation, finite PPO update, native catheter/fluoroscopy controls,
   keyboard release and successful process exit. A prepared Lab environment
   and explicit production asset root have been requested from the user.
4. Diagnostic old/new trajectories, deformation, reactions and performance at
   E=1,16,64; no new thresholds.
5. Validated upstream revision, all six dependency pins updated together,
   clean internal install/run with the old package absent, and only then the
   deprecation notice retaining the public fluoroscopy consumer.

Existing dirty checkouts and the endoluminal-ui worktree are outside these draft
branches. Source pins, old-package dependencies and public package availability
remain unchanged. The workflow draft targets public `i4h-workflows/main` at `5e2ebb17`, where
the navigation implementation is already present. Only four proposal files
are added. The internal repository is archived; publication moved to the
public repository with user authorization.

## Review corrections and reproduction

Final review reproduced and fixed default guide exclusion retaining an obsolete
`tip_edges` value after a structural edit. Added a regression test. The workflow
converter now authors the actual rate ceilings (0.060 m/s, 1.5 rad/s, 1.5 rad/s),
with normalization explicitly left to the future SI action adapter, and preserves
the legacy integer conversion for distal-free nodes. The provisional converter
used ceiling; its original 0.03 m / (0.1 m / 30) fixture incorrectly expected 9,
where the legacy float calculation truncates to 8. The corrected assertion and a
nonintegral 0.031 m fixture pin that source contract; no physics tolerance changed.
The converter's smoke test runs on CPU so it does not require a GPU in workflow CI.

From upstream `numerical_sim/`:

```bash
uv sync --locked --group dev --extra render
uv run --no-sync pytest tests/runtime tests/packages -q -ra
uv run --no-sync python examples/core_synthetic.py --device cpu
uv build --all-packages
uv run --no-sync python scripts/check_artifacts.py dist
```

For the workflow proposal, use the upstream candidate's Python environment (all
six editable packages from the same worktree), rather than the workflow's old
source pins. Set `UPSTREAM_PYTHON` to that environment's Python, then run from
the workflow root:

```bash
PYTHONPATH=arena:common "$UPSTREAM_PYTHON" -m pytest arena/tests/test_numerical_catheter.py arena/tests/test_catheter_drive.py arena/tests/test_catheter_diagnostics.py -q
```

Vascular verification used the existing vascular working tree and its declared
`data/` fixtures, with that same candidate environment:

```bash
PYTHONPATH=src "$UPSTREAM_PYTHON" -m pytest tests -q
```

These source-worktree checks do not establish clean installation against a
published/pinned candidate or an installation without the legacy solver.
