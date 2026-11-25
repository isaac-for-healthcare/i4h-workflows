# Numerical catheter runtime migration

This branch is stacked on Workflows PR #292. Its cross-repository dependency is
[physics PR #19](https://github.com/isaac-for-healthcare/i4h-physics-simulation/pull/19),
stacked on physics PR #17. Physics PR #8 and Workflows PR #252 remain source proposals and
are unchanged. The two workflow backends remain selectable; legacy remains the
fresh-run default until the required migration gates pass.

## Runtime integration

Free navigation writes complete component-frame root poses through
`CatheterControl.write_root_poses`. Mounted navigation retains the shared-model
rigid mount as root owner and commands relative insertion, rotation and bend.
Action terms integrate physical rates once; the upstream control consumes
`absolute_targets` with `target_mode: absolute`. The upstream default remains
rate integration for other callers.

Curved startup positions and transported node frames are authored independently
from uniform material rest lengths and optional shaft rest curvature. The
existing material conversion, physical-inertia selection and raised-cosine tip
bend taper are retained. Patient points and radii retain their existing frame
and unit conversion. Endpoint locking, independent wall damping, colored wall
projection, containment and wire-only interior pull use upstream APIs.

The six operational physics references move together to one immutable published
candidate. The historical PR #292 coupling report retains the source revisions
it actually tested. Dependency versions and lockfiles are unchanged. The approved
Isaac Lab/Arena stack and its compatibility checks remain required.

Native Arena owns simulation stepping and decimation. Existing action and task
contracts, success conditions, step limits, controls, fluoroscopy and opt-in
recording remain in force. Portable selected-world continuation includes State
and pending Control commands; renderer/task/arm state remains in the workflow
bundle. Replay uses recorded backend provenance, rejects conflicting overrides,
and requires explicit legacy selection for recordings without backend metadata.

## Qualification

The runtime capability matrix and full physics checks live with the physics
candidate. Workflow qualification uses the unchanged frozen geometry and
`scripts/catheter-migration-acceptance.json`, plus native-host lifecycle,
continuation, recording/replay, synthetic/patient and policy-update checks.
`profile_navigation_backends.py --worlds 1|16|64` records phase timing diagnostics
without adding a performance acceptance gate.

The historical mechanics, native-startup and numerical failures in
[endoluminal-elastic-rod-qualification.md](endoluminal-elastic-rod-qualification.md)
remain part of the record. Migration acceptance does not clear unrelated release
blockers. Converged-policy training and legacy package removal are outside scope.

A separate default-switch PR may be opened on this branch only after the required
migration gates pass. Draft status and a published candidate do not constitute
qualification or permission to merge.
