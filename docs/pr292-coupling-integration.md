# PR #292 coupling integration

## Source and preserved history

Numerical Sim is pinned to
[`2e4067f5a46b1908714a118468081a14c0f82f81`](https://github.com/isaac-for-healthcare/i4h-physics-simulation/commit/2e4067f5a46b1908714a118468081a14c0f82f81).
This merge has parents `8812708` (PR #17's neutral-host split) and `76cfb2a`
(the PR #12 coupling branch, including implementation `06a9e64`). The PR #12
implementation is included once, resolved to the already qualified local
candidate. The original PR #12 and PR #17 worktrees and the latter's staged
patch remain intact.

The physics commit contains the selected-world FK/mount preparation before
episode restoration, mounted-catheter geometry/history restoration, separation
of cache reset and continuation, Direct-task acceptance coverage, WORLD-row
test corrections and the retained qualification reports. Production package
payloads and acceptance tests match the qualified candidate; only the setup
utility and integration provenance differ. The utility now accepts Workflows
descendants of the qualified `442e495b` base and verifies the exact dependency
profile instead of rejecting every later host HEAD.

This Workflows update changes the setup revision, current-source documentation,
setup tests and license link. Fresh liver policy exports identify the actual
installed editable physics checkout rather than stamping an obsolete SHA.
Existing policies, snapshots, recordings, calibration provenance and historical
qualification manifests are not relabeled. The campaign-specific
`scripts/build_liver_qualification.py` guard remains tied to its historical
source; this update does not manufacture a fresh liver training qualification.

Isaac Lab `3939445` plus its maintained patches, Newton `d31603f`, stock Warp
1.17.0, package versions, assets and numerical parameters remain unchanged.
The existing Workflows environments and setup-managed checkout are not replaced
during integration; `./setup.sh` installs the new source pin for future runs.

## Qualification and limits

The exact registered default tasks `I4h-Chole-Direct`, `I4h-Aorta-Direct` and
`I4h-ReachTarget-Direct` passed all **24 acceptance checks**: 1/8 environments,
eager/CUDA graph, meaningful controls and physical response, three complete
episodes, selected reset with unchanged peers, close/reopen and finite PPO
rollouts with checked optimizer updates. Fluid-pool physical state is included.
The original candidate's affected Workflows hosting checks passed **129**.

The completed physics runtime/package suite was **1,265 passed, 4 failed,
13 skipped**. These results are reused by verified production-payload identity,
not relabeled as a new full qualification of the merge commit. The open gates
are clipping, Chole phase separation, standalone/host fluid-suction parity and
the package pin check: the package still declares Newton `7f0c13a`, so its
unmodified default pin gate rejects this explicitly approved `d31603f` host.
The ordered candidate scissors failure on the original locked stack remains a
separate regression risk despite passing in the delivery-stack full run.

See the [physics qualification report](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/2e4067f5a46b1908714a118468081a14c0f82f81/numerical_sim/docs/dlv-010-29.md)
for exact source/patch identity, task commands, layouts, failure dispositions and
local demonstration/evidence locations. Raw evidence remains in the original
physics worktree at `numerical_sim/.tmp_tests/dlv-010-29/`.

Integration-specific checks and provenance are recorded locally under
`runs/physics_coupling_integration/20261125/`; the PR description records the
completed checks on this source. The setup/runner commands in the physics
report run the registered tasks against separately installed wheels without
changing the existing Workflows environment.

| Integration source check | Result |
| --- | --- |
| Repository/common CPU | 493 passed, 3 skipped |
| Affected liver RL CPU, including source-export provenance | 23 passed |
| Arena lock check | Passed; lockfile unchanged |
| Production package source identity | All 183 tracked package source files match the qualified candidate |
| Applicable pre-commit hooks | Passed |
| Broader optional RL suite | 183 passed, 18 failed, 37 skipped |

All 18 broader RL failures report that the optional RLinf checkout is missing;
its optional source/runtime checks also account for the skips. The checkout
was not installed by this integration, and these results are not reported as a
passing full RL suite. The affected liver checks, including two real-Git export
provenance tests, pass without that backend.

**Release and merge readiness remain blocked.** Existing liver replay success,
viewport-video startup and final-source performance gates remain open. Prior
training and fixed-policy results retain their original source identities.
No new convergence, hardware, replay/video-release or performance qualification
is claimed. Unrelated GPU jobs remain running.
