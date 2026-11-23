# PR 292: apply the 7011011 test correction to current physics main

[Physics PR 23](https://github.com/isaac-for-healthcare/i4h-physics-simulation-internal/pull/23)
contains commit `6cf9eae60e16091397539262ee0a074aa95c827d`, based directly on
committed main `4f49fcf34e038b848355b60f1d14ff89772249a4`. Its only differences are
`tests/runtime/test_clipping.py` and `docs/nvbug-7011011.md`. Production Python,
case configuration, dVRK assets and dependency versions are byte-for-byte unchanged.
The older broader QA PR was not merged into main or restored by this correction.

The test follows the nearest live, eligible, unclipped duct node for 90 released-fire
steps before the second press. It checks actual reach, eligibility, live topology,
no premature clipping, unchanged graph and environment isolation before firing.
The second clip must be distinct and the other environment must remain at zero;
existing pinch, reach and reset assertions remain unchanged.

The standalone clipping/haptics selection passes **44 tests in 23.46 seconds**.
All 87 production manifest assets verify; actual source imports and the patched
Warp 1.17.0 native hash were checked in an isolated current-main worktree.

Fresh Workflows validation on the repinned checkout:

| Check | Result |
| --- | --- |
| Repository/common | 475 passed, 3 skipped |
| Arena CPU | 1,303 passed, 4 skipped |
| Isolated RL CPU | 222 passed, 1 skipped |
| Dataset tools | 46 passed |
| Arena lock and affected-source pre-commit | Passed |
| Corrected clipping/haptics on hosted dependencies | 44 passed in 22.53 seconds, exit 0 |
| Full standalone runtime/package campaign | **1,273 passed, 51 skipped, 4 failed**, exit 1; corrected clipping test passed |

The full run completed uninterrupted in 1,191.06 seconds. Its four failed nodes are:

- `test_env_isolation.py::test_chole_batched_matches_single`: environments differ
  by 0.0001804, below the test's >0.005 precondition. The intended environment
  comparison is not established by this failure.
- `test_instruments.py::test_hook_acquires_from_record`: acquisition `[1, 0]`
  does not match expected `[1, 1]`.
- `test_continuation.py::test_mounted_chole_tool_resumes_with_a_held_grasp_and_a_pending_reaction`:
  the fixture fails to establish any active grasp before testing continuation.
- `test_interaction_owned_storage.py::test_mounted_tool_resumes_with_a_held_grasp_and_a_pending_reaction`:
  the analogous owned-storage fixture also establishes no active grasp.

No failed node was rerun for a pass and their causes are not established here.
Rendering opt-in was not enabled in this full run, so skipped rendering checks do
not qualify the previously malformed render fixture. GB300 remains unvalidated.

The generated CPU synthetic example and all six wheel/source builds pass. Artifact
inspection verifies six wheel package boundaries and six source archive paths.

Workflows updates setup, all five package sources, lock metadata, exported-policy
revision metadata, qualification builder and source-dependent tests together.
The exact new dependency checkout is used by the local test environments.
The runtime-only checks from the preceding 4f49fcf repin are not rerun: the dependency
change contains no production or asset changes. Their original outcomes remain in
[the preceding report](pr292-physics-main-repin.md), including failing liver replay,
the video-enabled Kit startup crash and the malformed rendering fixture. This
7011011 correction does not waive or fix those issues, or restore containment caps.
GB300/aarch64 remains unavailable.

Physics evidence is under `numerical_sim/.scratch/20261123_225032/` in the isolated
`7011011-current-main` worktree. Workflows evidence is under
`runs/liver_reach_retract/20261123_225157/` in `v010-qualified-integration`.
No main merge or public Numerical Sim publication is performed by this checkpoint.
