# Rod contact and arm feedback

The contact path now alternates live vessel contact with a global elastic rod
solve. It retains the XPBD multipliers across those iterations and transmits
the resulting proximal force **and moment** to MJWarp. This replaces the WIP
contact path's colored local elastic sweeps and force-only arm feedback.

## Elastic solve and material units

Each edge contributes six stretch/shear and bend/twist constraints. Adjacent
edges share position and orientation variables, so their linearized system is
block tridiagonal. `RodLinearSystem` assembles and factors that system in FP64;
particle state, Jacobians, and stored multipliers remain FP32. Physical
rotational inverse inertia is much larger than translational inverse mass for
a slender rod, making FP32 elimination unreliable at the 120-edge resolution.

The bend residual is the imaginary part of the relative material quaternion,
minus its rest value. It is a dimensionless half-angle measure. For a uniform,
solid circular section:

```text
I = π r⁴ / 4                 J = 2 I
EI = E × bend_stiffness × I
GJ = G × twist_stiffness × J

effective bend stiffness = 4 EI / segment_length
effective twist stiffness = 4 GJ / segment_length
XPBD alpha = 1 / (effective stiffness × substep_dt²)
```

The former implementation omitted the section moments and used the opposite
length dependence. Its incomplete local solve concealed excessive stiffness.
The Arena mesh-compensation multiplier has been removed: refinement now keeps
the authored material multipliers unchanged. Existing stiffness values tuned
to the previous law need reconsideration. This corrects the small-angle elastic
law; it does not calibrate a particular manufactured catheter.

Stretch/shear retain the existing nearly inextensible constraint setting.
Arena supplies physical cylinder inertia. The standalone solver's legacy
identity-inertia option and alternative linear backends have not been removed.

## Contact iteration

Within each substep, the active centerline-contact path repeats:

1. Project vessel elasticity when vessel deformation is enabled.
2. Project contact against the current vessel geometry.
3. Recompute rod residuals, Jacobians, and world inertia, then solve the global
   elastic system and update both positions and material orientations.

Elastic multipliers are initialized once per substep. Velocities are computed
from the final solved state, with no subsequent position-only length cleanup.
`CatheterRodSpec.contact_coupling_iterations` defaults to 32. A fixed iteration
budget gives approximate contact/elastic reconciliation, not a convergence
guarantee. Setting it to zero selects the legacy ordering for comparisons.

This remains a centerline-tube wall model with its existing contact/friction
rules. The change does not introduce a monolithic contact matrix, a new friction
law, self-contact, or residual-based stopping. FP64 and repeated global solves
increase work per substep; real-time performance is not established.

## Proximal wrench

For edge zero, the translational Jacobian on the pinned root is identity. The
solver evaluates the final rotational Jacobian and reports:

```text
F_root = lambda_stretch / substep_dt²
M_root = Jrot_root.T × lambda_all_six / substep_dt²
```

The moment includes both the stretch constraint's local lever arm and the
bending/twisting couple. The public `proximal_wrench(out=...)` fills a persistent
buffer; `proximal_reaction` remains a force-only compatibility API.

The coupler treats that wrench as transmitted at the hardware grip. With
`mount_local` and `body_com` expressed in the drive body's frame:

```text
F_body = F_root
M_body = M_root + cross(R_body × (mount_local - body_com), F_root)
```

The Franka embodiment sets `drive_body_name="panda_hand"` and
`drive_mount_local=(0, 0, 0.1034)` metres. The fixed grip offset avoids making
the torque arm grow with commanded insertion. Pure torque is transmitted even
when the root force is zero. Reset clears the complete wrench history.

The rigid/rod exchange still has a one-substep lag and the existing relaxation
control. This is not a simultaneous or ADMM arm/rod solve. Private rod frames
are sufficient to recover this wrench; upstream Newton particle-frame support
is not required for the harvest.

## Verification on 2026-09-11

Physical CPU references in `arena/tests/test_catheter_physical_response.py`:

- Cantilevers with 20 and 40 edges match `F L³ / (3 EI)` within about 1.1%.
  Root force and moment match the applied tip load within 3%.
- A pure twist gives the energy-derived torque with zero force, for one and
  three independent environments.
- The 120-edge block solve matches an independently assembled NumPy dense
  system to relative tolerance `5e-5`.

Contact tests in `arena/tests/test_catheter_contact_coupling.py` exercise curved
hold/insert/stop/retract trajectories and independent batched rods. The full
303.2 mm, 120-edge cases use 32 iterations at 120 Hz, including a three-env
deforming vessel case. Sampled checks require surface penetration below 0.1 mm,
total length error below 0.5 mm, segment strain below 1.5%, and finite wrenches.
All eleven physical/contact reference cases pass. These are synthetic cases;
they do not establish the same tolerances for every patient geometry.

All 55 drive-feedback tests pass, including intrinsic torque, grip offset,
body-COM shift, relaxation, and reset. The centerline integration suite passes
after explicitly separating wall-only tests from gravity and checking transient
contact over the trajectory instead of assuming it persists at the final step.

The final broader CPU run reports **457 passed, 2 skipped, 2 deselected**;
together with the eleven physical/contact cases, 468 tests passed. The two
docstring import checks and four collision-manager tests were excluded from
that CPU run because they bootstrap Isaac Sim/Kit. One installed-config check
also skips when Kit cannot initialize; the actual coupled config and collision
manager were exercised by the GPU demo. Ruff checks on the changed Python
files and whitespace checks in both checkouts pass.

`./run.sh lint --all` passes. GPU integration used `endoluminal_navigation_arm`,
mode `demo`, patient twin `s0011`, with the local `basic/catheter_sweep` task:

- [Completed run](../runs/endoluminal_navigation_arm/20260911_192740/run.json):
  exit 0, **1/1 episodes succeeded**, one attempt, 192 recorded frames and one
  `catheter_sweep` segment. Action width is 4; state width is 11. All recorded
  state, action, and attenuation values are finite. Nonzero forces and moments
  appear in the coupling log. The default desktop display was unavailable, so
  this run rendered fluoroscopy without an interactive window.
- [Sampled sweep images](../runs/endoluminal_navigation_arm/20260911_192740/sweep_frames.jpg)
  were visually inspected across insertion, rotation, bending, retraction, and
  C-arm motion. They show a continuous rod without visible node-scale folds.
- [Visible attempt](../runs/endoluminal_navigation_arm/20260911_194000/run.json)
  used desktop `:1`. Playback stopped after 29 frames, leaving status `running`,
  **0/1 episodes succeeded**, exit 1; no solver exception was logged. Its
  [simulator image](../runs/endoluminal_navigation_arm/20260911_194000/simulator.png)
  and partial recording were inspected, but it is not a completed visible
  rollout.

The demo success flag means the scripted sweep completed; it does not mean the
catheter reached the navigation target. Wrench accuracy under all transient
contacts, patient-specific penetration bounds, and high-speed arm/rod stability
remain separate validation questions.
