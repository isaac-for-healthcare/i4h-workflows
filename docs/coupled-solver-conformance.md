# Does Our Coupled Solver Follow Isaac Lab's Coupling Model?

A conformance review of `NewtonCoupledMJWarpXPBDRodManager` against Isaac Lab's
[Coupled Solvers](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/coupled_solvers.html)
concept page. For *how* our coupling works, see
[Coupling MJWarp Rigid Bodies to the XPBD Catheter Rod](mjwarp-xpbd-rod-coupling.md); this
document only asks how it lines up with the documented framework.

## Verdict

**The algorithm matches. The API does not — and the API was never available on this pin.**

We implement proxy coupling in `lagged` mode, which is the mode the documentation recommends
starting from, with the same source/destination shape as the maintained Franka MJWarp–VBD
task. What we do not use is `isaaclab_contrib.coupling`: no `CouplerEntryCfg`, no
`CouplerProxyCfg`, no ownership selectors. Instead `NewtonCoupledMJWarpXPBDRodManager`
subclasses `NewtonManager` and owns the substep order directly.

That is not a shortcut around the framework. It predates it:

```
newton version: 1.2.1
NO   newton.solvers.experimental.coupled   (framework landed in Newton 1.4.0)
NO   isaaclab_contrib                      (not installed)
yes  isaaclab_newton.physics
```

`proxy_coupling.py` states this in its module docstring — it carries the upstream recipe "onto
the Newton revision this package pins, which predates the framework."

There is a sharper finding underneath that one, developed in
[Is this interface really a proxy?](#is-this-interface-really-a-proxy) below: our arm–catheter
interface is not a proxy interface at all. Upstream now models it directly, as a rigid-body-to-
particle ADMM attachment. Proxy coupling was a reasonable approximation of an interface the
framework had no primitive for at the time, and now has.

## The documented model, briefly

A coupled simulation starts from **one** Newton model, partitioned into named entries that own
disjoint parts of it. Each entry selects a solver and advances only what it owns. An interface
— proxy or ADMM — connects the entries, and Newton, not Isaac Lab, owns the coupling algorithm.

Two interfaces are offered. **Proxy** makes a source-owned body or particle appear as a virtual
endpoint in a destination solver, and the destination returns feedback on a later pass; it
suits naturally directional interaction such as a rigid collider inside a deformable. **ADMM**
creates symmetric interface constraints, iterates the sub-solvers, and applies equal and
opposite interface forces; it suits interfaces that should not be assigned a direction,
including body–particle attachments.

## Where we follow it

| Documented concept | Our equivalent |
|---|---|
| One shared Newton model, disjoint ownership | MJWarp owns the body arrays; the rod owns a contiguous `RodParticleRange` of the particle arrays |
| Proxy: source endpoint appears in the destination | The arm's mount pose is written to the rod root via `set_root_pose_gpu` |
| Destination returns feedback on a later pass | `proximal_reaction` is harvested after the rod solve and consumed next substep |
| `mode="lagged"` ordering | `apply(state_0)` → rigid solve → rod solve → `harvest(state_1)` |
| Proxy relaxation (`proxy_relaxation`) | `drive_reaction_relaxation`, blended against the previous iterate |
| Proxy supports at most two entries | We have exactly two |
| Stabilize each entry before tuning coupling | The rod and the arm are both validated standalone first |

The direction assignment is the same as the documented Franka example: source `rigid`,
destination the deformable. `blend_coupling_forces_kernel` is written explicitly to match
upstream's `proxy_relaxation`, and the lagged call order is spelled out in the
`DriveReactionCoupler` docstring.

## Where we depart deliberately

Each of these is documented in `proxy_coupling.py`, and each has a reason that is specific to a
catheter rather than a matter of taste.

**The harvest source.** Upstream reads a proxy *body's* momentum change, which works because
its sub-solvers move the proxy during their own solve. `NewtonCathRodSolver` never writes body
state, and the rod's proximal particle carries zero inverse mass because `lock_root` pins it.
So a body-momentum harvest and a particle-momentum harvest would both report exactly zero, and
two-way coupling would silently behave as one-way. The reaction is taken from the rod's own
constraints instead, via `XPBDRodSolver.proximal_wrench`. This keeps solver internals inside
the solver, the way upstream keeps its `CouplingInterface` hooks on each sub-solver rather than
in the coupler.

**No gravity cancellation.** Upstream cancels gravity so a proxy body is not pulled by it
twice. Our feedback is a constraint force that legitimately carries the weight of wire hanging
off the drive point, and the driving body's own weight stays with the rigid solver.

**No pre-step snapshot.** Upstream differences velocity across the step. We read constraint
state after the solve, and the rod supplies its own substep scaling, so there is nothing to
snapshot beforehand.

**The moment arm is the mount, not the proxy point.** `mount_local` is fixed in the drive
body's frame. Using the proximal particle instead would tie the arm to insertion depth: under
a roller drive that particle travels away up the vessel, inventing a torque that grows with
fed length. Measured on the `s0011` twin, that drifted the flange 28 mm off the access site
after one second and 34 mm after two.

## Where we are genuinely thinner

These are gaps rather than justified departures. Most of them only matter if we stay on a
proxy-shaped interface; the last one is the one that matters regardless.

- **No ownership selectors.** The documentation partitions by regex USD paths
  (`bodies=[...]`, `all_particles=True`, `include_static_shapes=True`). We use an integer
  particle range plus `register_rod`. Equivalent for one rod, but it cannot express a richer
  partition.
- **Coupling iterations are pinned at one.** We perform a single apply and a single harvest per
  substep. The documentation's advice to raise `iterations` once each entry is stable has no
  lever here; only the relaxation weight is exposed.
- **No `mass_scale`, `collide_interval`, or `staggered` mode.**
- **We do not reuse the destination's contact path.** The documentation lists that as a proxy
  advantage. Our rod ignores Newton's contact buffers entirely and resolves vessel containment
  in its own kernels, so `collide_interval` has no analogue.
- **The arm exchange remains lagged.** Intrinsic bending/twisting moments are now
  transmitted by contracting the rod's private rotational Jacobian with its
  multipliers; the generic Newton particle interface is not needed for that
  harvest. See [the implementation and tests](rod-contact-feedback.md).
- **No ADMM option — and this is the consequential one.** ADMM is not a nicety we skipped; it
  is where our interface actually belongs. See
  [Is this interface really a proxy?](#is-this-interface-really-a-proxy).

## Is this interface really a proxy?

No. Checked against upstream's `newton/_src/solvers/coupled/solver_coupled_proxy.py` and the
`newton.solvers.experimental.coupled` API reference, our interface does not fit the proxy
pattern, and ADMM has a primitive built for it.

**Why proxy does not fit.** A proxy body is a real body in the shared model, owned by neither
entry, whose inertia is overwritten from the source solver's effective mass
(`_apply_proxy_body_effective_masses`; `_validate_proxy_destination_ids_not_owned` enforces the
non-ownership). Feedback is then the destination's *momentum change* on that body. That works
only if the destination solver actually simulates the proxy, which is how VBD collides its
particles against the Franka hand. `NewtonCathRodSolver` has no body awareness whatsoever and
resolves collision as centerline containment in its own kernels, so a proxy body in the rod
solver has nothing to act on. Inverting the direction does not help: particle proxies would
require MJWarp to simulate particles.

The root cause is that **proxy coupling assumes a contact interface, and ours is an
attachment.** Our `two_way` mode is a reaction path bolted onto a directional exchange
precisely because the interface is not directional.

**What ADMM offers instead.** `SolverCoupledADMM` provides a purpose-built primitive:

```python
SolverCoupledADMM.add_body_particle_attachment(
    builder, body, particle, *, body_point=(0.0, 0.0, 0.0), stiffness=1.0e4, damping=0.0, enabled=True
)
```

"Add a model-level rigid-body-to-particle ADMM attachment." That is our interface in our own
terms: a rigid body, a rod particle, and a body-local attachment point that corresponds exactly
to our `mount_local`. Rows are stored as `coupling:body_particle_attachment` custom attributes
registered via `register_custom_attributes(builder)`, and `SolverCoupledADMM` "converts rows
whose body and particle endpoints are owned by different solver entries into ADMM attachment
constraints." Authoring the attachment at build time is therefore the sanctioned path — the
documented limitation about arbitrary user-authored endpoint records applies to something
narrower than it first appears.

None of this is available on our pin: Newton 1.2.1's `ModelBuilder` exposes only `add_particle`,
`add_particles`, `add_particle_grid`, and `_validate_kinematic_joint_attachment`.

**What ADMM would not solve.** The attachment is translational, so the intrinsic root bending
moment stays untransmitted. Attachment stiffness needs tuning between a wire that lags the
gripper and ADMM convergence trouble. And the roller-drive feed still needs its own mechanism,
because the attachment holds the wire without advancing it.

> **Partly corrected by the Newton team.** The untransmitted bending moment is confirmed:
> Newton models cables as rigid bodies, so particle frames do not exist and an angular row
> would have nothing to act on. The stiffness-tuning trade-off is not real; see
> `admm-coupling-newton-answers.md`.

## What porting to the framework would take

Not a configuration change. There is a hard prerequisite, a mechanical middle, and one physics
risk that dominates the estimate.

**The gate.** Newton 1.2.1 → ≥1.4.0, plus an Isaac Lab develop tree carrying
`isaaclab_contrib`, which is not installed today. This is the whole schedule risk. In our
favor, the solver only touches `Model`, `State`, `SolverBase`, and `SolverMuJoCo`, which the
package's `pyproject.toml` notes are "present in every newton release." Against us, both
Newton and Isaac Lab flag the framework experimental, and the arena compatibility surface is a
larger unknown than the solver.

**Declare ownership as entries.** Translate the `RodParticleRange` window into
`CouplerEntryCfg` selectors. Mechanical, since the partition already exists explicitly. One
thing to check first: whether `CouplerEntryCfg.solver_cfg` accepts a non-Isaac-Lab
configuration such as our `XPBDRodSolverCfg`, or only registered types like `MJWarpSolverCfg`
and `VBDSolverCfg`. If the latter, either register a solver type or drive Newton's coupler
directly and skip the Isaac Lab layer.

**Author the attachment.** Call `register_custom_attributes` on the builder and add one
`add_body_particle_attachment` row per env, mapping the drive body and the rod's proximal
particle with `body_point` set from `mount_local`. Small, and it replaces `DriveReactionCoupler`
outright.

**Un-pin the root — the real cost.** ADMM's attachment is compliant, a quadratic penalty with
stiffness in N/m and damping in N·s/m. Today the root is kinematic: `lock_root` gives it zero
inverse mass and the predictor zeroes velocity on zero-inverse-mass particles, so it teleports
to the commanded pose. Under ADMM it becomes a dynamic particle on a stiff spring, moved by the
body rather than by `set_root_pose_gpu`.

> **Corrected by the Newton team.** ADMM constraints are hard, not compliant — it is an
> augmented Lagrangian formulation, so the kinematic constraint is exactly satisfied on
> convergence, and the multipliers are warm-started across timesteps. Stiffness is a penalty
> weight, not a grip stiffness, and there is no Baumgarte term. Un-pinning the root therefore
> does not cost exact placement, and the stiff-spring prototype recommended below over-states
> the error. See `admm-coupling-newton-answers.md`.

The gain is a genuinely symmetric interface with equal and opposite forces, no lag, and no
relaxation weight to tune. The whole reason `proximal_reaction` exists — a zero-mass root that
no momentum harvest can see — disappears, because the attachment row *is* the force. Our stated
reason for commanding pose rather than rate also survives: a mount that stalls still stops the
wire by exactly as much as it stopped itself.

The cost is that un-pinning the root touches the machinery that took the most effort to
stabilize. Containment-versus-elastic-solve and the distance-cleanup pass both currently assume
a fixed root.

**Implement the surviving `CouplingInterface` hooks.** Fewer than the proxy path would need,
since the harvest problem is gone. `coupling_eval_gravity_acceleration` is confirmed necessary
— the rod applies its own per-env gravity in its predictor, so the coupler must remove exactly
that. `coupling_supports_inertial_property_refresh` governs graph capture.

**Validate on physical metrics.** Both docs are explicit that no approach is uniformly more
accurate and that comparison must be on task-relevant metrics. We already hold baselines: chord
uniformity, containment penetration, and the 28 mm / 34 mm flange drift figure.

### Recommended sequencing

Prototype the un-pinned compliant root **before** paying for the upgrade. It is testable in
isolation on the current stack by relaxing `lock_root` and holding the root with a stiff spring
by hand. If the rod stays stable under containment and cleanup with a compliant root, the rest
is largely plumbing; if it does not, that is the finding, and it is cheap to obtain.

Two documented limitations will still apply afterwards: no nested couplers, and no Newton
contact sensors. The second is already true of us for a different reason.

## Re-verifying the version claim

```bash
./arena/.venv/bin/python -c "
import importlib
for m in ('newton.solvers.experimental.coupled','isaaclab_contrib.coupling'):
    try: importlib.import_module(m); print('yes ', m)
    except Exception as e: print('NO  ', m, '->', type(e).__name__)
import newton; print('newton version:', newton.__version__)
from newton import ModelBuilder as B
print('attachment API:', [n for n in dir(B) if 'attach' in n.lower()])"
```

Checked 8 September 2026 against the `arena` environment, which reported Newton 1.2.1, neither
module importable, and no `ModelBuilder` attachment API.

Note that `rg` and editor search skip the gitignored virtualenv, so a text search for
`CouplerProxyCfg` returns nothing whether or not the package is installed — a control search
for `class SolverBase`, which certainly exists, also returns nothing. Import the module to tell
the difference.

## Sources

- Isaac Lab, [Coupled Solvers](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/coupled_solvers.html)
- Newton, [Coupled Solvers concept page](https://newton-physics.github.io/newton/stable/concepts/coupling.html)
- Newton, [`newton.solvers.experimental.coupled` API reference](https://newton-physics.github.io/newton/stable/api/newton_solvers_experimental_coupled.html)
- Newton, [`solver_coupled_proxy.py`](https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/coupled/solver_coupled_proxy.py)
