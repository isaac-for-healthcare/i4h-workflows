# The Catheter Solver and its Isaac Lab Integration

How the catheter is simulated, and how a solver that is not part of Isaac Lab ends up being
stepped by Isaac Lab's `NewtonManager` alongside a MuJoCo-Warp robot.

This document covers the solver and the integration seam. The force/pose exchange between the
robot arm and the wire is a separate concern with its own document,
[`mjwarp-xpbd-rod-coupling.md`](mjwarp-xpbd-rod-coupling.md).

## What is being modelled

A catheter is a **Cosserat elastic rod**: a slender body whose configuration is a centerline
plus a material frame at every point. The frame is what distinguishes it from a mass-spring
chain — it carries torsion, so the rod knows the difference between bending and twisting, and
an axial rotation applied at the groin propagates to the tip. That is the whole basis of
catheter steering.

Discretely the rod is `N + 1` particles carrying positions `x_i`, and `N` edges carrying
orientation quaternions `q_i`. Six constraints per edge tie them together:

```
C_str = (x_i + R(q_i)(0,0,+l/2)) - (x_{i+1} + R(q_{i+1})(0,0,-l/2))   # 3 rows: stretch + shear
C_dar = Im(q_i^-1 . q_{i+1}) - u_rest_i                                # 3 rows: bend + twist
```

`C_str` says the two endpoints of an edge, carried out along their own frames, must meet.
`C_dar` is the discrete **Darboux vector**, the rotation from one frame to the next; driving it
to `u_rest` is what gives the rod a rest shape. A straight rod has `u_rest = 0`; writing a
non-zero value onto the last few edges is how the steerable tip is modelled, as a rest shape
rather than an applied force.

## The solve

The integrator is **XPBD** (extended position-based dynamics). Each substep predicts, projects
onto the constraint manifold, and differentiates positions back into velocities:

```
S^h(X)                                            # one substep, h = dt / num_substeps
1 predict      v <- (1-d)(v + h(M^-1 f + g)),  x* <- x + h v
               w <- (1-d)(w + h I^-1 tau),     q* <- normalize(q + (h/2)(w,0) . q)
               x*_i <- x_i, v_i <- 0   where 1/m_i = 0        # kinematic particles
2 pre-project  x* <- Pi(x*)                        # optional, stage-dependent
3 compliance   a_str  = 1e-10                                  # near-inextensible
               a_bend = 1/(E b l h^2 + eps),  a_twist = 1/(G b_z l h^2 + eps)
4 solve        x*, q* <- C(x*, q* ; a)
5 post-project x* <- Pi(x*)
6 integrate    v <- (x* - x)/h,  x <- x*
               w <- 2 Im(q* . q^-1)/h,  q <- q*
```

Two properties of this are worth calling out because they drive most of the engineering.

**Compliance scales with `1/h²`.** Stiffness is therefore set by the *substep*, not by the
control rate. Changing `num_substeps` silently changes how stiff the catheter is, which is why
substep count is a physical parameter here rather than a quality knob.

**The constraint system is solved directly, not iteratively.** Each edge couples only its two
adjacent particles, so the system is block-tridiagonal in 6×6 blocks and is solved by a block
Thomas recursion (`solver_backend="block_thomas"`, with `split_thomas`, `block_jacobi` and
`banded_cholesky` available). A direct solve is what lets a near-inextensible rod hold its
length in a handful of substeps where Gauss-Seidel at the same budget would visibly stretch
under a push.

The direct solve also explains a failure mode we hit repeatedly: **it lands exactly on the
constraint manifold, so it overwrites anything written before it.** Any projection applied
pre-solve is a suggestion; only post-solve projections survive.

## Catheter-specific layers

`CathRodSolver` subclasses the generic `XPBDRodSolver` and adds the vasculature behaviour.

### Vessel containment (`Pi_M`)

A signed-distance query against a BVH of the vessel surface, pushing particles back inside a
clearance shell at `-r`, with outward motion forbidden:

```
phi = sigma|x*_i - c|,  n = sigma(x*_i - c)/|x*_i - c|      # c = closest surface point
if phi > -r:
    p = x*_i - n(phi + r)
    if <p - x_i, n> > 0:  p <- p - n<p - x_i, n>            # no outward push
    x*_i <- p
```

### Track guidance (`Pi_A`)

A soft blend of the non-tip particles toward an insertion path, leaving the distal tip free to
deform. Originally a straight ray, which is why it was disabled for curved anatomy; a polyline
variant following the centerline was added later.

### Distance cleanup

Applying containment *after* the solve keeps the wire inside the lumen but mangles edge lengths,
because the projection moves particles off the manifold the solve just put them on. Applying it
*before* preserves lengths and lets the wire leave the vessel. Neither substeps, bend stiffness,
nor an under-relaxation factor resolved this, and Mosaic Intelligence documented the same
trade-off independently.

The fix is a two-colour Gauss-Seidel distance pass run *after* containment, restoring edge
lengths without undoing the containment. The solver ships it off
(`containment_cleanup_iterations = 0`); `CatheterRodSpec` in
`arena/i4h_arena/medical/newton_catheter_physics.py` turns it on at 32 sweeps with relaxation
1.0. Measured effect: chord lengths went from 7–364% of rest to 100–112%, eliminating the
visible bunching and fold-backs, at the cost of worst-case wall penetration rising from
1.5 mm to 3.2 mm.

### Vessel deformation

The vessel is not a static mesh. `CenterlineVesselRuntime` models it as **branching Cosserat
rods along the vessel centerlines** — the same formulation as the catheter, applied to the
anatomy. Contact corrections are split between wire and wall by `vessel_collision_response`, so
the wall is compliant rather than rigid. The remaining gap is visual: the rendered surface is
not yet skinned to the deformed centerlines.

## The Isaac Lab integration

Isaac Lab's `NewtonManager` drives a Newton `SolverBase` and nothing else. Everything below
exists to present the catheter solver as one, without giving up any of the behaviour above.

There are four layers, each with a single job.

### 1. `rod_builder` — put particles on the model

Isaac Lab owns `ModelBuilder.finalize()`, so the rod's particles must be registered *before*
finalize, from scene setup:

```python
particle_range = add_catheter_rod_to_builder(builder, rod_config, num_envs=n)
NewtonXPBDRodManager.register_rod(particle_range)
```

**Only particles are added.** Orientations, rest Darboux frames, per-edge stiffness and
constraint topology stay inside `CathRodSolver`, so the rod needs no Newton custom attributes
and its entire Newton-side footprint is a contiguous slice of the particle arrays.

Two details keep the two representations consistent: particle mass is computed as
`π r² l ρ` by the same formula the solver uses, and `lock_root` gives particle 0 zero mass,
which is how Newton marks a particle kinematic — matching the solver's own locked root.

### 2. `NewtonCathRodSolver` — the `SolverBase` adapter

Implements Newton's contract, a single `step(state_in, state_out, control, contacts, dt)`:

```python
self._bridge.read_from(state_in)    # mirror Newton particles into the rod
self._bridge.advance(dt)            # run the rod's own substeps
self._bridge.write_to(state_out)    # mirror back
```

`control` and `contacts` are deliberately unused. Rod actuation goes through the solver's own
proximal-control API, and vessel contact is resolved by the containment kernels, which never
read Newton's contact buffers.

It also implements `reset(state, world_mask, flags)`, which is what Isaac Lab's
`_reset_solver_internals` hook calls, so per-env reset works without scene-side code.

### 3. `NewtonXPBDRodManager` — the Isaac Lab manager

Satisfies the solver-manager contract by building the solver and setting two flags:

- `_use_single_state = False` — rod dynamics read one state and write another.
- `_needs_collision_pipeline = False` — containment runs in the rod's own kernels, so Newton's
  shared contact pipeline is dead weight for a catheter-only scene.

Because vessel meshes and centerline runtimes are GPU buffers that cannot live in a config, a
scene needing them hands over a pre-built rod instead:

```python
NewtonXPBDRodManager.register_rod(particle_range, rod=CathRodSolver(...))
```

### 4. `NewtonCoupledMJWarpXPBDRodManager` — rigid plus rod

Follows Isaac Lab's existing MJWarp + VBD coupled-manager pattern: nested solver configs, and
the manager owning substep order.

```
1. Clear rigid / particle force accumulators
2. Apply the lagged catheter reaction        (two_way only)
3. Run the Newton collision pipeline         (shared contacts)
4. Step MJWarp rigid solver   (state_0 -> state_1)
5. Step the catheter rod      (state_1 -> state_1)
6. Harvest the catheter reaction             (two_way only)
```

Two ordering constraints are load-bearing. The reaction must be applied *after* the clear in
step 1, or it would be zeroed before the rigid solver reads it. And the rod reads and writes
`state_1` in place — the same buffer MJWarp just wrote — so particle state stays coherent
without a second ping-pong copy.

Here `_needs_collision_pipeline` flips back to `True`, since the rigid bodies genuinely need
Newton contacts even though the rod does not.

## Configuration surface

`XPBDRodSolverCfg` is a normal Isaac Lab `configclass`, dispatched by `class_type` string so
nothing imports the solver package until it is needed.

| Field | Default | Note |
|---|---|---|
| `solver_backend` | `block_thomas` | direct block-tridiagonal solve |
| `young_modulus` | `1.0e9` | Pa |
| `bend_stiffness` / `twist_stiffness` | `0.1` / `0.4` | multipliers, not moduli |
| `density` | `7800.0` | kg/m³ |
| `radius` | `0.002` | m |
| `num_segments` / `segment_length` | `24` / `0.02` | rest layout |
| `num_substeps` | `1` | multiplies with the outer runtime's substeps |
| `linear_damping` / `angular_damping` | `0.01` | per substep retention, not a physical coefficient |
| `track_enabled` / `collision_enabled` | `False` / `False` | projection policy |
| `sync_from_state` / `sync_velocities` | `True` / `True` | respect resets and other solvers |

`CoupledMJWarpXPBDRodSolverCfg` nests that alongside `MJWarpSolverCfg` and adds
`coupling_mode` (`one_way` / `two_way`), the drive body (`drive_body_name` preferred over
`drive_body_index`, which silently means a different link if scene body order changes),
`drive_reaction_relaxation`, and the soft contact material.

## Constraints worth knowing before you change something

**Substeps multiply.** An outer runtime calling `step` `n` times per frame with the rod
configured for `m` substeps runs `n·m` rod substeps — and since compliance goes as `1/h²`, that
changes the effective stiffness. Keep `num_substeps = 1` when the outer `NewtonCfg` already
substeps.

**CUDA graph capture cannot nest.** The rod's internal capture is disabled by default because
Isaac Lab captures the whole step itself. Separately, a centerline vessel feeds host scalars to
the solver, which capture would bake in — so combining a vessel with `use_cuda_graph=True`
raises up front rather than silently ignoring every later tuning change.

**Devices must match.** The rod solver and the Newton model must be on the same Warp device;
the constructor rejects a mismatch rather than copying across.

**Single environment.** `CathRodSolver` implements containment and track guidance on its
single-environment substep path only, and raises rather than silently dropping those
projections when asked for a batch. Vectorized data generation needs the batched path finished
first.

## How it runs in the workflow

The arena layer nests the rates exactly rather than approximately:

| Period | Value | Set by |
|---|---|---|
| Control step | 1/30 s | `decimation = 4` over `sim.dt` |
| Physics step | 1/120 s | `sim.dt` |
| XPBD substep | 1/480 s | `solver_substeps = 4` |
| Fluoroscopy frame | 1/15 s | `FluoroscopySensorCfg.update_period` |

One action is clamped once per control step and applied on all four physics steps, each of which
subdivides into four XPBD substeps — sixteen substeps between consecutive actions. Imaging is
pulled rather than pushed, so a frame is never composed from a stale polyline and a current
C-arm pose.

`arena/i4h_arena/runner.py` is the only caller of `env.step`, and it invalidates the shared read
cache immediately afterwards. That is what makes a recorded step self-consistent: the image, the
action and the joint state describe one simulator state rather than three a frame apart.

For the `s0011` twin this instantiates a 303.2 mm rod of 40 segments (`l = 7.58 mm`,
`r = 0.5 mm`) against a 431×311×311 grid at 1.5 mm isotropic spacing.
