# Coupling MJWarp Rigid Bodies to the XPBD Catheter Rod

How the robotic arm and the catheter are simulated together: what is exchanged, in what
order, and why each ordering constraint is not negotiable.

## The problem in one paragraph

A robotic arm and a guidewire do not want to be solved by the same integrator. MJWarp
represents the arm as **rigid bodies in generalized coordinates** and solves them with an
implicit method suited to stiff articulated chains. The catheter is a **Cosserat elastic rod**
represented as particles with orientations, solved position-based (XPBD) with many small
substeps. Merging them into one monolithic system would mean rewriting one solver in terms of
the other. Instead they run **sequentially over a shared state**, exchanging exactly two
quantities: a pose going down to the wire, and a force-and-moment wrench coming back up.

## The one-slide version

```
                 pose (where the mount holds the wire)
   MJWarp arm  ─────────────────────────────────────────>  XPBD rod
      ▲                                                       │
      └───────────────────────────────────────────────────────┘
              wrench (force and moment on the holder)
```

Two directions, two mechanisms:

| Direction | Mechanism | Quantity |
|---|---|---|
| Arm → catheter | `set_root_pose_gpu` | Pose of the proximal particle |
| Catheter → arm | `proximal_wrench` | World force (N) and moment (N·m) at the proximal particle |

The exchange is **lagged**, not simultaneous: the wrench applied this substep was harvested
from the previous one.

## Shared state, not a merged solver

Both solvers operate on the same Newton `Model` and `State`. The rigid solver owns the body
arrays (`body_q`, `body_f`); the rod owns a contiguous slice of the particle arrays. Nothing
is copied between them, so there is one source of truth per degree of freedom and no
synchronization step to get wrong.

## The substep

This is the core of the design. `NewtonCoupledMJWarpXPBDRodManager._step_solver` runs, per
substep:

```python
cls._clear_force_buffers(state_0)              # zero particle_f, body_f, joint_f
cls._apply_soft_to_rigid_reactions(contacts, state_0)   # two-way: inject lagged wrench
contacts = cls._run_collision_pipeline(state_0, state_1, control) or contacts

rigid.step(state_0, state_1, control, contacts, substep_dt)
rod.step(state_1, state_1, control, contacts, substep_dt)   # note: in place

coupler.harvest(state_1, rod.rod)              # two-way: read wrench for next substep
```

Every line's position is load-bearing:

- **Inject after the clear, before the rigid solve.** Land it before the clear and it is
  zeroed before anything reads it; land it after the rigid solve and the arm does not feel it
  until a substep later than intended.
- **The rod reads `state_1`, which the rigid solver just wrote.** It reads and writes the same
  buffer, so particle state stays coherent without a second ping-pong copy. The rod therefore
  sees the arm's *post-solve* configuration, not its stale one.
- **Harvest last.** The multipliers only carry force after a solve has run, so this is the
  earliest point the reaction exists — and it is consumed at the top of the next substep.

## Direction 1: the arm places the wire

`set_root_pose_gpu(positions, orientations)` hands the solver a **pose**, not a velocity. That
is a deliberate statement about what drives the wire:

- The rate-based alternative, `apply_proximal_control_gpu`, feeds the rod at a *speed* along
  its own tangent. It suits a rail or a roller drive, but the root's absolute position then
  accumulates from integration — whatever the mount does, the root only ever hears its speed.
- With a pose, a mount that lags, stalls against a joint limit, or is held back by contact
  stops the wire by exactly as much as it stopped itself.

Two implementation details matter:

**The root is kinematic.** `lock_root` gives the proximal particle zero inverse mass, and the
predictor zeroes the velocity of every zero-inverse-mass particle at the top of each substep.
The root therefore *teleports* to where it is put — it carries no momentum and no timestep is
involved. This is also why the pose command takes no `dt`.

**Commands are latched, not applied.** The next `step()` consumes the most recent command, so
one command per step reaches the solver. This keeps the sequence capturable in a CUDA graph.
It also means a pose command and a rate command contradict each other, and the last one issued
wins.

## Direction 2: the wire loads the arm

This is the harder direction, because the obvious approach is unavailable. The root is held
kinematically with zero mass, so **no momentum accumulates there** and the reaction cannot be
recovered from a velocity difference.

Instead, `proximal_wrench` reads edge 0's accumulated multipliers. The translational
Jacobian on the root is identity, giving `F = lambda_stretch / dt²`. The rotational
Jacobian gives `M = Jrot_root.T @ lambda / dt²`, evaluated at the final material
frames. This includes the bend/twist couple and the stretch constraint's local
lever arm. Both quantities are in world coordinates, with moment measured about
the proximal point. `proximal_reaction` remains available as a force-only API.

The multipliers are retained while contact and global elastic solves alternate
within each rod substep. Their accuracy therefore depends on convergence of that
combined solve. See [rod contact and feedback](rod-contact-feedback.md) for the
material model and physical reference tests.

## Applying the wrench to the right body

`harvest_drive_reaction_kernel` carries the wrench onto the holding body and adds
the force's moment about that body's center of mass:

```python
f   = wp.spatial_top(proximal_wrench[env])
arm = wp.transform_vector(body_q[body], mount_local[env] - body_com[body])
tau = wp.spatial_bottom(proximal_wrench[env]) + wp.cross(arm, f)
wp.atomic_add(out_coupling_forces, body, wp.spatial_vector(f, tau))
```

The subtle and important part is **which point supplies the moment arm**. It is `mount_local`,
fixed in the body's frame — not the rod's proximal particle. Physically, the wire is gripped by
hardware bolted to the body (a hemostatic valve on a flange), and that grip point does not move
relative to the body no matter how much wire has been fed through it.

Using the particle instead ties the moment arm to insertion depth: under a roller drive the
proximal particle travels away up the vessel, so the arm grows without bound and invents a
torque that scales with how much wire has been inserted. Measured on the `s0011` twin, that
error drifted the flange 28 mm off the access site after one second and 34 mm after two,
growing with fed depth.

The accumulation is atomic because several rods may be held by the same body.

## Stability: lagged exchange plus under-relaxation

The exchange is staggered rather than simultaneous — a Gauss-Seidel style pass, matching the
proxy path in Newton's coupled-solver framework. That is cheap and simple, but explicit
coupling of a stiff element to a rigid body is the classic setting for feedback instability.

The control for it is `drive_reaction_relaxation`:

```python
coupling_forces[i] = relaxation * coupling_forces[i] + (1.0 - relaxation) * previous[i]
```

Below 1 under-relaxes and damps the lagged feedback; 1 passes the harvest through unchanged;
above 1 over-relaxes. This mirrors upstream's `proxy_relaxation`.

## Configuration surface

`CoupledMJWarpXPBDRodSolverCfg` exposes these coupling fields:

| Field | Meaning |
|---|---|
| `coupling_mode` | `one_way` or `two_way` |
| `drive_body_name` | Which body holds the proximal end |
| `drive_mount_local` | Grip point in the drive body's local frame, in metres |
| `drive_reaction_relaxation` | Feedback damping, default `1.0` |

**`one_way`**: rigid bodies push the catheter, but never feel it.
**`two_way`**: additionally feeds the reaction into `body_f`, so the arm loads up against the
wire's weight, stiffness, and vessel contact.

In our wiring, `coupled_solver_cfg` selects the mode from whether a drive body was named:

```python
two_way = spec.drive_body_name is not None
"coupling_mode": "two_way" if two_way else "one_way"
```

The body is identified **by name, not index**, resolved against the Newton builder's labels. An
index would silently refer to a different link if the scene's body order ever changed.

The Franka catheter embodiment enables two-way coupling through `panda_hand` and
sets the grip point to `(0, 0, 0.1034)` m in that body's frame. The manager supplies
Newton's authored `body_com` when shifting the wrench.

## Caveats worth stating up front

- **Lagged, not monolithic.** Stability depends on relaxation and substep size rather than
  being unconditional.
- **Contact is approximate at a finite iteration count.** The default uses 32
  alternating contact/elastic iterations; the wrench is not a proof of contact
  convergence or device calibration.
- **The rod does not read Newton's contact buffers.** The reaction transmits through the rod's
  own constraints at the drive point, which is why the coupler ignores the `contacts` argument
  it is handed.
- **Collision pipeline arity.** `CollisionPipeline` is constructed from the model and holds its
  own reference, so it is called as `collide(state, contacts)`. Passing the model again shifts
  every argument by one and raises `TypeError` before any coupled step can run.
