# Changes to the catheter vasculature solver

What this branch changes inside
`third_party/i4h-physics-simulation-internal/.../catheter_vasculature_solver`, and
why. Three files, +563 / -40:

| File | Change |
| --- | --- |
| `cath_rod_solver.py` | Length cleanup interleaved with containment, a curved insertion track, and a runtime-steerable tip |
| `vessel_deformation/centerline_containment.py` | Containment gained an inward pull, so it is no longer only a wall test |
| `vessel_deformation/centerline_runtime.py` | Passes the two new containment parameters through to the kernels |

Every change is off by default. The new constructor arguments all default to the
old behaviour, so an existing scene that does not ask for them gets the solver it
had before.

## 1. Containment pulls inward as well as pushing out

**Before.** Containment was a one-sided radial test. If a particle's surface was
through the vessel wall it was projected back in; if it was anywhere inside the
lumen, nothing acted on it at all.

**The problem.** "Anywhere inside" is a lot of freedom. Nothing in the solve
preferred the vessel axis, so any spare arc length in the rod was free to bow
sideways until it found a wall. That is the shaft bunching and the long stretched
gaps seen in the viewport.

**Now.** A single `wp.func` decides the correction:

```python
@wp.func
def _containment_pull(gap, free_radius, interior_deadband, interior_stiffness) -> wp.vec2:
    if gap > 0.0:
        return wp.vec2(gap, 1.0)          # outside the wall: unchanged, full projection
    if interior_stiffness <= 0.0:
        return wp.vec2(0.0, 0.0)          # inert default: the old one-sided behaviour
    deadband = interior_deadband * free_radius
    distance = gap + free_radius
    if distance <= deadband:
        return wp.vec2(0.0, 0.0)          # inner lumen is left to the solve
    return wp.vec2((distance - deadband) * interior_stiffness, 0.0)
```

`gap` is the distance from the axis less the radius the wire is free to occupy,
so it is positive once the wire's surface is through the wall and negative while
the wire is inside.

Two new parameters control it, both fractions in `[0, 1]`:

- `interior_deadband` — how much of the free radius the wire may occupy before
  the inward pull starts. `1.0` puts the deadband at the wall, which is the old
  behaviour.
- `interior_stiffness` — how hard it pulls. `0.0` disables it entirely.

The deadband is the point: the inner lumen stays somewhere the wire is allowed to
be, so its shape there is still the solve's answer rather than the centerline's.
This is a regulariser, not a prescribed route. At unit stiffness the sample lands
exactly on the deadband surface, so no stiffness in `(0, 1]` can carry it past
that surface, let alone onto the axis.

**One correctness detail worth knowing.** An interior pull is *not* a contact. It
does not push the vessel back in two-way mode, and it is excluded from
`contact_depth` and `contact_count`:

```python
reciprocal = float(0.0)
if two_way != 0 and is_contact > 0.0:
    reciprocal = vessel_response
...
if is_contact > 0.0:
    wp.atomic_max(contact_depth, env, best_gap)
    wp.atomic_add(contact_count, env, 1)
```

Without that, a wall deformed by the regulariser would report a contact that
never physically happened, and the penetration diagnostics would be measuring the
solver's own bookkeeping.

Both the single-rod and batched kernels carry the same change, and
`CenterlineVesselRuntime` clamps both values into range before forwarding them.

## 2. Containment and length cleanup now alternate

**The conflict.** Containment has to run *after* the constraint solve, because
that is the only side of it where a projection persists. But that also makes it
the last word on position, so it overrides the stretch constraints the solve had
just satisfied. Particles slide radially and pile up along the vessel; measured
chords ran from 6% to 362% of rest length.

**The cleanup sweep.** `_distance_cleanup_kernel` re-satisfies the edge lengths
after containment has disturbed them. It runs as two-colour Gauss-Seidel — even
edges, then odd — so no two threads write the same particle while each sweep
still sees its neighbour's result. Jacobi would also be race-free but propagates
a correction only one edge per iteration, and a 40-edge chain does not have that
many to spare.

This works because containment's corrections are radial and small, so restoring
length mostly moves particles *along* the vessel rather than back out through its
wall.

**Why alternating was needed anyway.** Running containment once and then spending
every sweep after it lets the sweeps win outright, because they equalise edge
lengths knowing nothing about where the vessel is. Measured on the s0011 route
with one containment pass followed by 128 sweeps: chords reached a best-ever
100-104% while **13 of 41 particles sat outside the lumen** — the worst
containment recorded on that route.

`_project_containment_and_cleanup` alternates the two over
`containment_cleanup_rounds`. Projecting onto each constraint set in turn
converges toward a point in *both*, rather than to whichever was applied last.
The sweep budget is divided across rounds rather than multiplied by them, so
raising the round count trades sweep depth per round for agreement between the
two constraints and leaves the cleanup cost unchanged; the extra cost is the
additional containment passes. `containment_cleanup_rounds=1` reproduces the
original sequencing exactly.

New constructor arguments: `containment_cleanup_iterations=0`,
`containment_cleanup_relaxation=1.0`, `containment_cleanup_rounds=1`.

## 3. The tip is steerable at runtime, per environment

**Before.** `set_tip_bend(bend_angle: float)` took a single scalar for every
environment and rewrote the whole rest-Darboux buffer, zeroing the non-tip edges.
Any authored rest curvature in the body was destroyed by a steer. It was a
build-time shape, not a control input.

**Now.** `set_tip_bend(bend_angle: Any)` accepts either one value shared across
environments or one value per environment, and a device tensor is used where it
lies rather than copied to the host — so steering every step does not
synchronise.

The buffer is no longer overwritten from nothing. `capture_tip_bend_baseline()`
snapshots the unsteered rest curvature, and the kernel adds the bend to that
baseline on the tip edges while writing the baseline unchanged on the rest:

- The authored body shape survives a steer.
- The angle is **absolute**, measured from the baseline rather than from wherever
  the previous call left the buffer. A controller supplies an angle, not a delta,
  and calling it every step is safe.

The baseline is captured automatically before the first steer, so the common case
needs no extra call. It has to be re-captured if rest curvature is written
straight into the solver afterwards (`seed_rest_curvature_from_path` is the case
that exists), or that shape counts as part of the bend and the next steer undoes
it.

The bend is about the tip's **local X axis**. Aiming it at a particular branch is
the axial rotation command's job, which is also how a pre-shaped wire is aimed in
a real procedure.

## 4. Insertion guidance can follow a curved path

`_track_sliding_kernel` projects non-tip particles onto a straight ray, which
only suits a bench rail. Through a curved lumen it would pull the wire out
through the wall, which is why guidance was otherwise left off for anatomy.

`_track_polyline_kernel` is the curved counterpart: it projects onto a polyline
supplied via the new `track_path` argument, validated by `_validated_track_path`
and uploaded per device by `_track_path_for_device`.

The mechanism is worth stating plainly, because it is not a prescribed route
either. Removing the two lateral degrees of freedom leaves **arc length as the
only way the rod can move**, and the distance constraints already resist
compression along it — so spacing is recovered by the existing solve rather than
dictated by the track. Two properties follow:

- It only works *before* the solve. Run afterwards it becomes the last word on
  position and the chords never re-equalise.
- The projection is a blend weighted by `track_stiffness`, not a snap, so it
  argues with the elastic solve instead of overruling it. `end_idx` leaves the
  distal tip free to deform and steer.

## Known limitation

None of this removes excess arc length; it constrains where the excess can go and
redistributes it. Measured on a live s0011 run, against a 303.2 mm rest length:

| Probe step | Arc length | Excess |
| --- | --- | --- |
| 120 | 315.6 mm | +12.4 mm |
| 2400 | 321.4 mm | +18.2 mm |
| 7200 | 320.7 mm | +17.4 mm |
| 12000 | 320.4 mm | +17.2 mm |

Two things to read off it. Most of the excess is already present by step 120, so
it is injected during seeding and early insertion rather than accumulated
gradually. And it then drains only about 1 mm over the following 9600 steps —
slowly enough to look like nothing is happening, but that drain is exactly what
makes a stationary tip creep backwards from its target.

Pushing the sweep count higher does not fix it — the sweeps move the excess from
the free distal tip into the mid-shaft, so the bow relocates rather than leaves.
Removing it properly means making the containment projection length-preserving,
so arc length is never injected in the first place. That is not in this branch.

The arc figure is reported by `containment_report` in
`arena/i4h_arena/medical/newton_catheter_physics.py` and printed by the
`I4H_CATHETER_PROBE` diagnostic, in millimetres rather than as a chord
percentage. Percentages move whenever the sweeps redistribute the excess and are
not comparable across segment counts, so they can show an improvement that did
not happen.
