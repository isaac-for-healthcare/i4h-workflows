# ADMM coupling: answers from the Newton team

> Implementation update, 2026-09-11: the discussion below describes the previous
> force-only coupling. The current solver recovers the intrinsic root moment
> from its private material frames using `Jrot.T @ lambda / dt²`, without adding
> Newton particle-frame support. The Franka scene already enables `two_way` and
> now supplies the real grip offset and body COM. Thus the earlier claim that
> upstream particle frames are the only route to torque feedback, and the claim
> that no scene enables feedback, do not apply to the current code. The generic
> ADMM attachment limitations discussed here are a separate interface question.
> See [rod contact and feedback](rod-contact-feedback.md).

Summary of a discussion with Gilles Daviet (Newton), with Maximilian Krause, about
whether migrating our catheter/arm coupling to `SolverCoupledADMM` would close the
bending-moment gap, and what we would give up in root placement accuracy.

Two questions were asked. **One answer confirmed our reading. The other showed a
premise in `coupled-solver-conformance.md` was wrong, in our favour.**

## Question 1 — can ADMM transmit the wire's bending moment?

**What we asked.** Our coupling gives the arm the wire's proximal force `F`, plus
the moment that force makes at the grip offset, `r × F`. It does not carry the
*intrinsic bending couple* a clamped rod transmits independently of `F`. Reading
`solver_coupled_admm.py`, the body-particle attachment row carries a body point,
stiffness and damping but no frame or angular stiffness; the angular attachment
groups and `attach_rr_angular_*` kernels are rigid-to-rigid only. Since Newton's
shared particle state is positional, there is no rotational DOF on the particle
side for an angular row to act on.

**Answer.** Confirmed correct. In Newton, cables are modelled as rigid bodies, so
the case for prescribing *particle* frames has never come up. It should be
possible to add, but it does not exist today.

**What this means for us.** The gap is a missing capability in Newton, not a
solver choice we made. Migrating to ADMM would not close it, because an angular
attachment row would have nothing on the particle side to act on. Closing it is
upstream work — particle frames — and Gilles considers it feasible but unbuilt.

Worth being precise about what is actually lost: this affects the **arm's felt
load, not the wire's shape**. The rod still bends correctly under its own elastic
model. What is missing is the couple flowing *back* to the robot. So it matters
for force-controlled arm behaviour, torque-sensor realism and operator haptics —
not for navigation kinematics or insertion depth.

**And it is the second missing term, not the first.** Our `mount_local` defaults
to the body origin, and nothing outside the solver package sets it — no scene
passes a mount offset, and no shipped scene enables `two_way` at all. So today we
transmit a pure force with `r = 0`, meaning even the `r × F` term we *do* support
is unused:

```python
arm = wp.transform_vector(body_q[body], mount_local[env])
tau = wp.cross(arm, f)
```

That default was a deliberate conservative choice — it under-states the moment
rather than inventing one that grows with insertion depth, which is what using
the proximal particle as the moment arm would do (measured: 28 mm of flange drift
after one second, 34 mm after two). But it does mean **setting a real mount
offset is a cheaper fidelity gain than either ADMM or particle frames, and it is
already supported.** That is the first thing to do if the arm's felt load matters.

## Question 2 — do we trade exact root placement for a compliant spring?

**What we asked.** The guidewire root is held kinematically today: zero inverse
mass, so it sits exactly where the gripper puts it with no position error. An
ADMM attachment is compliant, so would we be trading exact placement for a spring
deflection needing stiffness tuning and Baumgarte correction, in exchange for a
symmetric force interface?

**Answer — the premise was wrong.** ADMM constraints are **hard, not compliant**.
It is an augmented Lagrangian formulation, so on convergence the kinematic
constraint is exactly satisfied.

On what the per-row stiffness and damping then do: the constraint *can* be made
compliant, but it does not need to be, and setting stiffness arbitrarily high
works. They are **penalty weights governing convergence, not a physical grip
stiffness.** There is no Baumgarte term; that concern came from a soft-constraint
mental model.

On the fixed iteration count: even with a single iteration, the constraint is
persistent and the Lagrange multipliers are **warm-started**, so the coupling
force resolves across timesteps. How well the constraint holds therefore depends
on how much fast motion you have relative to your tolerance.

## What this changes

**1. Two claims in `coupled-solver-conformance.md` are now wrong.** It says
"ADMM's attachment is compliant, a quadratic penalty with stiffness in N/m and
damping in N·s/m", and that "attachment stiffness needs tuning between a wire
that lags the gripper and ADMM convergence trouble." Neither holds. Un-pinning
the root does not cost us exact placement, and stiffness is not a fidelity knob.

**2. The sub-millimetre insertion-depth requirement is achievable.** The steady
state is exact. Error is a *transient* — it appears only when the commanded pose
moves faster than the warm-started multiplier catches up over successive steps.
Our motion is slow: 9 mm/s commanded insertion at 30 Hz control is 0.3 mm per
step, comfortably inside what a warm start tracks. The real risk cases are
episode resets, teleport-like re-poses, fast retraction, and the first steps
after a re-grip, where the multiplier starts cold.

**3. The recommended prototype in that doc is now a pessimistic proxy.** It
suggests testing an un-pinned root on the current stack by relaxing `lock_root`
and holding the root with a stiff explicit spring. A penalty spring has a
steady-state error proportional to load over stiffness; ALM with warm-started
multipliers drives that error to zero. So the prototype would over-state the
placement error and could talk us out of a migration that is fine. If we run it,
read the result as an upper bound.

**4. Stiffness must not be exposed as a tunable.** If we migrate, set it high and
document it as a penalty weight. Putting it in a manifest invites someone to tune
it as grip physics, which it is not.

**5. The migration buys a symmetric *force* interface and nothing angular.** That
is still worth having — it removes the zero-mass-root harvest problem that
`proximal_reaction` exists to work around — but it should not be justified on
bending-moment fidelity.

## What to watch, and open items

Gilles' answer to "how well is it satisfied in practice" reduces to: measure it,
because it depends on our motion profile. So the acceptance test for any
migration is a **residual metric we do not currently have** — per-step distance
between the wire's proximal node and the gripper mount, logging the peak rather
than the mean, exercised across resets and fast retraction against a sub-mm
budget. This follows the pattern already in place for `I4H_CATHETER_PROBE` and
`I4H_CATHETER_INSERTION`, and it should exist before the migration, not after.

Open items:

- Set a real `mount_local` offset and enable `two_way` in a scene, so the `r × F`
  term we already support is actually used. Cheapest fidelity gain available.
- Add the proximal residual metric.
- File the particle-frame / angular-attachment-to-a-deformable-endpoint request
  upstream with Newton. Not a migration blocker, but it is the only route to the
  intrinsic bending couple, and Gilles indicated it is feasible.
- Correct the compliance claims in `coupled-solver-conformance.md`.
