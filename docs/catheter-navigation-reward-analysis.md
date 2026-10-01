# Catheter Navigation: Reward Analysis

A technical reading of the reward specified in
[catheter-navigation-reward.md](catheter-navigation-reward.md): what each term does
mechanically, what design pathology it defends against, and three cross-term problems the
specification does not cover. Nothing here has been observed in a training run, because no
catheter policy has been trained yet. All of it follows from the constants and the
composition rule, and all of it is checkable without a GPU.

## The contract the reward operates under

Every term is a pure function `f(env) -> Tensor[num_envs]` registered as an IsaacLab
`RewardTermCfg`. The manager composes them as

```
value  = func(env, **params) * weight * dt        # dt = 1/30 s
reward = sum over terms of value
```

That `dt` multiply matters when reading the specification's arithmetic. Every absolute
figure in it -- a traverse pays 99, the hold pays 75 -- is the pre-`dt` number. In the
buffer the policy is trained on they are 3.3 and 2.5. Because `dt` is a common factor, all
the ratios the specification argues from survive unchanged; only the absolute scale is 30x
smaller than stated.

The state the reward reads is not the policy's observation. It reads
`env.scene["catheter"].data.positions_world_m`, the raw Newton particle buffer -- 121
particles for the `DEFAULT_n = 120` segments -- plus the route polyline and lumen radii
bound into the term params when the scene is built. The reward therefore has privileged
access to ground truth the policy never sees. That is normal for simulated RL, but it means
reward and observation can disagree about the world without anything failing loudly.

## Term 1: `progress`, the potential-based one

`project_to_route` computes, for every route segment at once,

```
frac_i = clamp(dot(p - start_i, span_i) / |span_i|^2, 0, 1)
proj_i = start_i + frac_i * span_i
i*     = argmin_i |p - proj_i|
arc    = start_arc[i*] + frac_i* * |span_i*|
```

`start_arc` is the cumulative-length prefix, built once and cached per device. The
zero-length guard (`safe = where(len^2 > 0, len^2, 1)`) is there because extracted
centerlines contain duplicated samples, and a NaN would win the `argmin` outright rather
than losing it.

Remaining arc is `R_t = max(0, L - arc)`, and the reward is the difference `R_{t-1} - R_t`
clamped to +/- 2.5 mm.

### Why a difference rather than a level

Ng, Harada and Russell's shaping theorem says that adding `F(s, s') = gamma * Phi(s') -
Phi(s)` to a reward leaves the set of optimal policies unchanged. With `Phi = -R` and
`gamma = 1` that is exactly this term. The practical consequence is that summing it over an
episode telescopes to `R_0 - R_T`, the arc actually closed, independent of the path taken
to close it.

### Where the clamp breaks that

The specification says the clamp "is what keeps the shaping potential-based". Strictly it
is the opposite: once `|R_{t-1} - R_t| > delta` the term is no longer a difference of a
potential, the telescoping fails, and policy invariance no longer holds. It is still the
right engineering call, because what it defends against is worse.

Nearest-point projection onto a curve that approaches itself is not continuous in the tip
position. Where the aortic arch doubles back, an infinitesimal tip movement can flip the
`argmin` to a segment far away in arc coordinate. Measured on a recorded s0011 episode the
projected arc jumped by up to 31 mm in a single step, six times over the episode, crediting
153 mm of travel against 112 mm actually made. That 41 mm surplus is free and repeatable
return for oscillating the tip across the arch instead of advancing through it. The clamp
caps each jump at the physical ceiling: 0.05 m/s of insertion at 30 Hz is 1.67 mm per step,
plus margin for the tip travelling further than the root it is fed from while the shaft
straightens.

Clamping both directions is load-bearing. A one-sided clamp makes a round trip pay
differently from standing still, which is an arbitrage cycle, and PPO finds those reliably.

### The reset hazard

`REMAINING_ARC_ATTR` lives on the env object and survives episode boundaries. Without
`reset_route_progress` writing NaN into the reset slots, the first step of a new episode
differences the fresh `R_0` against the previous episode's final `R_T`, which on a
successful reset is nearly the whole route -- paying out the entire task for one step of
doing nothing. This is the same class of defect as per-environment success state surviving
a reset: the bug is not in the arithmetic but in the lifetime of the buffer it reads.

## Term 2: `approach`, and the problem with it

> **Resolved.** The first of the three fixes proposed below was taken: `approach_reward`
> now pays `Phi_t - Phi_{t-1}`, the weight moved from 1 to 3.75 so the fine scale still
> hands off cleanly from `progress`, and `reset_approach_potential` clears the stored
> potential on reset. The analysis is kept as written because the arithmetic below is why
> the change was made. See `catheter-navigation-reward.md` for the current specification.

```
r_app = exp(-d / 0.025)
```

The stated rationale is sound. `R_t` saturates once the tip is within one route sample of
the endpoint, so `progress` has no gradient at all across the last centimetre -- which is
exactly the region the tolerance adjudicates. A second, finer-scale signal is genuinely
needed there.

But this term is a level, not a difference, and it is unbounded in time. That makes it
collectable by standing still, and the numbers are not close. Post-`dt`, over a 600-step
episode:

| Behaviour | `progress` | `approach` | `arrival` | Total |
| --- | --- | --- | --- | --- |
| Arrive at step 100, hold 15, terminate | 3.3 | 0.44 | 2.5 | **6.2** |
| Stall at `d = 6 mm`, just outside a 5 mm tolerance, for 500 steps | 3.3 | 13.1 | 0 | **16.4** |

Loitering just outside the tolerance pays about 2.6 times what succeeding pays. The success
termination is what closes the trap: arriving ends the episode and truncates the `approach`
stream, and under GAE with zero bootstrap on a non-timeout terminal the agent forfeits every
remaining step of reward by completing the task.

The pathology does not require precision. At `d = 25 mm`, one `scale_m` out and nowhere near
the target, parking for 500 steps yields 6.1 -- still more than double the entire arrival
bonus.

Three fixes are available and they are not exclusive. Make the term potential-based as well,
paying `exp(-d_t/sigma) - exp(-d_{t-1}/sigma)`, which telescopes and cannot be farmed. Gate
it on the hold counter so it only pays while the agent is actually committing to an arrival.
Or bootstrap the value function on success termination so that exiting early is not itself
penalised. The first is the smallest change and preserves the reason the term exists.

## Term 3: `arrival`

Paid per step inside the tolerance rather than once at termination, and the reasoning is
credit assignment. The success criterion demands fifteen consecutive in-tolerance steps, and
a single terminal bonus makes each of those fifteen steps individually worthless. An agent
that has already banked the approach has no local reason to spend them.

## Term 4: `lateral`

```
p_lat = max(0, ell - 0.5 * rho(s)) / rho(s)
```

This exists because arc position and axial alignment are independent coordinates and
`progress` measures only the first. The failure that motivated it: an episode ending with
7.1 mm of arc remaining and 7.2 mm of lateral offset inside a 4.5 mm-radius lumen. Under
`progress` alone that scores as roughly 99 per cent of the task completed, when the tip was
in fact outside the vessel and had stopped being steerable.

The inner half of the lumen is free because a real guidewire tracks the inside of a curve,
and penalising that would train an unphysical centring behaviour. Normalizing by the local
radius matters because 3 mm of offset is nothing in the aorta and is wall contact in a
branch. It is a penalty rather than a centring reward for the same reason `approach` is
dangerous: a per-step payout for being on-axis is collectable without going anywhere.

## Term 5: `penetration`, and the dilution problem

**Resolved.** The aggregation below has been replaced by a `max` over particles; the analysis
is kept because it is the derivation of the weight's meaning. See the end of the section.

```
p_pen = mean over all 121 particles of max(0, ell_i - rho(s_i))    # former behaviour
```

The specification flags the `-200` weight as the term most likely to need moving, on the
grounds that it may be too strong. The arithmetic says the opposite for the failure mode
that actually matters.

Take the tip poking 1 mm through the wall with the other 120 particles inside the lumen:

```
mean depth    = 0.001 m / 121   = 8.3e-6 m
per-step cost = 200 * 8.3e-6    = 1.65e-3      (pre-dt)
over 600 steps                  = 0.99
```

Against a traverse worth 99, that is a one per cent penalty for driving the tip through
tissue for an entire episode. To reach the "roughly 1.0 per step" the specification quotes
you would need all 121 particles simultaneously 1 mm outside the vessel -- the whole
catheter outside the anatomy, not a perforation.

The mean is the problem, not the weight. Localized penetration, which is the clinically
meaningful event, is diluted by the particle count, and the dilution factor changes with
insertion depth because particles still outside the patient contribute zero. A `max` over
particles, a sum, or a mean restricted to inserted particles would each behave more sanely.
As written the weight is also not transferable across scenes with a different segment count.

The `max` was taken. It needs no estimate of how many particles are inserted, keeps the
weight meaningful across rods with different particle counts, and is continuous in the
particle positions. Under it the same 1 mm perforation costs `200 * 0.001 * 600 = 120` over
a full episode, against a traverse worth 99 and an arrival bonus of 75 — so perforating
throughout is no longer compatible with a winning episode. The trade is that extent no
longer registers: one particle 2 mm out scores the same as twenty. Both are already serious,
a shaft cutting a corner still registers at its deepest point, and `fold` prices distributed
deformation, so this is the preferable direction to lose information in. The consequence to
watch is that the term now dominates when it fires, and its balance against `lateral` has
not been retuned since the change.

## Term 6: `fold`

```
R_i    = (|a| * |b| * |c|) / (2 * |cross(a, b)|)      sides of each node triple
p_fold = mean over interior nodes of max(0, 0.010 / R_i - 1)
```

Sharing `bend_radii_m` with `newton_catheter_physics` is not cosmetic. It guarantees the
reward and the recorded diagnostic cannot disagree about what a fold is, which is what keeps
a post-hoc analysis of a training run valid.

The continuity argument is the important one and it generalizes. The boolean predecessor
fired on roughly nine frames in ten of a recorded episode. A binary term with ninety per cent
duty is, to the advantage estimator, approximately a constant: GAE subtracts a baseline, and
a near-constant contributes almost nothing to the advantage. It looks like it is doing safety
work and is in fact invisible to the gradient. The `relu(R_fold / R - 1)` form is unbounded
above, so a hard crease produces a large signal, a gentle bend produces none, and the value
varies frame to frame.

`fold` carries the same `1/(n-2)` mean structure as `penetration`, but there the
dimensionless normalization is defensible, because a fold is a distributed property of a
region of the rod rather than a point event.

## Success and failure

The hard reset-to-zero counter means the fifteen steps have to be consecutive, so a tip
chattering across the tolerance boundary never accumulates. Using straight-line rather than
arc distance avoids ending an episode on an ambiguous `argmin`, and the specification
verifies the margin for that choice: the route's closest non-terminal approach to its own
endpoint is 44 mm, far outside any tolerance worth setting.

There is one failure termination, `time_out` at 600 steps. The specification frames this as
an alignment property -- a policy cannot shorten its exposure to penalties by failing fast,
so the only way out is to succeed. That is correct in isolation. Combined with `approach` it
produces the inversion above: the only early exit is success, early exit forfeits reward, and
so the reward structure quietly makes success the second-best outcome.

## Summary of the three gaps

The specification is accurate and unusually well-justified term by term; every design choice
is tied to a measurement rather than a convention. The gaps are all cross-term.

1. ~~**`approach` is farmable and beats success.**~~ **Fixed.** Stalling just outside the
   tolerance paid 2.6x what arriving paid, because arriving terminates the episode and
   truncates the stream. The term is now potential-based, so a stationary tip collects
   nothing wherever it parks. The other two options — gating on the hold counter, or
   bootstrapping the value on success termination — were not needed and were not taken.
2. ~~**`penetration`'s mean dilutes the failure it exists to catch.**~~ **Fixed.** Tip-only
   perforation for a whole episode cost about one per cent of a traverse. The term now
   reports the deepest particle instead of the average of all of them, so the same
   perforation costs 120 against a traverse worth 99 and an episode that perforates
   throughout can no longer outscore one that does not. A sum or a mean over inserted
   particles were the alternatives; the `max` was taken because it also makes the weight
   portable across rods with different particle counts and removes the drift with insertion
   depth. The cost is extent sensitivity, which `fold` partly covers.
3. **The shaping theorem does not survive the clamp.** Policy invariance holds only while the
   clamp is slack. This is an acceptable trade against the projection discontinuity, and both
   the specification and the module docstring now state it in that direction. Noted here as
   a standing caveat rather than an open defect.
