# Catheter Navigation: Reward, Success, and Failure

The exact reward specification for `endoluminal_navigation`, what terminates an episode
successfully, and what does not terminate it at all. Implementation lives in
`arena/i4h_arena/medical/navigation_reward.py`, weights in
`arena/i4h_arena/envcfg/endoluminal_navigation.py`, and the success criterion in
`arena/i4h_arena/medical/navigation_goal.py`.

## Shape

Per environment, per control step at 30 Hz. Total reward is the weighted sum of seven
terms, all computed batched in torch from the Newton rod's `positions_world_m` and the
scene's route polyline.

| Term | Function | Weight | Range/step | Units |
| --- | --- | --- | --- | --- |
| `progress` | `route_progress_reward` | +150 | [-2.5e-3, +2.5e-3] | m of arc |
| `approach` | `approach_reward` | +3.75 | (-1, 1) | dimensionless |
| `arrival` | `arrival_reward` | +5 | {0, 1} | indicator |
| `lateral` | `lateral_offset_penalty` | -2 (0 without radii) | [0, inf) | dimensionless |
| `penetration` | `wall_penetration_penalty` | -200 (0 without radii) | [0, inf) | m |
| `fold` | `fold_penalty` | -1 | [0, inf) | dimensionless |
| `action_rate` | `base_mdp.action_rate_l2` | -0.01 | [0, inf) | dimensionless |

### 1. `progress` — potential-based arc closure

Project the tip onto the route polyline by nearest-point over all segments, giving arc
coordinate `a_t` and remaining arc `R_t = max(0, L - a_t)`. Then

```
r_prog = clamp(R_{t-1} - R_t, -delta, +delta),   delta = 2.5 mm
```

Three properties are load-bearing.

It pays the **difference**, not the level, so it is a potential function `Phi = -R` and
provably leaves the optimal policy unchanged.

It is **symmetric** — backing out costs exactly what advancing the same distance pays. A
one-sided clamp would pay more for a round trip than for standing still.

`delta` is the **physical ceiling**: insertion clamps at 0.05 m/s
(`CatheterDriveSpec.max_insertion_velocity_mps`) and controls advance at 30 Hz, so 1.67 mm
per step, plus margin for the tip travelling further than the root it is fed from while the
shaft straightens. The clamp exists because nearest-point projection onto a route that
doubles back is not continuous in tip position: on a recorded s0011 episode the projected
arc jumped by up to 31 mm across five steps and six times over the episode, crediting
153 mm of travel against 112 mm actually made. Unclamped, that surplus is free return for
wiggling the tip across the arch rather than advancing through it.

Reset must call `reset_route_progress`, which sets the stored remaining arc to NaN.
Without it the first post-reset step differences the new episode's remaining arc against
the old one's, which on a successful reset is the whole route and would pay out the entire
task for doing nothing. NaN maps to zero.

Over the 0.66 m s0011 route a full traverse pays about `150 * 0.66 = 99`.

### 2. `approach` — fine-scale terminal guidance, also potential-based

```
Phi_t = exp(-d_t / sigma),   sigma = 25 mm
r_app = Phi_t - Phi_{t-1}
```

with `d` the straight-line tip-to-target distance. Remaining arc goes flat once the tip is
within one route sample of the end, so the progress term carries no gradient across exactly
the last centimetre the tolerance is judged on. This is the fine-scale companion, mirroring
the two-scale position reward the ultrasound probe reach task uses.

Like `progress` it pays the **difference**, for the same reason and after the level version
was priced. At the old weight of 1 the level form paid up to 1 per step for hovering, and
hovering has no end: at `gamma = 0.995` the best stationary spot just outside the tolerance
discounted to about 150, against about 84 for holding the arrival and terminating.
Finishing the task was a pay cut. A smaller weight does not fix that — any positive
per-step payout for *being* somewhere is collectable forever, and forever beats a bonus
that terminates. Differenced, a stationary tip earns exactly zero wherever it parks, and
the only way to collect is to close distance.

The weight is derived, not chosen: `APPROACH_WEIGHT = PROGRESS_WEIGHT * sigma = 3.75`. The
potential's slope at the target is `1/sigma` per metre, so this makes the last millimetre
pay what `progress` pays for a millimetre of arc — 0.147 against 0.150 — and the two scales
hand off without a cliff where `progress` goes flat. Tying it to the progress weight in
code keeps the handoff from drifting when either is retuned. Note the whole-episode
payout now telescopes to at most `3.75`, down from a possible 600; `approach` is a
gradient, not a bank.

Shaping is applied undiscounted, where strict policy invariance wants `gamma * Phi' - Phi`.
The omission leaves a residual per-step payout of `w * (1 - gamma) * Phi`, about 0.019 at
the target against the 5.0 that `arrival` pays there, so it cannot recreate the inversion.
Taking `gamma` as a term parameter was the alternative and is worse: a second copy of the
trainer's discount, free to drift from it, and a wrong `gamma` breaks the very invariance
it would be added to guarantee.

Reset must call `reset_approach_potential`, for the reason `progress` needs its own hook:
an episode ending at the target and resetting to the vessel entry would otherwise
difference a potential near 1 against one near 0 and bill the agent the entire approach for
the reset itself.

### 3. `arrival` — per step, not terminal

```
r_arr = 1[d <= tau]
```

Paid on every step inside the tolerance rather than once at termination. A single terminal
bonus leaves the fifteen hold steps unpaid, and an agent that has already banked the
approach has no reason to spend them. Fifteen steps at weight 5 is 75.

### 4. `lateral` — off-axis, charged separately from short-of-end

```
p_lat = max(0, ell - f * rho(s)) / rho(s),   f = 0.5
```

`ell` is the tip's perpendicular distance to the route and `rho(s)` the lumen radius at the
projected segment. The inner half of the lumen is free, so hugging the inside of a curve —
what a real wire does — costs nothing. Normalized by the local radius because the same
absolute offset is harmless in the aorta and against the wall in a branch.

The term exists because of a measured failure: an episode that never arrived ended with
**7.1 mm of arc remaining and 7.2 mm of lateral offset inside a 4.5 mm-radius lumen** —
nearly all the length, none of the alignment. With arc progress as the only positive term,
that episode scores as near-total success.

It is a penalty rather than a reward for being centred, because a per-step payout for
sitting on the axis is collectable without going anywhere.

### 5. `penetration` — mean depth outside the wall

```
p_pen = mean_i max(0, ell_i - rho(s_i))
```

over **every rod particle**, not just the tip: a tip that threads the arch while the shaft
behind it cuts the corner is the failure this is for. Depth rather than a count, so easing
off a deep contact pays before the contact clears.

### 6. `fold` — mean excess curvature

Circumradius `R_i` of each consecutive node triple, matching
`newton_catheter_physics.bend_radii_m` so a reward and a diagnostic cannot disagree about
what counts as a fold. Then

```
p_fold = mean_i max(0, R_fold / R_i - 1),   R_fold = 10 mm
```

Dimensionless, so it does not change meaning if the segment count changes, and continuous,
which the boolean version it replaces was not: a boolean fold flag fires on roughly nine
frames in ten of a recorded episode, which makes it a constant offset the advantage
estimator subtracts away rather than a gradient pointing anywhere. The s0011 route's own
tightest curve is 13.1 mm, so anatomy cannot trip the threshold.

### Degenerate states

Every term returns exact zeros when the tip is non-finite (before Newton finalizes its
model there are no particles and the tip reads as infinite), when rod positions are
non-finite, or when the scene supplies no `lumen_radii_m`. In that last case `lateral` and
`penetration` are wired at zero weight rather than dropped, so the term set keeps the same
shape across scenes and a log comparing two runs lines up.

## Success

```
counter = where(d <= tau, counter + 1, 0)
success = counter >= 15
```

A hard reset-to-zero counter, not a running average.

`d` is **straight-line** tip-to-target distance, deliberately not arc. Remaining arc is
only defined while the projection is unambiguous, and making success depend on it would let
a tip that wandered off the route end an episode on a guess. On the shipped s0011 aorta the
route's closest approach to its own endpoint from elsewhere is 44 mm, well outside the
tolerance, so the arch doubling back cannot satisfy the straight-line test early.

`tau` defaults to 5 mm and is overridable up to 20 mm through `I4H_CATHETER_ARRIVAL_MM`.

Fifteen consecutive steps is 0.5 s of simulation time, and roughly 26 s of wall clock at the
1024x1024 DRR render rate. It keeps a tip that merely swings through the target from
registering as arrival.

`reset_arrival_progress` clears the counter as a `mode="reset"` event.

## Failure

There is exactly one failure termination: `time_out` at the scene's `max_steps = 600`.

No wall-contact termination, no fold termination, no out-of-lumen termination. All safety
is priced into the reward and never terminated on. Every failed teleoperated attempt logs
the same line:

```
step=600 workflow_finished (workflow exceeded max_steps=600)
```

The consequence for RL is that the episode is 600 steps regardless, so a policy cannot
shorten its exposure to penalties by failing fast. The only way to stop accumulating
`penetration` and `fold` charges is to reach the target and terminate early, which is
itself the alignment mechanism.

## Open questions before training

A cross-term reading of these weights, including three problems that follow from the
constants alone, is in
[catheter-navigation-reward-analysis.md](catheter-navigation-reward-analysis.md). The
shortest version: `penetration`'s mean over 121 particles dilutes tip perforation to
roughly one per cent of a traverse, and the clamp on `progress` costs the potential-shaping
guarantee this document claims for it. The third problem it raises — `approach` being
collectable by standing still and out-paying arriving — is the one fixed above.

**The weight ordering is the design intent; the numbers are not tuned.** Arriving worth
more than traversing, traversing worth more than any amount of loitering. Over the 600-step
cap and the 0.66 m route: a full traverse pays about 99 through `progress`, the fifteen-step
hold pays 75 through `arrival`, and a tip pinned against the wall gives up roughly 1.0 per
step across `lateral` and `penetration`.

Loitering being worth nothing is now structural rather than a property of those numbers.
Every positive term is either a difference of a potential, which a stationary tip cannot
collect, or `arrival`, which is gated on the tolerance and terminates. No choice of weights
reintroduces a payout for holding still short of the target.

**`penetration` at -200 is the term most likely to need moving.** A 1 mm mean penetration
across the whole rod costs 0.2 per step, so 120 over a full episode — larger than the entire
traverse payout of 99. The mean over all particles also dilutes localized contact heavily,
so the effective magnitude depends strongly on how much of the rod is inserted. That
interaction is not obviously stable across an episode.

**Reward hacking is the live risk in a medical task.** This is why the trocar profile pairs
PPO with a KL penalty against the reference SFT policy. The reward function is not the only
safety mechanism; the anchor to demonstrated behaviour does real work alongside it, and its
quality is exactly the quality of the demonstrations.

**Sequencing.** Tune this after IL is rollout-validated, not before. Tuning a reward against
an IL policy that already mostly works is far easier than debugging a reward and a
from-scratch policy simultaneously.
