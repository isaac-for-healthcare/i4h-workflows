# START Position Statement: Endovascular Testbeds and Effectiveness Metrics

Summary of Robertshaw et al., *A position statement on endovascular models and effectiveness
metrics for mechanical thrombectomy navigation, on behalf of the Stakeholder Taskforce for
AI-assisted Robotic Thrombectomy (START)*. JAHA 2026;15:e044931 (arXiv:2603.28129v1), with
notes on what it means for validating the catheter navigation workflow.

Endorsed by the Society of Vascular and Interventional Neurology (USA) and the UK
Neurointerventional Group.

## Why the paper exists

A prior systematic review found that AI-driven autonomous endovascular navigation has not
advanced past experimental proof of concept — **technology readiness level 3** — and, more
damagingly, that there are *no* standardized testbeds or effectiveness metrics in the field.
Studies therefore cannot be compared quantitatively. This statement is the consensus attempt
to fix that, built from a 21-expert incubator day followed by a three-round Delphi with 22
panelists and 100% response in every round.

The scope is deliberately narrow: **testbeds and metrics**. It is not a guideline, and the
authors flag follow-up statements on implementation and ethics.

## The two axes

### Testbed environments

Four environments, each with a distinct validation role: **in silico**, **in vitro**,
**ex vivo** (human cadaver), **in vivo** (non-human, typically porcine).

### Complexity levels

Cutting across those environments is a three-level "innovation funnel," so that early work
is not blocked by having to model everything at once:

| Level | Requirement |
|---|---|
| Simple | Realistic vessel anatomy compatible with guidewire and catheter use |
| Standard | Adds **deformable vessels** |
| Complex | Adds blood flow, pulsatility, and disease features such as atheromatous plaques |

The authors explicitly recommend permissiveness at the outset — a "minimum viable product" —
on the grounds that demanding full physiology up front would impede innovation.

### Procedure phases

Metrics are scoped to phases of anterior circulation MT. The **navigation phases** are A and B:

- **A1** primary access, femoral artery to common/internal carotid
- **A2** primary access, radial artery to common/internal carotid
- **B** secondary access, internal carotid to cerebral artery
- **C** treatment (stent retriever or aspiration)
- **D** removal and access closure — consensus that this is only effectively developed
  *ex vivo*

## Consensus for in silico

These are the numbers that matter for a simulator. Percentages are agreement at the round
where consensus was first reached.

**Required experimental factors:** deformable vessels 89%, catheter *and* guidewire together
rather than a single instrument 95%, diseased vessel with plaques 90%, pulsatility 86%,
realistic anatomy 84%, blood flow 82%. Simulated respiration did *not* reach consensus (43%).

**Required effectiveness metrics:**

| Metric | Agreement | Unit |
|---|---|---|
| Success rate | **100%** | % |
| Number of failures (e.g. wrong branch) | 95% | count |
| Fluoroscopy time | 90% | s |
| Contact forces at instrument tip | 89% | N |
| Contact forces on vessel walls | 84% | N |
| Number of phases or steps completed | 84% | count |
| Number of handling errors | 84% | count |
| Procedure time | 85% | s |
| Contact forces at instrument base | 83% | N |
| Path following error | 81% | % |

Success rate — how often the robot reaches the target in a given number of evaluations — was
the single most important metric, reaching 100% at *every* developmental stage.

Notably **excluded** in silico: path length (76%, consensus only in vitro), instrument tip
speed (76%, in vitro only), tip acceleration, contrast volume, and number of guidewire tip
touches on the wall.

For low-TRL work the paper recommends concentrating on success rate, phases/steps, failures,
handling errors, path length, and procedure time, deferring the rest.

## Two caveats the paper raises that matter for reward design

These are the most useful parts of the discussion and both cut against the obvious approach.

**Centerline distance is the wrong reference for path following.** The paper states plainly
that current metrics use the vessel centerline "however, this may not be ideal as the vessel
centerline is not necessarily the best path to take during MT navigation." The recommended
alternative is to pre-record expert navigations and measure deviation against the closest
expert path.

**There is no defensible force threshold yet.** Force measurement is feasible in silico, and
the authors note the advantage that measuring it there does not perturb the device. But
"force measurements may only be considered a useful metric if a consensus existed on what
should be considered excessive," and no such consensus exists because applied force cannot be
measured in patients and tied to safety. Their suggested substitute is again expert-relative:
compare the robot's force profile to an expert demonstrator's. They call correlating in vitro
force to in vivo complication rates the single "requisite patient safety task needed now."

## How this helps validate our workflow

### Where we already conform

Realistic CT-derived anatomy (the `s0011` patient twin) puts us at **simple**, and the
deformable vessel puts us at most of **standard** — the Cosserat centerline vessel runs its
full predict/project/finalize cycle every substep with `two_way=True` and
`vessel_response=1.0`, so the wire genuinely displaces the wall. That is the 89%-consensus
requirement, and it is done. Femoral access maps to **phase A1**.

### The gap that matters most

Success rate is the one metric at 100% consensus at every stage, and we are close but not
there. A real arrival criterion exists and is installed: `reached_navigation_target` requires
the tip within 5 mm of the target for 15 consecutive steps, and
`CatheterEmbodiment.get_termination_cfg` registers it as IsaacLab's `success` termination term
whenever a patient twin supplies a navigation target — the distal end of the centerline.

What is missing is a task that *consults* it. `--mode demo` runs
`basic/catheter_sweep`, whose manifest describes it as exercising insertion, rotation, and
C-arm orbit "for simulator validation," and which contains no goal of its own. The graph is a
bare `.flow(task("basic/catheter_sweep"))` with no success predicate, so the recorded
`success` attribute reflects **graph completion, not arrival**. A headless demo run over 118
steps recorded `success: True` after 3.2 mm of insertion along a route roughly 40 cm long.

So reporting a START-conformant success rate needs a navigation task whose graph success is
the arrival term, and that term recorded per episode — not new arrival machinery. That makes
it the highest-value and among the cheapest of the gaps.

### Gaps in rough order of effort

- **Path following error** — cheap, since the ordered centerline is already on the rod spec.
  But heed the caveat: report it against recorded teleop demonstrations, not the centerline,
  or it measures the wrong thing.
- **Contact forces** — mostly present but unsurfaced. `proximal_reaction` gives the base
  load, and the vessel runtime maintains `contact_depth` / `contact_count` per call. Neither
  is recorded as a metric. Worth noting that the current containment-versus-solve conflict
  makes wall contact forces untrustworthy until fixed, since the wire is 8% overstretched
  with chord lengths between 3% and 380% of rest.
- **Phases and steps, handling errors** — needs procedure segmentation we do not have.
- **Guidewire** — 95% consensus in silico and we simulate one rod, not a catheter over a
  wire. This is the largest single realism gap.
- **Blood flow, pulsatility, plaques** — the **complex** level, and explicitly deferrable
  under the paper's own minimum-viable-product guidance.

### Two incidental alignments

The paper cites synthetic vascular models "derived from 3D spline-based methods" as a way to
generate anatomically varied in silico testbeds — which is precisely the spline-perturbation
approach proposed for anatomical variation, now with an external consensus statement behind
it.

And its expert-relative framing for both path error and force is well matched to what we
have: our imitation data is recorded human keyboard teleoperation, so expert reference
trajectories are a by-product of the data collection rather than extra work.

## Bottom line for the partner thread

The paper does not resolve the substep-sensitivity and contact-response questions raised in
the email — it is a metrics and testbed consensus, not a solver paper. What it does supply is
an externally agreed, society-endorsed definition of what a navigation result must report,
which is exactly what is needed to make two solvers' stability claims comparable. It also
warns against the two most tempting reward shortcuts: centerline distance and absolute force
thresholds.
