# Internal endoluminal migration contract

Upstream dependency: [runtime capability draft #14](https://github.com/isaac-for-healthcare/i4h-physics-simulation-internal/pull/14). This proposal was tested against upstream commit `addb3f7`; the six production source pins are deliberately unchanged.

Implementation baseline fetched on 2026-10-12:

- physics upstream main: `7b5cf1c252dd641c1f2827053d3e4020a5528929`
- archived internal workflow main: `fd1e5fb6d9f3c88e99736328e600a08ab4412081`
- archived internal workflow feat/catheter-navigation-reward: `95d0335c4a670fa5ffd8760efc94b45d0de7578c`
- existing component integration reference: workflow `438bb8044bb239ec20800052ef496f1438facdaa`
  (`f6-lab-6864dd1`, I4hComponent); package-import reference `148eb9c6`
  predates that component conversion. Selective reuse only.

Publication base: public `i4h-workflows/main` at
`5e2ebb1730bea235e8aa17cd2d52133cdb9d8e81` (fetched 2026-10-12), which already
contains the navigation stack via PR #245. Only the four proposal files are
applied to this base; no wholesale internal branch merge is included.

These are source baselines, not validated migration revisions. Do not remove the
legacy dependency or advertise deprecation before all release gates below pass.
Existing dirty checkouts and the endoluminal-ui worktree are outside this change.

## Consumer and configuration matrix

All geometry and actions below use metres, seconds, radians and kilograms.
Patient centreline coordinates are millimetres and must pass through the patient
twin's world transform before the component inverse transform. Rotations use xyzw
quaternions; preserving the tangent alone is insufficient (roll is observable).

| Consumer | Placement and guidance | Material and contact | Public contract |
| --- | --- | --- | --- |
| Navigation branch without patient twin | origin (-0.11, 0.04, 0.68), +X; 0.4 m, 120 segments, radius 0.0005; fixed track drive without patient route | shared wire defaults below; no vessel without twin | 3 rates: insertion m/s, rotation rad/s, tip rad/s; reset startup pose and zero commands |
| Route navigation | ordered twin centreline, resampled at 0.0075 m; length max(0.1 route length, route length - 0.12); startup sampled along route; root driven by route arc/roll; guide stiffness 0.35; distal free length 0.18 m converted by int(length/segment), as in the navigation branch | physical inertia; polyline tip; material straight unless explicitly seeded; linear centreline containment, interior deadband 0.5/stiffness 0.25; wall response 0.5; endpoints locked; wall linear/angular damping 0.01 | preserve route arc and applied feed recording; observations fluoroscopy_rgb, tip_position, tip_direction, target_offset, route_state, drive_state; selected reset only |
| Robot-mounted catheter | same patient/world conversion; root owned by panda_hand with local translation (0,0,0.1034); no external-pose owner | existing rigid mount and two-way reaction, one-step delay | existing arm and catheter action columns and joint names; reset selected worlds |
| Fluoroscopy | existing renderer consumes world catheter positions; patient conversion unchanged | shares catheter dynamics; sensor never advances physics | existing image, terminal and recording fields |
| Aorta component | authored case geometry, component pose and case material parameters | existing I4hComponent integration and case values retained | existing aorta task IDs/actions/observations; component selective reset; rebind views after rebuild |

Wire defaults (CatheterRodSpec and parity fixture): density 7800, Young modulus
1e9, Poisson ratio 0.3, bend multiplier 0.1, twist multiplier 0.4, linear/angular
damping 0.01, tip length 0.025 m. Per-node mass is density*pi*radius^2*segment_length
(0.000020420352248333657 kg for the standalone default). Shear modulus is
Young/(2*(1+Poisson)); I=pi*radius^4/4, J=2I. The direct solver's bend parameter
is 4*bend_multiplier*I/segment_length^2 and twist is
4*twist_multiplier*J/segment_length^2, matching scripts/parity/catheter.py.
Startup positions/frames and material rest lengths/Darboux are independent.
Guidance projection must not silently seed material rest curvature.

Additional baseline consumers, kept distinct from navigation defaults:

- Internal main standalone sensor: origin (-0.11, 0.04, 0.08), +X, 48 segments,
  length 0.22, radius 0.0005, insertion cap 0.10, tip 8 edges at 0.35 rad,
  track stiffness 0.65, four solver substeps, zero gravity, static collision off.
  Two catheter action columns capped at 0.030 m/s and 1.5 rad/s. The sensor
  currently advances physics and must hand stepping to the manager.
- Internal main guided demo: 40 segments, shaft length 0.65 times patient CT
  width in metres, whole-shaft placement resampled from the route on every
  action; insertion clamps to route length minus shaft length. Preserve this
  as an explicitly kinematic mode, not evidence of physical navigation.
- Navigation branch: 120 Hz physics, decimation 4; catheter has three rate
  columns, capped at 0.060 m/s, 1.5 rad/s, 1.5 rad/s; the UI insertion
  slider defaults to 0.009 m/s; tip clamp +/-1.5 rad.
  Policy low-dimensional widths are 3/3/3/2/4 for tip_position, tip_direction,
  target_offset, route_state, drive_state (15 total), alongside fluoroscopy_rgb.
  C-arm rate is 0.6 rad/s, initial pi/4, angle range [-pi/6, pi/2].
- Aorta component reference uses `aorta_static`: authored coordinates deliberately
  retain the existing scaled scene. Wire starts (0,0,1), +X, 192 nodes,
  segment 0.05, radius 0.02, node mass 1; Young 1e6, torsion 1e9, bend 0.1,
  twist 1, legacy unit inertia/per-edge tip mapping remain the aorta defaults
  (physical/polyline requirement applies to migrated workflow catheters).
  Guide 6.35, 16 tip edges, insertion cap 6.35; rates (0.5,1.5,1), tip bounds
  +/-1.5, guide stiffness 1, root slew 0.01. dt=0.016666667, 8 substeps,
  1 iteration, particle damping 0.99, zero gravity. Mesh scale 0.01,
  translation (9.833,0.111,0.503), Euler degrees (-76.375273,78.437867,0),
  mesh containment search 0.09. Target (9.69,0.18,0.38), radius 0.1,
  hold 0.5 s, timeout 120 s. Do not apply SI workflow conversions to this case.

Patient-specific geometry/radii remain authored inputs, not invented defaults. Kinematic whole-shaft guided demo must be labelled separately
from physical navigation. Legacy cleanup, alternative backends and SCR are not
ported. Requested obsolete overrides must fail with migration guidance.

## Contact and mount settings for the proposal

The navigation branch's vessel adapter supplies node mass 1, segment inertia 1,
stretch/bend/twist multipliers 1, four Jacobi iterations, zero gravity,
linear/angular damping 0.01, root and endpoints locked. Containment is two-way,
wall response 0.5, search distance 0.05 m, wire/vessel displacement clamps
0.005 m, and two collision projections; the rod alternates contact/elastic
solves 32 times. Navigation interior deadband/stiffness are 0.5/0.25.
The replacement uses staged execution and cached linear containment with exact
overflow fallback, not the legacy scheduling/cleanup passes. Vessel graph loading,
cache settings and end-to-end behavior still require integration qualification.

The robot drive body is `panda_hand`, local translation (0,0,0.1034) m;
its introducer geometry uses a 0.11 m sheath, 45-degree tilt and a 1 s approach.
Preserve the authored quaternion/roll and the existing one-step reaction delay.
The component mount and coordinate conversion are proposal work still to implement.

The consumer matrix is a source-backed proposal contract. It is not a declaration
that the listed consumers already use the replacement. The wire-case converter
is the only workflow code added by this draft.

## Frozen functional gates

Retain existing assertions and analytic tolerances. In particular the bent-lumen
node bound is free lumen radius + 0.0002 m (test_catheter_port.py). No trajectory
matching or new performance threshold is a release gate.

- Material conversion, physical inertia, polyline tip, seeded material curvature,
  startup/reset placement and roll.
- External root commands and curved guide endpoints; complete batches; root/mount
  ownership exclusion; selective reset isolation and stable device buffer identity.
- Interior pull moves only the wire and reports no contact; real contact moves
  the vessel. Candidate overflow falls back to exact search.
- Straight, bent, branching lumens; finite state and exact pinned-node preservation.
- E=1,2,16; captured/eager equivalence with existing runtime tolerances.
- Root force/moment analytic fixtures and mounted two-way coupling.
- Existing navigation target reached; other environments unaffected; finite PPO update.
- Native interaction, keyboard release, successful process exit.
- Full runtime/packages, CPU example, builds, workflows and vascular regressions.
- Old/new positions, deformation, reactions and timing at E=1,16,64 (diagnostic).
- Clean internal install and execution with old package absent, then deprecation
  notice retaining old source/package for public fluoroscopy.

Lab qualification uses Isaac Lab 6864dd1, Newton 1.6.1 and Warp 1.17.0. Standalone
uses its own lock. Update all six upstream source pins together only to a validated
revision. Public Docker, old API compatibility, static mesh edges and wall friction
are separate deliveries.
