# Reflecting vessel deformation in the fluoroscopy image

The sensor-side half of lumen deformation. When the physics deforms the vessel,
the fluoroscopy image must show the deformed lumen, not the original CT
geometry. Otherwise the perception model trains on observations that do not
match the state, which is an observation-space sim-to-real gap that domain
randomisation does not close.

**Summary:** approach (c) is already built and running. The catheter uses it
every frame. The task is extending that path to the vessel.

## The hybrid compositor already exists

`arena/i4h_arena/medical/catheter_attenuation.py` implements "static background
volume plus dynamic geometry composited in the Beer-Lambert integral". Each span
between two nodes is a cylinder with its own linear attenuation coefficient. Its
contribution `sum(mu_i * chord_i)` is added to the line integral the volume
produced. Beer-Lambert is additive in the exponent, so compositing is a multiply
in transmission:

```python
line_integral = self._catheter_attenuation.line_integral(...)
return np.exp(-line_integral).astype(np.float32)
```

Two design decisions in that path carry over to the vessel:

- **Chords are computed analytically, not sampled.** A sub-millimetre shaft is
  thinner than a usable march step, and sampling aliases it in and out of view
  along its length. Contact-induced wall motion is the same order of magnitude
  and has the same problem.
- **Spans end flat, not rounded.** Consecutive spans meet at a shared plane.
  Rounded caps would overlap and count the same material twice where nodes sit a
  fraction of a millimetre apart.

## Double counting is the only new problem

The catheter is additive because it is not in the CT. The vessel is. Adding a
deformed vessel on top of the volume renders it twice.

Two options:

1. **Carve once, add dynamically (recommended).** At build time, set the lumen
   voxels to surrounding soft-tissue μ. The dynamic vessel is then purely
   additive and the runtime path matches the catheter's exactly. One-time cost,
   trivial per-frame arithmetic. The compositing is the easy half, though: see
   [`vessel-carve-fluoroscopy-challenges.md`](vessel-carve-fluoroscopy-challenges.md)
   for what the carve itself costs, measured on `s0011`. In short, there is no
   single surrounding soft-tissue value to carve to, the carved mask and the
   added tube agree on only 83% of their voxels, and the residual from either is
   the same order as one millimetre of wall motion.
2. **Differential composite.** Add the deformed vessel, subtract the undeformed
   one. Also correct, but it needs both geometries every frame, and the delta
   can go negative, so the total line integral has to be clamped rather than the
   delta.

Option 1 is simpler and keeps the rest of this change small.

### The carve replaces the DSA boost

Contrast already has a place in the pipeline. DSA currently builds a static
centerline-derived mask at construction and brightens those voxels into a second
renderer:

```220:228:arena/i4h_arena/medical/slang_fluoroscopy.py
            vessel_mask = _contrast_bolus_mask(patient)
            ...
            boosted_volume[vessel_mask > 0] *= float(dsa_boost)
            self._dsa_renderer = SlangDiffDRRRenderer(boosted_volume, patient.spacing_zyx_mm, cfg=base_config)
```

That baked mask is the code that becomes wrong as soon as the vessel deforms,
and it derives from the same centerline the physics runtime deforms. Replacing
the boost with a carve moves the contrast column from baked voxels to dynamic
geometry.

## Why not re-voxelization or a deformation field

The s0011 volume is 431 x 311 x 311 at 1.5 mm isotropic: 41.7M voxels, 167 MB as
float32.

**(a) Re-voxelization** has two problems. Rebuilding and re-uploading 167 MB at
30 Hz is about 5 GB/s for the volume alone, and the renderer takes its volume at
construction, so per-frame re-voxelization means rebuilding renderer state every
frame. The larger problem is quantization: contact-induced wall motion is 0.5-2
mm against 1.5 mm voxels, so the signal is at roughly the scale of the grid
sampling it. Deformation would advance a voxel at a time rather than smoothly.
This is the aliasing failure the catheter path avoids by computing chords
analytically.

**(b) Deformation-field compositing** is the most physically complete option and
is the right answer if surrounding tissue eventually has to deform coherently
with the vessel. It requires shader changes to the upstream `xray_simulator` ray
march, which means forking upstream, plus per-frame scattered-data interpolation
from a sparse centerline to a dense 3D field. That is a large cost for a first
version, and the field is ill-defined away from the vessel.

**(c) Hybrid** requires no shader change, no volume rebuild and no
interpolation, and reuses a tested kernel.

## RQ1: the state contract

**Feature #6 should publish deformed centerline nodes plus per-node radii, per
env, as device arrays.** Not a mesh, and not a displacement field.

`vessel_skinning.py` "turns the deforming centerline into a render/collision
surface", so the mesh is derived from the centerline. Publishing a mesh means
publishing a downstream artifact and asking the renderer to recover the
primitive it wants. `CenterlineVesselRuntime` already holds `positions` and
`radii` on device in the right layout, and spans-with-radii is what an
analytic-chord kernel consumes directly. It is hundreds of nodes against 41.7M
voxels: roughly five orders of magnitude less data to move, with no host round
trip.

**Limitation and extension point.** Centerline plus radius assumes a circular
cross-section. Device-vessel contact breaks that assumption, since the catheter
presses the wall out on one side. When that fidelity is needed, add an
eccentricity vector alongside the radius. It stays compact and the ray-ellipse
chord remains analytic. Reserve that field in the contract now rather than
switching to a mesh later.

## Work breakdown

1. **Generalize `CatheterAttenuation` from a cylinder to a truncated cone.**
   `line_integral` takes a single scalar `radius_mm`. Vessels taper from roughly
   2 mm to 15 mm along the aorta, so spans need per-node radii. It also takes
   host numpy, which is fine for one catheter but should become device arrays
   for multi-env vessels. This is most of the new kernel work and it is well
   bounded.
2. **Add the build-time carve** in place of the DSA boost, so the contrast
   column becomes dynamic geometry.
3. **Pass the runtime's deformed `positions` and `radii` to the compositor each
   step,** in the renderer's volume-millimetre frame.

## Two scoping decisions

**Prioritise DSA and guidance over plain DRR.** In plain DRR the vessel wall is
soft tissue against soft tissue. The μ contrast is small and deformation is
nearly invisible. In DSA the contrast-filled lumen is the image. The observation
gap in the requirement lives almost entirely in the contrast path.

**Two-way vessel deformation is already on, so the gap is live today.** No scene
sets it, which makes it easy to assume it is off, but the defaults enable it:
`centerline_vessel_from_twin` takes `two_way=True` and `vessel_response=0.5`,
and `CatheterRodSpec.vessel_enabled` defaults to `True`. CUDA graph capture is
also disabled whenever a vessel is present
(`use_cuda_graph=not (spec.wants_vessel or spec.rigid_bodies_enabled)`), which
confirms the branch is taken at runtime.

The wall now absorbs half of each contact correction rather than all of it, and
both its ends are anchored and damped, matching the reference endoluminal scene.
That reduces how far the lumen moves under contact but does not remove the
motion, so the observation gap this document describes still stands.

The renderer's contrast mask, meanwhile, is built once in the constructor from
the static twin:

```34:36:arena/i4h_arena/medical/slang_fluoroscopy.py
def _contrast_bolus_mask(patient: PatientVolume) -> np.ndarray:
    """Build the thin centerline-derived mask used by the reference DSA view."""
    centerline_path = patient.twin.artifacts.get("centerline_points")
```

So containment constrains the wire against the deformed wall while the image
shows the undeformed one. The policy is controlled against one geometry and
observed through another, today. Feature #6 does not introduce this divergence;
it formalises a state contract for one that already exists.

**The magnitude is unmeasured.** `CenterlineDynamicsParams` ships placeholder
unit values throughout: `node_mass`, `bend_stiffness`, `twist_stiffness` and
`stretch_stiffness` are all `1.0`. How far the wall actually moves under contact
has never been calibrated. Measure that first. At 0.1 mm of wall motion the
rendering work is cosmetic; at 3 mm it is the dominant observation error. The
measurement is cheap, since the deformed centerline is already exposed by
`CenterlineVesselRuntime.positions()`.
