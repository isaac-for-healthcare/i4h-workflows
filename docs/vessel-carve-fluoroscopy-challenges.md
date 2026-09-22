# Carving the Vessel Out of the CT: What Makes It Hard

Feature #5 asks that fluoroscopy reflect vessel and lumen deformation. The recommended
approach in [`vessel-deformation-in-fluoroscopy.md`](vessel-deformation-in-fluoroscopy.md) is
to carve the vessel out of the attenuation volume at build time, set those voxels to
surrounding soft tissue, and composite the deformed vessel back in as dynamic geometry — so
the runtime path matches the catheter's exactly and the per-frame arithmetic stays trivial.

That recommendation stands. This document is the part it does not say: **the compositing is
the easy half, and every real difficulty is in the carve.** The numbers below are measured
against `data/TotalSegmentator/s0011`, not estimated.

## The runtime half is already built

`arena/i4h_arena/medical/catheter_attenuation.py` composites analytic geometry into the
Beer-Lambert line integral every frame, with chords solved analytically rather than sampled
and spans ending flat so neighbours cannot double-count. A vessel is the same kernel with a
per-node radius and a truncated cone in place of a cylinder. That work is well bounded.

## The measurements

| Quantity | Value |
| --- | --- |
| `vessel_mask.npy` | 65,072 voxels, 220 mL |
| HU inside the vessel | mean 109, median 117 — a contrast CT |
| μ inside the vessel | 0.00098 /mm |
| μ in a 3-voxel shell around it | 0.00055 /mm |
| HU spread in that shell | std 237 HU, 10% of it below −300 HU |
| Mask voxels inside the centerline-plus-radius tube | 83.2% |
| Radius needed to cover 95% of the mask | 1.15× nominal |

Reproduce with:

```python
import numpy as np, pathlib
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.spatial import cKDTree

d = pathlib.Path("data/TotalSegmentator/s0011")
mask = np.load(d / "vessel_mask.npy").astype(bool)
hu = np.load(d / "hu_volume.npy")
mu = np.load(d / "mu_volume.npy")

shell = binary_dilation(mask, structure=generate_binary_structure(3, 1), iterations=3) & ~mask
print("lumen  mu %.5f  HU %.0f" % (mu[mask].mean(), hu[mask].mean()))
print("shell  mu %.5f  HU %.0f  std %.0f  air frac %.2f"
      % (mu[shell].mean(), hu[shell].mean(), hu[shell].std(), (hu[shell] < -300).mean()))

points = np.load(d / "centerline_points_mm.npy").astype(np.float64)
radii = np.load(d / "centerline_radii_mm.npy").astype(np.float64)
origin = np.array([-228.455078125, -419.455078125, -61.0])
voxels_mm = origin[None, :] + np.argwhere(mask)[:, ::-1] * 1.5
distance, nearest = cKDTree(points).query(voxels_mm, k=1)
print("inside the tube: %.3f" % (distance <= radii[nearest]).mean())
```

## 1. There is no single soft-tissue value to carve to

This is the one that sinks the naive version.

The aorta's neighbourhood is not soft tissue. It is soft tissue *and* lung *and* vertebral
body *and* fat *and* heart. In the shell immediately around the segmented vessel the HU
standard deviation is 237 and a tenth of the voxels are air. Fill the lumen with one scalar
and the projection carries a vessel-shaped ghost: a uniform-μ column where the descending
aorta should have lung behind it on one side and bone on the other. The ghost is static, so
it never clears, and it sits exactly where the dynamic signal is meant to be read.

The fix is inpainting from the local neighbourhood rather than a global constant. But "local"
has to exclude the lung and the spine, so the carve needs anatomical awareness rather than
one `volume[mask] = soft_tissue_mu`.

## 2. What you carve and what you add back are different shapes

The carve uses the segmentation mask. The add-back uses the physics model, which is a
centerline with per-node radii — a circular tube. Only **83% of the mask's voxels fall inside
that tube**, 95% coverage needs 1.15× the nominal radius, and some mask voxels sit at 3× it.
Branch ostia, non-circular cross-sections, and any aneurysmal or calcified segment are where
the two diverge.

So carve-then-add does not cancel. Everywhere the geometries disagree leaves a permanent
static artifact: material removed and never restored, or restored in the wrong place.

## 3. The residual is the same size as the signal

This is what makes the first two load-bearing rather than cosmetic. In line-integral units:

| Term | Magnitude |
| --- | --- |
| Lumen-to-surroundings contrast | 0.00043 /mm |
| Carve boundary off by one 1.5 mm voxel, entry and exit | ≈ 0.0013 |
| 1 mm of wall motion — the effect being added | ≈ 0.002 |

Same order of magnitude. The artifact introduced competes with the phenomenon being shown.
Carve accuracy is therefore a correctness requirement, not a quality target, and it has to be
validated directly: render the carved volume with the *undeformed* vessel recomposited,
difference it against the original render, and require the residual to sit well below the
deformation signal. There is currently no reason to assume it would.

## 4. Which μ does the dynamic vessel get? (RQ3)

RQ3 requires the deformed region's μ to pass through the same calibrated transfer function as
the static volume. Worth knowing before starting: **the existing dynamic-geometry path
already violates this.**

`CatheterMaterial` hardcodes 0.8 /mm for the nitinol shaft and 3.0 /mm for the tungsten
marker, which are roughly physical linear attenuation coefficients. The volume's
`interventional` HU→μ preset maps water to about 0.0006 /mm, where physical is ~0.019 /mm —
roughly 30× low, recovered downstream by the display window fitted from the first frame.

The catheter is therefore composited on a physical scale into anatomy rendered on a scale
about thirty times weaker, which is why it reads as essentially opaque. Derive the contrast
column's μ physically and it inherits the same mismatch and comes out black; derive it from
the preset and it is consistent but not physical. Feature #5 has to choose deliberately, and
Feature #4 fixing the scale first would make the choice easy.

## 5. Three inconsistent vessel geometries already exist

The renderer does not currently use the geometry you would carve:

| Geometry | Where | What it is |
| --- | --- | --- |
| `_contrast_bolus_mask` | `slang_fluoroscopy.py` | centerline voxels dilated 1.2 mm, multiplied by `dsa_boost = 6.0` into a second renderer |
| `vessel_mask.npy` | patient twin artifact | the segmentation, 220 mL |
| centerline tree | physics runtime | nodes plus per-node radii, what actually deforms |

Carving forces all three into agreement. That is good hygiene and more work than the ticket
implies, and it means touching the DSA path — which is where the observation gap actually
lives, since in plain DRR the wall is soft tissue against soft tissue and deformation is
nearly invisible.

## 6. The carve is baked at construction; the model is not

`SlangDiffDRRRenderer` takes its volume at construction, so the carve is a build-time
artifact while the add-back geometry is a runtime model. Recalibrate the radii, add the
eccentricity term that device contact will eventually need, or switch twins, and the carve
has to be rebuilt to match. Nothing would currently catch the two drifting apart.

The carve also forces a scope decision. The physics deforms a centerline *tree*, but the
catheter only touches one route. Carving and re-adding the whole tree pays a per-frame cost
for branches that never move; carving only the navigated vessel reintroduces the
inconsistency for any branch that does.

## 7. 1.5 mm voxels against 0.5–2 mm of motion

A binary mask on a 1.5 mm grid gives a stair-stepped carve boundary, against a vessel wall
about 2 mm thick, for motion of 0.5–2 mm. Fractional occupancy at the boundary is needed
rather than a binary mask, or the carve aliases at roughly the scale of the effect. This is
the same argument that made the catheter path solve chords analytically instead of sampling
them.

## Suggested order

1. **Measure the wall motion first.** `CenterlineDynamicsParams` still ships placeholder unit
   values for `node_mass`, `bend_stiffness`, `twist_stiffness` and `stretch_stiffness`, so
   displacement under contact has never been calibrated, and
   `CenterlineVesselRuntime.positions()` already exposes the deformed centerline. At 0.1 mm
   the feature is cosmetic and carve artifacts would dominate it; at 3 mm it is the dominant
   observation error.
2. **Settle the μ scale** — ideally as Feature #4, so Feature #5 inherits one calibrated
   transfer function instead of choosing between two.
3. **Build the carve and validate it at rest**, by the difference test in section 3, before
   any deformation is composited.
4. **Reconcile the three geometries**, replacing the DSA boost with the carve so the contrast
   column becomes dynamic geometry.
5. **Generalize `CatheterAttenuation` from cylinder to truncated cone** with per-node radii on
   device arrays, and feed it the runtime's deformed positions each step.

Steps 1 and 2 are cheap and they decide whether the rest is worth doing.
