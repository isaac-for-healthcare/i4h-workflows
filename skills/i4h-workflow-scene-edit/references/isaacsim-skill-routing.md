# Isaac Sim Skill Routing

Load this reference only when creating or modifying scene content. The official catalog is pinned to the installed Isaac Sim 6.2.0 release and sparsely checked out by `./third_party/setup.sh`:

```bash
ISAACSIM_SKILLS="./third_party/IsaacSim-74a9356/skills"
test -f "${ISAACSIM_SKILLS}/SKILLS.md" || ./third_party/setup.sh
```

Read `${ISAACSIM_SKILLS}/SKILLS.md`, then load only the upstream `SKILL.md` files needed for the edit:

| Scene work | Upstream skill |
|---|---|
| Cube/primitive rigid body, collider, mass, friction, contact, kinematic behavior | `physics-simulation/SKILL.md` |
| Camera optics, intrinsics, clipping, coordinate convention, distortion, AOVs | `isaac-camera/SKILL.md` |
| Camera/depth/LiDAR/IMU/contact sensor concepts | `isaac-sim-sensor/SKILL.md` |
| Positioning from bounds, transform math, look-at, clearance, layout | `spatial-reasoning/SKILL.md` |
| USD references, scale, materials, asset measurement | `usd-pipeline/SKILL.md` |
| Layered USD composition | `usd-composition-architecture/SKILL.md` |
| Lighting, renderer, exposure, capture quality | `isaac-sim-rendering/SKILL.md` |
| Final generic render/simulation QA criteria | `isaac-sim-validator/SKILL.md` |

## Integration Boundary

Use upstream skills for generic Isaac Sim 6 concepts and current API semantics. Use the i4h skill and the closest working scene for integration:

- Follow the closest working scene's IsaacLab cfg pattern rather than copying a standalone raw-USD tutorial into the runtime.
- Preserve the quaternion convention already required by the concrete API boundary; do not add version-history conversions or aliases.
- Keep policy cameras env-local and update the scene manifest, embodiment observation path, remote-policy client and task-manifest modality, and dataset/training consumers together.
- Keep the scene Python and its manifest as the scene's source of truth, and the `docs/workflows/` page as the workflow's.
- Validate visibly, then run every affected policy or teleoperation device under IsaacLab-Arena after generic upstream QA.

Do not load the whole upstream catalog into context. Do not edit the pinned checkout; update its revision in `./third_party/setup.sh` when upgrading Isaac Sim.
