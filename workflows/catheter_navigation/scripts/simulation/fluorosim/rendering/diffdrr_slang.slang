/**
 * Differentiable Digitally Reconstructed Radiograph (DRR) using Slang Autodiff
 *
 * This implementation uses Slang's automatic differentiation to compute exact
 * gradients of the rendered image with respect to pose parameters.
 *
 * Key Features:
 * - Software trilinear interpolation for differentiable volume sampling
 * - [Differentiable] marked functions for gradient computation
 * - Forward and backward pass entry points
 * - Compatible with PyTorch autograd via slangpy
 *
 * Physics:
 *   I(u,v) = I₀ · exp(-∫ μ(s) ds)
 */

// ============================================================================
// Configuration
// ============================================================================

static const int MAX_STEPS = 2048;
static const float EPS = 1e-12f;
static const float PI = 3.14159265359f;

// ============================================================================
// Data Structures
// ============================================================================

struct Pose
{
    float3 rotation;     // Euler angles (rx, ry, rz) in radians
    float3 translation;  // Translation (tx, ty, tz) in mm
};

struct CarmGeometry
{
    float sdd;              // Source-to-Detector Distance (mm)
    float sid;              // Source-to-Isocenter Distance (mm)
    float2 detectorSize;    // Detector physical size (mm)
    int2 detectorPixels;    // Detector resolution (pixels)
    float pixelSpacing;     // Pixel pitch (mm)
};

struct VolumeInfo
{
    float3 spacing;     // Voxel spacing (X, Y, Z) in mm
    int3 dimensions;    // Volume dimensions (X, Y, Z) in voxels
    float3 origin;      // Volume origin in world coordinates (mm)
};

// Catheter segment for Beer-Lambert compositing during ray march.
// Each segment is a cylinder between two 3D endpoints with uniform radius
// and a linear attenuation coefficient (mm^-1).
// Layout: 8 × float32 = 32 bytes, tightly packed (no float3 alignment issues).
struct CatheterSegment
{
    float p0x, p0y, p0z;   // Proximal endpoint (mm, world frame)
    float p1x, p1y, p1z;   // Distal endpoint (mm, world frame)
    float radius;           // Cylinder radius (mm)
    float mu;               // Linear attenuation coefficient (mm^-1)

    float3 getP0() { return float3(p0x, p0y, p0z); }
    float3 getP1() { return float3(p1x, p1y, p1z); }
};

// ============================================================================
// Differentiable Helper Functions
// ============================================================================

/**
 * Convert Euler angles to rotation matrix (ZXY convention).
 * Marked [Differentiable] for gradient propagation.
 */
[Differentiable]
float3x3 eulerToRotationMatrix(float3 euler)
{
    float cx = cos(euler.x);
    float sx = sin(euler.x);
    float cy = cos(euler.y);
    float sy = sin(euler.y);
    float cz = cos(euler.z);
    float sz = sin(euler.z);

    // ZXY convention: R = Rz * Rx * Ry
    float3x3 R;
    R[0][0] = cz * cy - sz * sx * sy;
    R[0][1] = -sz * cx;
    R[0][2] = cz * sy + sz * sx * cy;
    R[1][0] = sz * cy + cz * sx * sy;
    R[1][1] = cz * cx;
    R[1][2] = sz * sy - cz * sx * cy;
    R[2][0] = -cx * sy;
    R[2][1] = sx;
    R[2][2] = cx * cy;

    return R;
}

/**
 * Transform a point from local C-arm frame to world frame.
 */
[Differentiable]
float3 transformPoint(float3 localPoint, float3x3 R, float3 translation)
{
    return mul(R, localPoint) + translation;
}

/**
 * Ray-box intersection using slab method.
 * Returns (tNear, tFar). If tFar < tNear, ray misses box.
 */
[Differentiable]
float2 rayBoxIntersect(float3 rayOrigin, float3 rayDir, float3 boxMin, float3 boxMax)
{
    // Safe inverse direction. Use an explicit +/-EPS fallback (not sign(), since
    // sign(0)==0 would leave a zero denominator -> inf/nan for axis-aligned rays
    // with an exactly-zero direction component).
    float3 invDir;
    invDir.x = 1.0f / (abs(rayDir.x) > EPS ? rayDir.x : (rayDir.x >= 0.0f ? EPS : -EPS));
    invDir.y = 1.0f / (abs(rayDir.y) > EPS ? rayDir.y : (rayDir.y >= 0.0f ? EPS : -EPS));
    invDir.z = 1.0f / (abs(rayDir.z) > EPS ? rayDir.z : (rayDir.z >= 0.0f ? EPS : -EPS));

    float3 t0 = (boxMin - rayOrigin) * invDir;
    float3 t1 = (boxMax - rayOrigin) * invDir;

    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);

    float tNear = max(max(tMin.x, tMin.y), tMin.z);
    float tFar = min(min(tMax.x, tMax.y), tMax.z);

    return float2(tNear, tFar);
}

/**
 * Software trilinear interpolation - fully differentiable.
 *
 * The gradient flows through the interpolation weights which depend on uvw.
 * The actual voxel values are sampled with no_diff() since we don't need
 * gradients w.r.t. the volume data itself (only w.r.t. pose).
 */
[Differentiable]
float trilinearSample(
    Texture3D<float> volume,
    SamplerState sampler,
    float3 uvw,
    int3 dimensions
)
{
    // Clamp UVW to valid range [0, 1]
    float3 uvwClamped = clamp(uvw, float3(0.0f), float3(1.0f));

    // Convert to voxel coordinates
    float3 voxelCoord = uvwClamped * float3(dimensions) - float3(0.5f);
    voxelCoord = clamp(voxelCoord, float3(0.0f), float3(dimensions) - float3(1.001f));

    // Get integer indices of the 8 surrounding voxels
    int3 idx0 = int3(floor(voxelCoord));
    int3 idx1 = min(idx0 + int3(1), dimensions - int3(1));

    // Compute fractional position within the voxel cell [0, 1]
    float3 frac = voxelCoord - float3(idx0);

    // Compute trilinear weights - these ARE differentiable w.r.t. uvw!
    float w000 = (1.0f - frac.x) * (1.0f - frac.y) * (1.0f - frac.z);
    float w100 = frac.x * (1.0f - frac.y) * (1.0f - frac.z);
    float w010 = (1.0f - frac.x) * frac.y * (1.0f - frac.z);
    float w110 = frac.x * frac.y * (1.0f - frac.z);
    float w001 = (1.0f - frac.x) * (1.0f - frac.y) * frac.z;
    float w101 = frac.x * (1.0f - frac.y) * frac.z;
    float w011 = (1.0f - frac.x) * frac.y * frac.z;
    float w111 = frac.x * frac.y * frac.z;

    // Sample the 8 corners using hardware texture sampling (non-differentiable)
    float3 texelSize = float3(1.0f) / float3(dimensions);

    float3 uv000 = (float3(idx0) + float3(0.5f)) * texelSize;
    float3 uv100 = (float3(idx1.x, idx0.y, idx0.z) + float3(0.5f)) * texelSize;
    float3 uv010 = (float3(idx0.x, idx1.y, idx0.z) + float3(0.5f)) * texelSize;
    float3 uv110 = (float3(idx1.x, idx1.y, idx0.z) + float3(0.5f)) * texelSize;
    float3 uv001 = (float3(idx0.x, idx0.y, idx1.z) + float3(0.5f)) * texelSize;
    float3 uv101 = (float3(idx1.x, idx0.y, idx1.z) + float3(0.5f)) * texelSize;
    float3 uv011 = (float3(idx0.x, idx1.y, idx1.z) + float3(0.5f)) * texelSize;
    float3 uv111 = (float3(idx1) + float3(0.5f)) * texelSize;

    // Sample with no_diff() - we don't need gradients through the texture lookup
    float v000 = no_diff(volume.Sample(sampler, uv000));
    float v100 = no_diff(volume.Sample(sampler, uv100));
    float v010 = no_diff(volume.Sample(sampler, uv010));
    float v110 = no_diff(volume.Sample(sampler, uv110));
    float v001 = no_diff(volume.Sample(sampler, uv001));
    float v101 = no_diff(volume.Sample(sampler, uv101));
    float v011 = no_diff(volume.Sample(sampler, uv011));
    float v111 = no_diff(volume.Sample(sampler, uv111));

    // Weighted sum - gradients flow through the weights
    return w000 * v000 + w100 * v100 + w010 * v010 + w110 * v110 +
           w001 * v001 + w101 * v101 + w011 * v011 + w111 * v111;
}

// ============================================================================
// Catheter Ray-Cylinder Helpers (non-differentiable — catheter positions fixed)
// ============================================================================

/**
 * Squared distance from a 3D point to a line segment (p0, p1).
 * Returns the squared perpendicular distance clamped to the segment extent.
 */
float pointSegmentDistSq(float3 pt, float3 p0, float3 p1)
{
    float3 seg = p1 - p0;
    float segLenSq = dot(seg, seg);
    if (segLenSq < EPS)
        return dot(pt - p0, pt - p0);

    float t = clamp(dot(pt - p0, seg) / segLenSq, 0.0f, 1.0f);
    float3 proj = p0 + seg * t;
    float3 diff = pt - proj;
    return dot(diff, diff);
}

/**
 * Ray-cylinder intersection for a capped cylinder defined by segment (p0, p1)
 * with the given radius. Returns (tNear, tFar); tFar < tNear means no hit.
 *
 * NOTE: This helper is currently not used by the active catheter render path.
 * It is kept as groundwork for an exact chord-length attenuation path
 * (higher-fidelity Beer-Lambert than step-based inside/outside checks),
 * which is deferred until the quality/performance trade-off is validated.
 *
 * The test is performed in world coordinates by solving the quadratic:
 *   || (O + tD - P0) - ((O + tD - P0)·A) A ||^2 = r^2
 * where A = normalize(P1 - P0), then clamping to the segment extent.
 */
float2 rayCylinderIntersect(
    float3 rayOrigin,
    float3 rayDir,
    float3 p0,
    float3 p1,
    float radius
)
{
    float3 axis = p1 - p0;
    float axisLen = length(axis);
    if (axisLen < EPS)
        return float2(1.0f, -1.0f);  // degenerate

    float3 A = axis / axisLen;
    float3 dp = rayOrigin - p0;

    // Remove component along cylinder axis from both ray direction and delta
    float dDotA = dot(rayDir, A);
    float dpDotA = dot(dp, A);

    float3 dPerp  = rayDir - A * dDotA;
    float3 dpPerp = dp     - A * dpDotA;

    float a = dot(dPerp, dPerp);
    float b = 2.0f * dot(dPerp, dpPerp);
    float c = dot(dpPerp, dpPerp) - radius * radius;

    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f || a < EPS)
        return float2(1.0f, -1.0f);  // miss

    float sqrtDisc = sqrt(disc);
    float invA = 1.0f / (2.0f * a);
    float t0 = (-b - sqrtDisc) * invA;
    float t1 = (-b + sqrtDisc) * invA;

    // Clamp to segment extent [0, axisLen] along axis
    float s0 = dpDotA + t0 * dDotA;
    float s1 = dpDotA + t1 * dDotA;

    if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; tmp = t0; t0 = t1; t1 = tmp; }

    // Clip to valid segment range
    if (s1 < 0.0f || s0 > axisLen)
        return float2(1.0f, -1.0f);  // entirely outside caps

    if (s0 < 0.0f)      t0 = t0 + (0.0f - s0) / (dDotA + EPS * sign(dDotA));
    if (s1 > axisLen)    t1 = t1 - (s1 - axisLen) / (dDotA + EPS * sign(dDotA));

    // Re-clamp (rounding safety)
    if (t0 > t1)
        return float2(1.0f, -1.0f);

    return float2(t0, t1);
}

/**
 * Compute total catheter chord length traversed by the ray at parameter t.
 * For each catheter segment whose cylinder contains the sample point,
 * accumulates the segment's mu value (Beer-Lambert: I *= exp(-sum(mu_i * chord_i))).
 *
 * Uses pointSegmentDistSq for a simple inside/outside test rather than full
 * intersection, so attenuation is accumulated per ray step (stepMM acts as ds).
 *
 * Important: this step-based path intentionally uses two factors:
 *   1) stepMM in the ray march = longitudinal path-length integration along the ray.
 *   2) chordFactor below        = radial soft weighting inside the cylinder cross-section.
 * They are not the same distance term; this is an anti-aliasing approximation that
 * softens binary inside/outside transitions. The exact entry/exit-chord path is
 * tracked separately (rayCylinderIntersect) and is currently not enabled.
 */
float catheterAttenuation(
    float3 samplePoint,
    StructuredBuffer<CatheterSegment> segments,
    int numSegments
)
{
    float totalMu = 0.0f;
    for (int i = 0; i < numSegments; i++)
    {
        CatheterSegment seg = segments[i];
        float distSq = pointSegmentDistSq(samplePoint, seg.getP0(), seg.getP1());
        float rSq = seg.radius * seg.radius;
        if (distSq < rSq)
        {
            // Radial soft weighting inside this step.
            // - stepMM (ray march) handles along-ray distance integration.
            // - chordFactor handles where the sample lies in the cylinder cross-section.
            //   center sample -> higher weight, near-edge sample -> lower weight.
            float chordFactor = sqrt(1.0f - distSq / rSq);
            totalMu += seg.mu * chordFactor;
        }
    }
    return totalMu;
}

/**
 * Differentiable ray-march through volume.
 * Uses [MaxIters] to satisfy Slang's autodiff loop requirements.
 */
[Differentiable]
float rayMarchIntegral(
    Texture3D<float> volume,
    SamplerState sampler,
    float3 rayOrigin,
    float3 rayDir,
    float tStart,
    float tEnd,
    VolumeInfo volInfo,
    float stepMM
)
{
    float3 volumeSize = float3(volInfo.dimensions) * volInfo.spacing;
    float integral = 0.0f;

    float rayLength = tEnd - tStart;
    int numSteps = int(rayLength / stepMM) + 1;
    numSteps = min(numSteps, MAX_STEPS);

    // [MaxIters] tells Slang the maximum iterations for autodiff
    [MaxIters(2048)]
    for (int i = 0; i < numSteps; i++)
    {
        float t = tStart + (float(i) + 0.5f) * stepMM;
        if (t >= tEnd)
            break;

        float3 worldPos = rayOrigin + rayDir * t;
        float3 uvw = (worldPos - volInfo.origin) / volumeSize;

        float mu = trilinearSample(volume, sampler, uvw, volInfo.dimensions);
        integral += mu * stepMM;
    }

    return integral;
}

// ============================================================================
// Core Differentiable Rendering Function
// ============================================================================

/**
 * Compute the intensity for a single pixel using differentiable ray-marching.
 *
 * This function is marked [Differentiable] so Slang will automatically
 * generate the backward pass for gradient computation.
 */
[Differentiable]
float computePixelIntensity(
    Texture3D<float> muVolume,
    SamplerState volumeSampler,
    VolumeInfo volInfo,
    CarmGeometry carm,
    float3 rotation,      // Euler angles in radians
    float3 translation,   // Translation in mm
    int2 pixelCoord,
    float stepMM,
    float i0
)
{
    // Build rotation matrix from Euler angles
    float3x3 R = eulerToRotationMatrix(rotation);

    // Volume center (isocenter)
    float3 volumeCenter = volInfo.origin + float3(volInfo.dimensions) * volInfo.spacing * 0.5f;

    // Source position in world coordinates
    float3 sourceLocal = float3(0.0f, 0.0f, -carm.sid);
    float3 sourceWorld = transformPoint(sourceLocal, R, translation + volumeCenter);

    // Detector pixel position in world coordinates
    float detZ = carm.sdd - carm.sid;
    float2 pixelOffset = float2(
        (float(pixelCoord.x) + 0.5f - float(carm.detectorPixels.x) * 0.5f) * carm.pixelSpacing,
        (float(pixelCoord.y) + 0.5f - float(carm.detectorPixels.y) * 0.5f) * carm.pixelSpacing
    );

    float3 detectorLocal = float3(pixelOffset.x, pixelOffset.y, detZ);
    float3 detectorWorld = transformPoint(detectorLocal, R, translation + volumeCenter);

    // Ray direction (normalized)
    float3 rayDir = normalize(detectorWorld - sourceWorld);

    // Volume bounding box
    float3 boxMin = volInfo.origin;
    float3 boxMax = volInfo.origin + float3(volInfo.dimensions) * volInfo.spacing;

    // Ray-box intersection
    float2 tHit = rayBoxIntersect(sourceWorld, rayDir, boxMin, boxMax);

    // Check for valid intersection
    if (tHit.x > tHit.y || tHit.y < 0.0f)
    {
        return i0;
    }

    // Clamp to positive t
    float tStart = max(tHit.x, 0.0f);
    float tEnd = tHit.y;

    // Compute line integral
    float integral = rayMarchIntegral(
        muVolume, volumeSampler,
        sourceWorld, rayDir,
        tStart, tEnd,
        volInfo, stepMM
    );

    // Beer-Lambert law: I = I₀ * exp(-∫μds)
    return i0 * exp(-integral);
}

// ============================================================================
// Entry Points for slangpy
// ============================================================================

/**
 * Forward pass: Render DRR image.
 * Called via slangpy's dispatch for each pixel.
 */
[CudaKernel]
void renderDRR_forward(
    uint3 dispatchThreadID,
    Texture3D<float> muVolume,
    SamplerState volumeSampler,
    RWTexture2D<float> outputImage,
    VolumeInfo volInfo,
    CarmGeometry carm,
    Pose pose,
    float stepMM,
    float i0
)
{
    int2 pixelCoord = int2(dispatchThreadID.xy);

    if (pixelCoord.x >= carm.detectorPixels.x || pixelCoord.y >= carm.detectorPixels.y)
        return;

    float intensity = computePixelIntensity(
        muVolume, volumeSampler,
        volInfo, carm,
        pose.rotation, pose.translation,
        pixelCoord,
        stepMM, i0
    );

    outputImage[pixelCoord] = intensity;
}

/**
 * Backward pass: Compute gradients of loss w.r.t. pose.
 * Uses Slang's automatic differentiation via bwd_diff().
 */
[CudaKernel]
void renderDRR_backward(
    uint3 dispatchThreadID,
    Texture3D<float> muVolume,
    SamplerState volumeSampler,
    Texture2D<float> gradOutput,        // ∂L/∂I (read-only)
    RWTexture2D<float4> gradRotation,   // ∂L/∂(rx,ry,rz)
    RWTexture2D<float4> gradTranslation,// ∂L/∂(tx,ty,tz)
    VolumeInfo volInfo,
    CarmGeometry carm,
    Pose pose,
    float stepMM,
    float i0
)
{
    int2 pixelCoord = int2(dispatchThreadID.xy);

    if (pixelCoord.x >= carm.detectorPixels.x || pixelCoord.y >= carm.detectorPixels.y)
        return;

    // Get upstream gradient for this pixel
    float dL_dI = gradOutput[pixelCoord];

    // Create differential pairs for inputs we want gradients for
    DifferentialPair<float3> dpRotation = diffPair(pose.rotation, float3(0.0f));
    DifferentialPair<float3> dpTranslation = diffPair(pose.translation, float3(0.0f));

    // Wrap constants in diffPair with zero derivatives
    DifferentialPair<float> dpStepMM = diffPair(stepMM, 0.0f);
    DifferentialPair<float> dpI0 = diffPair(i0, 0.0f);

    // Call backward differentiation
    bwd_diff(computePixelIntensity)(
        muVolume, volumeSampler,
        volInfo, carm,
        dpRotation,
        dpTranslation,
        pixelCoord,
        dpStepMM, dpI0,
        dL_dI
    );

    // Store per-pixel gradients
    gradRotation[pixelCoord] = float4(dpRotation.d, 0.0f);
    gradTranslation[pixelCoord] = float4(dpTranslation.d, 0.0f);
}

// ============================================================================
// Catheter-Aware DRR: volume + catheter Beer-Lambert in a single ray march
// ============================================================================

/**
 * Ray-march through volume AND catheter geometry simultaneously.
 *
 * At each step the mu contribution is:
 *   mu_total(s) = mu_volume(s) + sum_i mu_catheter_i(s)
 *
 * This produces depth-correct compositing: catheter behind bone appears
 * correctly attenuated, and self-crossings are handled additively.
 */
float rayMarchWithCatheter(
    Texture3D<float> volume,
    SamplerState sampler,
    float3 rayOrigin,
    float3 rayDir,
    float tStart,
    float tEnd,
    VolumeInfo volInfo,
    float stepMM,
    StructuredBuffer<CatheterSegment> catheterSegments,
    int numCatheterSegments
)
{
    float3 volumeSize = float3(volInfo.dimensions) * volInfo.spacing;
    float integral = 0.0f;

    float rayLength = tEnd - tStart;
    int numSteps = int(rayLength / stepMM) + 1;
    numSteps = min(numSteps, MAX_STEPS);

    for (int i = 0; i < numSteps; i++)
    {
        float t = tStart + (float(i) + 0.5f) * stepMM;
        if (t >= tEnd)
            break;

        float3 worldPos = rayOrigin + rayDir * t;

        // Volume attenuation
        float3 uvw = (worldPos - volInfo.origin) / volumeSize;
        float3 uvwClamped = clamp(uvw, float3(0.0f), float3(1.0f));
        bool insideVolume = all(uvw >= float3(0.0f)) && all(uvw <= float3(1.0f));

        float muStep = 0.0f;
        if (insideVolume)
        {
            muStep = volume.Sample(sampler, uvwClamped);
        }

        // Catheter attenuation (additive Beer-Lambert)
        muStep += catheterAttenuation(worldPos, catheterSegments, numCatheterSegments);

        integral += muStep * stepMM;
    }

    return integral;
}

/**
 * Compute pixel intensity with both volume AND catheter Beer-Lambert
 * in a single fused ray march. Depth-correct: the catheter is treated
 * as an additional attenuation source at its 3D world position.
 */
float computePixelIntensityWithCatheter(
    Texture3D<float> muVolume,
    SamplerState volumeSampler,
    VolumeInfo volInfo,
    CarmGeometry carm,
    float3 rotation,
    float3 translation,
    int2 pixelCoord,
    float stepMM,
    float i0,
    StructuredBuffer<CatheterSegment> catheterSegments,
    int numCatheterSegments
)
{
    float3x3 R = eulerToRotationMatrix(rotation);
    float3 volumeCenter = volInfo.origin + float3(volInfo.dimensions) * volInfo.spacing * 0.5f;

    float3 sourceLocal = float3(0.0f, 0.0f, -carm.sid);
    float3 sourceWorld = transformPoint(sourceLocal, R, translation + volumeCenter);

    float detZ = carm.sdd - carm.sid;
    float2 pixelOffset = float2(
        (float(pixelCoord.x) + 0.5f - float(carm.detectorPixels.x) * 0.5f) * carm.pixelSpacing,
        (float(pixelCoord.y) + 0.5f - float(carm.detectorPixels.y) * 0.5f) * carm.pixelSpacing
    );

    float3 detectorLocal = float3(pixelOffset.x, pixelOffset.y, detZ);
    float3 detectorWorld = transformPoint(detectorLocal, R, translation + volumeCenter);

    float3 rayDir = normalize(detectorWorld - sourceWorld);

    float3 boxMin = volInfo.origin;
    float3 boxMax = volInfo.origin + float3(volInfo.dimensions) * volInfo.spacing;

    // Pad the volume box slightly to include catheter segments near the boundary
    float3 pad = float3(20.0f);  // 20 mm padding
    float3 extMin = boxMin - pad;
    float3 extMax = boxMax + pad;

    // Primary range: volume bounding box (always needed for the DRR)
    float2 tHitVol = rayBoxIntersect(sourceWorld, rayDir, boxMin, boxMax);
    if (tHitVol.x > tHitVol.y || tHitVol.y < 0.0f)
    {
        // Ray misses volume entirely — check catheter-only range within padded box
        float2 tHitPad = rayBoxIntersect(sourceWorld, rayDir, extMin, extMax);
        if (tHitPad.x > tHitPad.y || tHitPad.y < 0.0f)
            return i0;
        // Catheter only: use padded range but limit to avoid huge step counts
        tHitVol = tHitPad;
    }

    float tVolNear = max(tHitVol.x, 0.0f);
    float tVolFar  = tHitVol.y;

    // Extend range modestly for catheter segments near the volume boundary
    float2 tHitPadded = rayBoxIntersect(sourceWorld, rayDir, extMin, extMax);
    if (tHitPadded.x <= tHitPadded.y && tHitPadded.y >= 0.0f)
    {
        tVolNear = min(tVolNear, max(tHitPadded.x, 0.0f));
        tVolFar  = max(tVolFar,  tHitPadded.y);
    }

    if (tVolNear >= tVolFar || tVolFar < 0.0f)
        return i0;

    float integral = rayMarchWithCatheter(
        muVolume, volumeSampler,
        sourceWorld, rayDir,
        tVolNear, tVolFar,
        volInfo, stepMM,
        catheterSegments, numCatheterSegments
    );

    return i0 * exp(-integral);
}

/**
 * Forward pass: Render DRR with catheter compositing (Beer-Lambert, depth-correct).
 */
[CudaKernel]
void renderDRR_withCatheter_forward(
    uint3 dispatchThreadID,
    Texture3D<float> muVolume,
    SamplerState volumeSampler,
    RWTexture2D<float> outputImage,
    VolumeInfo volInfo,
    CarmGeometry carm,
    Pose pose,
    float stepMM,
    float i0,
    StructuredBuffer<CatheterSegment> catheterSegments,
    int numCatheterSegments
)
{
    int2 pixelCoord = int2(dispatchThreadID.xy);

    if (pixelCoord.x >= carm.detectorPixels.x || pixelCoord.y >= carm.detectorPixels.y)
        return;

    float intensity = computePixelIntensityWithCatheter(
        muVolume, volumeSampler,
        volInfo, carm,
        pose.rotation, pose.translation,
        pixelCoord,
        stepMM, i0,
        catheterSegments, numCatheterSegments
    );

    outputImage[pixelCoord] = intensity;
}

// ============================================================================
// Multi-Environment (Batched) Helpers and Entry Points
// ============================================================================

/**
 * Catheter attenuation with a slice into a flat StructuredBuffer.
 * Used by batched kernels where all environments share one large buffer;
 * each env owns elements [segOffset, segOffset + segCount).
 */
float catheterAttenuation_slice(
    float3 samplePoint,
    StructuredBuffer<CatheterSegment> segments,
    int segOffset,
    int segCount
)
{
    float totalMu = 0.0f;
    for (int i = 0; i < segCount; i++)
    {
        CatheterSegment seg = segments[segOffset + i];
        float distSq = pointSegmentDistSq(samplePoint, seg.getP0(), seg.getP1());
        float rSq    = seg.radius * seg.radius;
        if (distSq < rSq)
        {
            float chordFactor = sqrt(1.0f - distSq / rSq);
            totalMu += seg.mu * chordFactor;
        }
    }
    return totalMu;
}

/**
 * Ray-march with a catheter slice (batched variant).
 */
float rayMarchWithCatheterSlice(
    Texture3D<float> volume,
    SamplerState sampler,
    float3 rayOrigin,
    float3 rayDir,
    float tStart,
    float tEnd,
    VolumeInfo volInfo,
    float stepMM,
    StructuredBuffer<CatheterSegment> catheterSegments,
    int segOffset,
    int segCount
)
{
    float3 volumeSize = float3(volInfo.dimensions) * volInfo.spacing;
    float  integral   = 0.0f;
    int    numSteps   = min(int((tEnd - tStart) / stepMM) + 1, MAX_STEPS);

    for (int i = 0; i < numSteps; i++)
    {
        float t = tStart + (float(i) + 0.5f) * stepMM;
        if (t >= tEnd) break;

        float3 worldPos = rayOrigin + rayDir * t;
        float3 uvw      = (worldPos - volInfo.origin) / volumeSize;
        float muStep    = 0.0f;

        if (all(uvw >= float3(0.0f)) && all(uvw <= float3(1.0f)))
            muStep = volume.Sample(sampler, clamp(uvw, float3(0.0f), float3(1.0f)));

        if (segCount > 0)
            muStep += catheterAttenuation_slice(worldPos, catheterSegments, segOffset, segCount);

        integral += muStep * stepMM;
    }
    return integral;
}

/**
 * Per-pixel intensity for one environment, using a shared flat catheter buffer.
 * segCount == 0 → pure volume DRR (no catheter for this env).
 */
float computePixelIntensityBatched(
    Texture3D<float> muVolume,
    SamplerState     volumeSampler,
    VolumeInfo       volInfo,
    CarmGeometry     carm,
    Pose             pose,
    int2             pixelCoord,
    float            stepMM,
    float            i0,
    StructuredBuffer<CatheterSegment> catheterSegments,
    int segOffset,
    int segCount
)
{
    float3x3 R           = eulerToRotationMatrix(pose.rotation);
    float3 volumeCenter  = volInfo.origin + float3(volInfo.dimensions) * volInfo.spacing * 0.5f;
    float3 sourceWorld   = transformPoint(float3(0.0f, 0.0f, -carm.sid), R, pose.translation + volumeCenter);

    float2 pixOff = float2(
        (float(pixelCoord.x) + 0.5f - float(carm.detectorPixels.x) * 0.5f) * carm.pixelSpacing,
        (float(pixelCoord.y) + 0.5f - float(carm.detectorPixels.y) * 0.5f) * carm.pixelSpacing
    );
    float3 detectorWorld = transformPoint(float3(pixOff.x, pixOff.y, carm.sdd - carm.sid), R, pose.translation + volumeCenter);
    float3 rayDir        = normalize(detectorWorld - sourceWorld);

    float3 boxMin = volInfo.origin;
    float3 boxMax = volInfo.origin + float3(volInfo.dimensions) * volInfo.spacing;
    float3 pad    = float3(20.0f);

    float2 tHitVol = rayBoxIntersect(sourceWorld, rayDir, boxMin, boxMax);
    if (tHitVol.x > tHitVol.y || tHitVol.y < 0.0f)
    {
        if (segCount == 0) return i0;
        float2 tHitPad = rayBoxIntersect(sourceWorld, rayDir, boxMin - pad, boxMax + pad);
        if (tHitPad.x > tHitPad.y || tHitPad.y < 0.0f) return i0;
        tHitVol = tHitPad;
    }

    float tNear = max(tHitVol.x, 0.0f);
    float tFar  = tHitVol.y;

    if (segCount > 0)
    {
        float2 tHitPad = rayBoxIntersect(sourceWorld, rayDir, boxMin - pad, boxMax + pad);
        if (tHitPad.x <= tHitPad.y && tHitPad.y >= 0.0f)
        {
            tNear = min(tNear, max(tHitPad.x, 0.0f));
            tFar  = max(tFar, tHitPad.y);
        }
    }

    if (tNear >= tFar || tFar < 0.0f) return i0;

    float integral = rayMarchWithCatheterSlice(
        muVolume, volumeSampler,
        sourceWorld, rayDir, tNear, tFar,
        volInfo, stepMM,
        catheterSegments, segOffset, segCount
    );
    return i0 * exp(-integral);
}

/**
 * Batched forward DRR render — N environments in one dispatch.
 *
 * Dispatch size:  (det_width, det_height, num_envs)
 * Thread mapping: x=pixel_x, y=pixel_y, z=env_idx
 *
 * Output layout (row-major flat float32):
 *   outputBuffer[envIdx * H * W + y * W + x]
 *
 * Pose layout:    poses[envIdx]  — one Pose (float3 rotation, float3 translation) per env
 *
 * Catheter layout (flat, per-env slice):
 *   catheterSegments[catheterOffsets[envIdx] .. + catheterCounts[envIdx]]
 *   Pass catheterCounts[envIdx] == 0 for a pure-volume render on that env.
 */
[CudaKernel]
void renderDRR_forward_batched(
    uint3 dispatchThreadID,
    Texture3D<float>   muVolume,
    SamplerState       volumeSampler,
    RWStructuredBuffer<float> outputBuffer,
    VolumeInfo         volInfo,
    CarmGeometry       carm,
    StructuredBuffer<Pose> poses,
    float              stepMM,
    float              i0,
    int                numEnvs,
    StructuredBuffer<CatheterSegment> catheterSegments,
    StructuredBuffer<int> catheterOffsets,
    StructuredBuffer<int> catheterCounts
)
{
    int x      = int(dispatchThreadID.x);
    int y      = int(dispatchThreadID.y);
    int envIdx = int(dispatchThreadID.z);

    if (x >= carm.detectorPixels.x || y >= carm.detectorPixels.y || envIdx >= numEnvs)
        return;

    Pose pose  = poses[envIdx];
    int segOff = catheterOffsets[envIdx];
    int segCnt = catheterCounts[envIdx];

    float intensity = computePixelIntensityBatched(
        muVolume, volumeSampler, volInfo, carm,
        pose, int2(x, y), stepMM, i0,
        catheterSegments, segOff, segCnt
    );

    int W = carm.detectorPixels.x;
    int H = carm.detectorPixels.y;
    outputBuffer[envIdx * H * W + y * W + x] = intensity;
}
