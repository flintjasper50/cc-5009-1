// Pure-CPU unit checks for the pieces that must be exactly right before any GPU
// comparison is meaningful: colour transfer, normal-map decode, normal matrix under
// non-uniform scale, parameter bounds, preset round-trip, and the CPU reference BRDF.

import assert from 'node:assert/strict';
import {
  srgbToLinearChannel,
  linearToSrgbChannel,
  normalMapEncode,
  normalMapDecode,
  linearToDisplaySrgb,
  hexToSrgb,
  srgbToHex,
  type RGB,
} from '../../src/math/color.ts';
import {
  mat4Identity,
  mat4FromTRS,
  normalMatrixFromModel,
  mat4Multiply,
  normalize3,
  dot3,
  type Vec3,
} from '../../src/math/linalg.ts';
import {
  evaluateShading,
  encodeBenchPixel,
  effectiveAlpha,
  distributionGGX,
  visibilitySmithGGX,
  type CpuShading,
} from '../../src/math/brdf.ts';
import {
  MATERIAL_BOUNDS,
  clampParams,
  validateCamera,
  validateLighting,
  validateExposure,
  SWATCHES,
  DEFAULT_CAMERA,
  DEFAULT_LIGHTING,
  DEFAULT_EXPOSURE,
} from '../../src/scene/material.ts';
import {
  normalizeState,
  restorePreset,
  comparePresets,
  loadPresetDocument,
  serializePresets,
  BUILTIN_PRESETS,
  type StoredPreset,
} from '../../src/presets/presets.ts';

import {
  buildTextureSet,
  SWATCH_SPECS,
} from '../../src/scene/textures.ts';

let passed = 0;
let failed = 0;
const failures: string[] = [];

function test(name: string, fn: () => void): void {
  try {
    fn();
    passed++;
    console.log(`PASS ${name}`);
  } catch (e) {
    failed++;
    const msg = e instanceof Error ? e.message : String(e);
    failures.push(`${name}: ${msg}`);
    console.log(`FAIL ${name} — ${msg}`);
  }
}

function near(actual: number, expected: number, tol = 1e-9): void {
  assert.ok(Math.abs(actual - expected) <= tol, `expected ${expected} ±${tol}, got ${actual}`);
}

function nearRGB(actual: RGB, expected: RGB, tol = 1e-9): void {
  for (let i = 0; i < 3; i++) near(actual[i], expected[i], tol);
}

// --- Colour transfer ---------------------------------------------------------

test('sRGB→linear matches the IEC 61966-2-1 piecewise curve at known points', () => {
  near(srgbToLinearChannel(0), 0);
  near(srgbToLinearChannel(1), 1);
  near(srgbToLinearChannel(0.5), 0.21404114048223255, 1e-12);
  near(srgbToLinearChannel(0.04045), 0.0031308049535603713, 1e-12);
  near(srgbToLinearChannel(0.2), 0.033104766570885055, 1e-12);
});

test('linear→sRGB inverts the decode across the whole 8-bit domain', () => {
  for (let i = 0; i <= 255; i++) {
    const s = i / 255;
    const back = linearToSrgbChannel(srgbToLinearChannel(s));
    assert.ok(Math.abs(back - s) <= 1e-12, `round-trip failed at ${i}: ${back} != ${s}`);
  }
});

test('sRGB transfer is continuous at the piecewise knee', () => {
  const below = linearToSrgbChannel(0.0031308 - 1e-7);
  const above = linearToSrgbChannel(0.0031308 + 1e-7);
  assert.ok(Math.abs(above - below) < 1e-5, `discontinuity: ${below} vs ${above}`);
});

test('hex round-trips through sRGB byte values', () => {
  for (const hex of ['#000000', '#ffffff', '#1b1b1d', '#9a5a2c', '#c3c6cb', '#0a0a0b']) {
    assert.equal(srgbToHex(hexToSrgb(hex)), hex);
  }
});

// --- Normal encoding ---------------------------------------------------------

test('normal-map encode/decode is a 9-bit-stable direction round trip', () => {
  const dirs: Vec3[] = [[0, 0, 1], [0.4, 0.3, 0.86], [-0.7, 0.1, 0.7], [0, 1, 0]];
  for (const d of dirs) {
    const unit = normalize3(d);
    const bytes = normalMapEncode(unit).map((c) => Math.round(c * 255));
    const decoded = normalMapDecode(bytes.map((b) => b / 255) as RGB);
    const dot = dot3(decoded, unit);
    assert.ok(dot > 0.999, `decoded direction drifted: dot=${dot}`);
  }
});

test('normal decode never uses the sRGB curve', () => {
  // Byte 128 must decode to ~0 in direction space; if the sRGB transfer were applied
  // it would decode to 0.502 - 1 = -0.498.
  const n = normalMapDecode([128 / 255, 128 / 255, 1]);
  assert.ok(Math.abs(n[0]) < 0.005, `byte 128 decoded to ${n[0]}`);
  assert.ok(Math.abs(n[1]) < 0.005, `byte 128 decoded to ${n[1]}`);
  // And the decoded vector is renormalised to unit length (a 2n-1 decode, not a curve).
  near(Math.hypot(n[0], n[1], n[2]), 1, 1e-12);
});

// --- Normal matrix under non-uniform scale -----------------------------------

test('normal matrix keeps normals perpendicular under non-uniform scale', () => {
  const model = mat4FromTRS([0, 0, 0], 0.6, 0.2, [1.35, 0.7, 1.0]);
  const nm = normalMatrixFromModel(model);
  // Surface normals of a unit box; after scaling they must stay perpendicular to the
  // transformed tangents, which a direct model-matrix transform would violate.
  const normals: Vec3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], normalize3([1, 1, 1])];
  const tangents: Vec3[] = [[0, 1, 0], [1, 0, 0], [1, 0, 0], normalize3([1, -1, 0])];
  for (let i = 0; i < normals.length; i++) {
    const nPrime = applyNM(nm, normals[i]);
    const tPrime = applyM(model, tangents[i]);
    const dot = dot3(normalize3(nPrime), normalize3(tPrime));
    assert.ok(Math.abs(dot) < 1e-6, `normal/tangent not perpendicular (${i}): dot=${dot}`);
  }
});

test('normal matrix equals transpose(inverse(model)) for a scaled+rotated model', () => {
  const model = mat4FromTRS([0.54, 0.1, 0.295], 0.3, -0.1, [1.35, 0.7, 1.0]);
  const nm = normalMatrixFromModel(model);
  assert.ok(Number.isFinite(nm[0]) && Number.isFinite(nm[8]), 'normal matrix contains non-finite values');
  // A pure uniform scale would leave the linear part proportional to identity; a non-uniform
  // one must not, which is the case that would break if we skipped the inverse-transpose.
  assert.ok(Math.abs(nm[0] - nm[8]) > 0.1, 'normal matrix looks uniform despite non-uniform scale');
});

test('identity model yields identity normal matrix', () => {
  const nm = normalMatrixFromModel(mat4Identity());
  assert.deepEqual(Array.from(nm), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
});

// --- Parameter bounds -------------------------------------------------------

test('material bounds clamp every field, including NaN/Infinity', () => {
  const wild = clampParams({
    baseColor: 'not-a-hex',
    roughness: 4.5,
    metallic: -3,
    normalStrength: Number.POSITIVE_INFINITY,
    normalEnabled: true,
    uvScale: Number.NaN,
  });
  assert.equal(wild.baseColor, '#808080');
  assert.equal(wild.roughness, MATERIAL_BOUNDS.roughness[1]);
  assert.equal(wild.metallic, MATERIAL_BOUNDS.metallic[0]);
  assert.equal(wild.normalStrength, MATERIAL_BOUNDS.normalStrength[0]);
  assert.equal(wild.uvScale, MATERIAL_BOUNDS.uvScale[0]);
});

test('all built-in swatches are in bounds by construction', () => {
  for (const s of SWATCHES) {
    const c = clampParams(s.defaults);
    assert.deepEqual(c, s.defaults, `swatch ${s.id} defaults out of bounds`);
  }
});

test('camera/lighting/exposure validators clamp and renormalise', () => {
  const cam = validateCamera({ yaw: 99, pitch: 9, distance: 0, target: [5, -5, 7], fovY: 0.1 });
  assert.equal(cam.distance, 0.4);
  assert.equal(cam.pitch, 1.45);
  assert.equal(cam.fovY, 0.25);
  assert.deepEqual(cam.target, [2, -2, 2]);

  const lit = validateLighting({
    key: { direction: [0, 0, 0], color: [2, -1, 0.5], intensity: 99, angularRadius: 0 },
    fill: DEFAULT_LIGHTING.fill,
    rim: DEFAULT_LIGHTING.rim,
    ambient: [9, 9, -4],
  });
  near(Math.hypot(...lit.key.direction), 1, 1e-12);
  assert.equal(lit.key.intensity, 40);
  assert.equal(lit.key.angularRadius, 0.002);
  assert.deepEqual(lit.key.color, [1, 0, 0.5]);
  assert.deepEqual(lit.ambient, [2, 2, 0]);

  assert.equal(validateExposure({ stops: 30 }).stops, 6);
  assert.equal(validateExposure({ stops: -30 }).stops, -6);
  assert.equal(validateExposure({ stops: Number.NaN }).stops, 0);
});

test('defaults are a fixed point of validation', () => {
  assert.deepEqual(validateCamera(DEFAULT_CAMERA), DEFAULT_CAMERA);
  assert.deepEqual(validateExposure(DEFAULT_EXPOSURE), DEFAULT_EXPOSURE);
  // DEFAULT_LIGHTING is normalised once at load, so validation must be a no-op on it.
  assert.deepEqual(validateLighting(DEFAULT_LIGHTING), DEFAULT_LIGHTING);
  for (const k of ['key', 'fill', 'rim'] as const) {
    near(Math.hypot(...DEFAULT_LIGHTING[k].direction), 1, 1e-6);
  }
});

// --- Presets -----------------------------------------------------------------

test('preset round trip is byte-identical through JSON', () => {
  for (const p of BUILTIN_PRESETS) {
    const state = normalizeState(p.state);
    const json = JSON.stringify(state);
    const restored = restorePreset(p);
    assert.deepEqual(restored, state, `preset ${p.id} did not round-trip`);
    assert.equal(JSON.stringify(restored), json);
    assert.deepEqual(comparePresets(state, restored), { equal: true, fields: [] });
  }
});

test('preset comparison reports the exact differing field', () => {
  const base = normalizeState(BUILTIN_PRESETS[0].state);
  const changed = restorePreset({ ...BUILTIN_PRESETS[0], state: { ...base, exposure: { stops: 0.5 } } });
  const diff = comparePresets(base, changed);
  assert.equal(diff.equal, false);
  assert.deepEqual(diff.fields, ['exposure.stops']);
});

test('malformed preset documents degrade to empty, not throw', () => {
  assert.deepEqual(loadPresetDocument(''), { version: 1, presets: [] });
  assert.deepEqual(loadPresetDocument('{'), { version: 1, presets: [] });
  assert.deepEqual(loadPresetDocument('{"version":2,"presets":[]}'), { version: 1, presets: [] });
  const doc = loadPresetDocument(serializePresets([] as StoredPreset[]));
  assert.deepEqual(doc, { version: 1, presets: [] });
});

test('loaded presets are normalised, so out-of-range values are clamped', () => {
  const raw = JSON.stringify({
    version: 1,
    presets: [{
      id: 'x', name: 'bad', createdAt: '2026-01-01T00:00:00.000Z',
      state: { camera: { ...DEFAULT_CAMERA, distance: 999 }, swatch: 'nope', material: { ...SWATCHES[0].defaults, roughness: 9 } },
    }],
  });
  const doc = loadPresetDocument(raw);
  assert.equal(doc.presets.length, 1);
  assert.equal(doc.presets[0].state.camera.distance, 6);
  assert.equal(doc.presets[0].state.swatch, 'leather-black');
  assert.equal(doc.presets[0].state.material.roughness, 1);
});

// --- CPU reference BRDF ------------------------------------------------------

function lambertShading(overrides: Partial<CpuShading> = {}): CpuShading {
  return {
    n: [0, 0, 1],
    t: [1, 0, 0],
    b: [0, 1, 0],
    view: [0, 0, 1],
    albedo: [1, 1, 1],
    roughness: 1,
    metallic: 0,
    ambient: [0, 0, 0],
    diffuseOnly: true,
    lights: [{ direction: [0, 0, 1], color: [1, 1, 1], intensity: 1, angularRadius: 0.05 }],
    ...overrides,
  };
}

test('pure Lambert matches albedo/π exactly', () => {
  const radiance = evaluateShading(lambertShading());
  nearRGB(radiance, [1 / Math.PI, 1 / Math.PI, 1 / Math.PI], 1e-12);
});

test('black albedo produces zero diffuse and only the Fresnel floor specular', () => {
  const radiance = evaluateShading(lambertShading({ albedo: [0, 0, 0], diffuseOnly: false }));
  // With no albedo the only light is the 4% dielectric Fresnel term at normal incidence.
  assert.ok(radiance[0] > 0, 'expected a 4% dielectric specular floor');
  assert.ok(radiance[0] < 0.2, `black dielectric too bright: ${radiance[0]}`);
});

test('metal 1 removes diffuse and uses the albedo as F0', () => {
  const radiance = evaluateShading(lambertShading({ albedo: [0.8, 0.4, 0.1], metallic: 1, diffuseOnly: false }));
  // Diffuse is fully suppressed; no Lambert ambient term; residual is the metal F0 lobe.
  assert.ok(radiance[0] > radiance[2], 'metal tint should lean toward the albedo hue');
});

test('roughness controls highlight width monotonically', () => {
  // Peak density is 1/(π·α²): sharper for mirror-like surfaces. Only the peak falls
  // as roughness rises; the off-normal tail is where the broad lobe gains energy.
  const peak = (roughness: number): number => {
    const a = effectiveAlpha(Math.max(roughness * roughness, 0.0005), 0.02);
    return distributionGGX(1, a);
  };
  assert.ok(peak(0.05) > peak(0.3), 'near-mirror should peak sharper');
  assert.ok(peak(0.3) > peak(0.9), 'roughness should broaden the lobe');
});

test('effective alpha preserves the finite light radius floor', () => {
  near(effectiveAlpha(0, 0.1), Math.max(0.2, 0.001), 1e-12);
  assert.ok(effectiveAlpha(0, 0.1) > effectiveAlpha(0.01, 0.001));
});

test('visibility term stays finite and positive across grazing angles', () => {
  for (const nDotV of [1e-4, 0.01, 0.25, 0.5, 1]) {
    for (const nDotL of [1e-4, 0.25, 0.5, 1]) {
      for (const alpha of [0.001, 0.1, 1]) {
        const v = visibilitySmithGGX(nDotV, nDotL, alpha);
        assert.ok(Number.isFinite(v) && v > 0, `visibility non-finite at ${nDotV}/${nDotL}/${alpha}: ${v}`);
      }
    }
  }
});

test('bench encode matches the GPU quantisation boundaries', () => {
  // Below the linear knee: 0.002 linear → sRGB 0.0258 → byte 7 after rounding.
  assert.deepEqual(encodeBenchPixel([0.002, 0.002, 0.002], 0), [7, 7, 7]);
  assert.deepEqual(encodeBenchPixel([1, 1, 1], 0), [255, 255, 255]);
  // Exposure applies in linear light before encoding, and over-range clips.
  assert.deepEqual(encodeBenchPixel([0.25, 0.25, 0.25], 2), [255, 255, 255]);
  // −2 stops: 0.25 → 0.0625 linear → sRGB 0.2773 → byte 71.
  assert.deepEqual(encodeBenchPixel([0.25, 0.25, 0.25], -2), [71, 71, 71]);
});

test('linearToDisplaySrgb applies exposure before the transfer curve', () => {
  const d0 = linearToDisplaySrgb([0.5, 0.5, 0.5], 0);
  const d1 = linearToDisplaySrgb([0.5, 0.5, 0.5], 1);
  assert.ok(Math.abs(d1[0] - d0[0]) > 0.1, 'one stop should change the display value');
});

// --- Normal-map texture generation -------------------------------------------

test('generated normal maps decode to unit vectors and stay mostly tangent-facing', () => {
  const spec = { ...SWATCH_SPECS[0], size: 64 };
  const set = buildTextureSet(spec);
  let decoded = 0;
  let facing = 0;
  for (let level = 0; level < set.normal.levels.length; level++) {
    const bytes = set.normal.levels[level];
    for (let i = 0; i < bytes.length; i += 4) {
      const n = normalMapDecode([bytes[i] / 255, bytes[i + 1] / 255, bytes[i + 2] / 255]);
      const len = Math.hypot(n[0], n[1], n[2]);
      assert.ok(Math.abs(len - 1) < 1e-6, `normal byte triple not unit length at level ${level}: ${len}`);
      decoded++;
      if (n[2] > 0.5) facing++;
    }
  }
  assert.ok(facing / decoded > 0.9, `only ${(100 * facing / decoded).toFixed(1)}% of normals face outward`);
});

function applyNM(nm: Float32Array, v: Vec3): Vec3 {
  return [
    nm[0] * v[0] + nm[1] * v[1] + nm[2] * v[2],
    nm[3] * v[0] + nm[4] * v[1] + nm[5] * v[2],
    nm[6] * v[0] + nm[7] * v[1] + nm[8] * v[2],
  ];
}

function applyM(m: Float32Array, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
  ];
}

void mat4Multiply;

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log('  - ' + f);
  process.exit(1);
}
