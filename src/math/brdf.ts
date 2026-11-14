import { clamp, cross3, dot3, normalize3, sub3, type Vec3 } from './linalg.ts';
import type { RGB } from './color.ts';

export const PI = Math.PI;

export interface CpuLight {
  /** Unit vector pointing from the surface toward the light. */
  direction: Vec3;
  /** Linear-light colour. */
  color: RGB;
  /** Radiant intensity multiplier. */
  intensity: number;
  /** Angular radius of the light disc in radians; see effectiveAlpha. */
  angularRadius: number;
}

export interface CpuShading {
  /** World-space shading normal (already includes normal mapping and strength). */
  n: Vec3;
  /** World-space tangent and bitangent, orthonormal basis. */
  t: Vec3;
  b: Vec3;
  /** World-space unit vector from the surface toward the camera. */
  view: Vec3;
  albedo: RGB;
  roughness: number;
  metallic: number;
  ambient: RGB;
  lights: CpuLight[];
  diffuseOnly: boolean;
}

export interface ShadingDebug {
  radiance: RGB;
  diffuse: RGB;
  specular: RGB;
  f0: RGB;
  nDotL: number[];
}

export function fresnelSchlick(u: number, f0: RGB): RGB {
  const k = Math.pow(clamp(1 - u, 0, 1), 5);
  return [
    f0[0] + (1 - f0[0]) * k,
    f0[1] + (1 - f0[1]) * k,
    f0[2] + (1 - f0[2]) * k,
  ];
}

// Finite disc light: convolve the slope distribution with the light's angular radius.
// Keeps mirror highlights bounded; punctual lights would be singular at alignment.
export function effectiveAlpha(alpha: number, angularRadius: number): number {
  const r = Math.max(2 * angularRadius, 0.001);
  return Math.sqrt(alpha * alpha + r * r);
}

export function distributionGGX(nDotH: number, alpha: number): number {
  const a2 = alpha * alpha;
  const d = nDotH * nDotH * (a2 - 1) + 1;
  return a2 / (PI * d * d);
}

export function visibilitySmithGGX(nDotV: number, nDotL: number, alpha: number): number {
  const a2 = alpha * alpha;
  const gv = nDotL * Math.sqrt(nDotV * nDotV * (1 - a2) + a2);
  const gl = nDotV * Math.sqrt(nDotL * nDotL * (1 - a2) + a2);
  return 0.5 / Math.max(gv + gl, 1e-6);
}

// Mirror of the WGSL in src/gpu/shaders.ts. The benchmark compares this reference
// against the GPU output pixel-by-pixel; both use the same stated model:
// Lambert diffuse + Cook-Torrance GGX specular (Smith height-correlated visibility,
// Schlick Fresnel) + constant Lambertian ambient.
export function evaluateShading(s: CpuShading, debug?: ShadingDebug): RGB {
  const n = normalize3(s.n);
  const t = normalize3(sub3(s.t, scale(n, dot3(n, s.t))));
  const b = normalize3(cross3(n, t));
  const view = normalize3(s.view);

  const viewTS: Vec3 = [dot3(view, t), dot3(view, b), dot3(view, n)];
  const roughness = clamp(s.roughness, 0.02, 1);
  const metallic = clamp(s.metallic, 0, 1);
  const alpha = Math.max(roughness * roughness, 0.0005);
  const f0: RGB = [
    0.04 + (s.albedo[0] - 0.04) * metallic,
    0.04 + (s.albedo[1] - 0.04) * metallic,
    0.04 + (s.albedo[2] - 0.04) * metallic,
  ];

  const radiance: RGB = [0, 0, 0];
  const diffuseSum: RGB = [0, 0, 0];
  const specSum: RGB = [0, 0, 0];
  const nDotLs: number[] = [];
  const nTS: Vec3 = [0, 0, 1];

  for (const light of s.lights) {
    const l = normalize3(light.direction);
    const lTS: Vec3 = [dot3(l, t), dot3(l, b), dot3(l, n)];
    const nDotL = Math.max(lTS[2], 0);
    nDotLs.push(nDotL);
    if (nDotL <= 0) continue;
    const h = normalize3([viewTS[0] + lTS[0], viewTS[1] + lTS[1], viewTS[2] + lTS[2]]);
    const nDotV = Math.max(viewTS[2], 1e-4);
    const nDotH = Math.max(h[2], 0);
    const vDotH = Math.max(dot3(viewTS, h), 0);

    const f = fresnelSchlick(vDotH, f0);
    // Diffuse-only disables both the specular lobe and Fresnel energy compensation,
    // matching the shader's textbook Lambert branch used by the reference scenes.
    const kd: RGB = s.diffuseOnly
      ? [1 - metallic, 1 - metallic, 1 - metallic]
      : [
        (1 - f[0]) * (1 - metallic),
        (1 - f[1]) * (1 - metallic),
        (1 - f[2]) * (1 - metallic),
      ];
    const diff: RGB = [
      (kd[0] * s.albedo[0]) / PI,
      (kd[1] * s.albedo[1]) / PI,
      (kd[2] * s.albedo[2]) / PI,
    ];
    let spec: RGB = [0, 0, 0];
    if (!s.diffuseOnly) {
      const alphaEff = effectiveAlpha(alpha, light.angularRadius);
      const d = distributionGGX(nDotH, alphaEff);
      const vis = visibilitySmithGGX(nDotV, nDotL, alphaEff);
      spec = [d * vis * f[0], d * vis * f[1], d * vis * f[2]];
    }
    const scaleL = light.intensity * nDotL;
    for (let c = 0; c < 3; c++) {
      diffuseSum[c] += diff[c] * light.color[c] * scaleL;
      specSum[c] += spec[c] * light.color[c] * scaleL;
    }
    // nTS is (0,0,1) by construction for the CPU reference (shading normal is n).
    void nTS;
  }

  const ambient: RGB = [
    s.ambient[0] * s.albedo[0] * (1 - metallic),
    s.ambient[1] * s.albedo[1] * (1 - metallic),
    s.ambient[2] * s.albedo[2] * (1 - metallic),
  ];

  for (let c = 0; c < 3; c++) {
    radiance[c] = diffuseSum[c] + specSum[c] + ambient[c];
  }

  if (debug) {
    debug.radiance = radiance;
    debug.diffuse = diffuseSum;
    debug.specular = specSum;
    debug.f0 = f0;
    debug.nDotL = nDotLs;
  }
  return radiance;
}

function scale(v: Vec3, s: number): Vec3 { return [v[0] * s, v[1] * s, v[2] * s]; }

// Mirrors the WGSL bench entry point: exposure in linear light, clamp, then the exact
// sRGB OETF, quantised to 8-bit as the GPU readback does.
export function encodeBenchPixel(linear: RGB, exposureStops: number): [number, number, number] {
  const k = Math.pow(2, exposureStops);
  const enc = (c: number): number => {
    const v = Math.min(1, Math.max(0, c * k));
    const s = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, s)) * 255);
  };
  return [enc(linear[0]), enc(linear[1]), enc(linear[2])];
}
