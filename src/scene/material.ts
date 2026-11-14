import type { Vec3 } from '../math/linalg.ts';

export type SwatchId =
  | 'leather-black'
  | 'leather-cognac'
  | 'alcantara-graphite'
  | 'aluminum-brushed'
  | 'carbon-twill'
  | 'piano-gloss'
  | 'flatscreen-void';

export type MeshTarget = 'dashboard' | 'trim' | 'softpad';

export interface MaterialParams {
  /** sRGB hex, authored like a colour swatch. Linearised before upload. */
  baseColor: string;
  roughness: number;
  metallic: number;
  normalStrength: number;
  normalEnabled: boolean;
  /** Multiplies the mesh UVs; affects texture tiling, not the geometry. */
  uvScale: number;
}

export interface MaterialBounds {
  roughness: [number, number];
  metallic: [number, number];
  normalStrength: [number, number];
  uvScale: [number, number];
}

// Every material parameter is bounded and the UI clamps to these ranges. The shader
// also clamps, so a malformed preset cannot push values out of a sane range.
export const MATERIAL_BOUNDS: MaterialBounds = {
  roughness: [0.02, 1],
  metallic: [0, 1],
  normalStrength: [0, 4],
  uvScale: [0.25, 16],
};

export interface SwatchEntry {
  id: SwatchId;
  label: string;
  target: MeshTarget;
  defaults: MaterialParams;
  /** Documented reason the preset values are what they are. */
  note: string;
}

export const SWATCHES: SwatchEntry[] = [
  {
    id: 'leather-black',
    label: '黑色打孔皮革',
    target: 'dashboard',
    defaults: { baseColor: '#1b1b1d', roughness: 0.62, metallic: 0.0, normalStrength: 1.0, normalEnabled: true, uvScale: 3.2 },
    note: '皮革：低金属度、中高粗糙度，孔洞法线在掠射角可见。',
  },
  {
    id: 'leather-cognac',
    label: '干邑色软皮',
    target: 'dashboard',
    defaults: { baseColor: '#9a5a2c', roughness: 0.48, metallic: 0.0, normalStrength: 0.9, normalEnabled: true, uvScale: 3.2 },
    note: '软皮：颜色随起伏变化，粗糙度略低于黑色皮革。',
  },
  {
    id: 'alcantara-graphite',
    label: '石墨灰翻毛皮',
    target: 'dashboard',
    defaults: { baseColor: '#4a4c50', roughness: 0.94, metallic: 0.0, normalStrength: 1.6, normalEnabled: true, uvScale: 5.0 },
    note: '翻毛皮：接近完全漫反射，法线强度提高以表现纤维方向。',
  },
  {
    id: 'aluminum-brushed',
    label: '拉丝铝饰条',
    target: 'trim',
    defaults: { baseColor: '#c3c6cb', roughness: 0.28, metallic: 1.0, normalStrength: 0.5, normalEnabled: true, uvScale: 8.0 },
    note: '金属饰条：金属度 1，F0 等于基础色；拉丝方向由 UV 各向异性纹理给出。',
  },
  {
    id: 'carbon-twill',
    label: '碳纤维斜纹',
    target: 'trim',
    defaults: { baseColor: '#2a2c31', roughness: 0.22, metallic: 0.35, normalStrength: 1.2, normalEnabled: true, uvScale: 6.0 },
    note: '碳纤维：清漆下树脂的介质金属感，粗糙度较低。',
  },
  {
    id: 'piano-gloss',
    label: '钢琴黑高光',
    target: 'softpad',
    defaults: { baseColor: '#0a0a0b', roughness: 0.06, metallic: 0.0, normalStrength: 0.15, normalEnabled: true, uvScale: 2.0 },
    note: '高光饰板：近镜面，Fresnel 边缘反射是主要视觉信号。',
  },
  {
    id: 'flatscreen-void',
    label: '纯漫反射基准',
    target: 'dashboard',
    defaults: { baseColor: '#ffffff', roughness: 1.0, metallic: 0.0, normalStrength: 0.0, normalEnabled: false, uvScale: 1.0 },
    note: '基准：白色、完全粗糙、无金属、法线贴图关闭；用于对照解析值。',
  },
];

export function swatchById(id: SwatchId): SwatchEntry {
  const s = SWATCHES.find((x) => x.id === id);
  if (!s) throw new Error(`unknown swatch ${id}`);
  return s;
}

export function clampParams(p: MaterialParams): MaterialParams {
  const b = MATERIAL_BOUNDS;
  return {
    baseColor: /^#[0-9a-fA-F]{6}$/.test(p.baseColor) ? p.baseColor : '#808080',
    roughness: clampRange(p.roughness, b.roughness),
    metallic: clampRange(p.metallic, b.metallic),
    normalStrength: clampRange(p.normalStrength, b.normalStrength),
    normalEnabled: !!p.normalEnabled,
    uvScale: clampRange(p.uvScale, b.uvScale),
  };
}

function clampRange(x: number, r: [number, number]): number {
  if (!Number.isFinite(x)) return r[0];
  return Math.min(r[1], Math.max(r[0], x));
}

// --- Camera / lighting / exposure ---------------------------------------------

export interface CameraState {
  yaw: number;      // radians, orbit around target
  pitch: number;    // radians, elevation
  distance: number; // metres
  target: Vec3;
  fovY: number;     // radians
}

export interface LightState {
  direction: Vec3;  // unit vector toward the light, world space
  color: [number, number, number]; // linear
  intensity: number;
  /**
   * Angular radius of the light's disc in radians. A punctual light makes a
   * near-mirror surface either infinitely bright (exact alignment) or black (a
   * fraction of a degree off) — both useless for comparison. Treating the light as a
   * finite disc and convolving its slope with the GGX roughness keeps mirror
   * highlights finite and physically motivated. Packed into color.w on the GPU.
   */
  angularRadius: number;
}

export interface LightingState {
  key: LightState;
  fill: LightState;
  rim: LightState;
  ambient: [number, number, number]; // linear ambient radiance
}

export interface ExposureState {
  stops: number;
}

export interface PresetState {
  camera: CameraState;
  lighting: LightingState;
  exposure: ExposureState;
  swatch: SwatchId;
  material: MaterialParams;
}

// --- Validation ----------------------------------------------------------------

export function validateCamera(c: CameraState): CameraState {
  return {
    yaw: finiteOr(c.yaw, 0.45),
    pitch: Math.min(1.45, Math.max(-0.4, finiteOr(c.pitch, 0.22))),
    distance: Math.min(6, Math.max(0.4, finiteOr(c.distance, 2.1))),
    target: [
      Math.min(2, Math.max(-2, finiteOr(c.target?.[0], 0))),
      Math.min(2, Math.max(-2, finiteOr(c.target?.[1], 0.25))),
      Math.min(2, Math.max(-2, finiteOr(c.target?.[2], 0.3))),
    ],
    fovY: Math.min(1.6, Math.max(0.25, finiteOr(c.fovY, 0.62))),
  };
}

export function validateLighting(l: LightingState): LightingState {
  const fix = (x: LightState): LightState => {
    const d = x?.direction ?? [0, 0, 1];
    const len = Math.hypot(d[0], d[1], d[2]);
    // A zero-length direction cannot be normalised; fall back to +Z rather than
    // uploading a zero vector, which would silently disable the light's N·L term.
    // Vectors already unit to within 1e-6 are left untouched so defaults, presets and
    // the values shown in the UI are preserved exactly.
    const dir: [number, number, number] = len > 1e-6
      ? Math.abs(len - 1) > 1e-6 ? [d[0] / len, d[1] / len, d[2] / len] : [d[0], d[1], d[2]]
      : [0, 0, 1];
    return {
      direction: dir,
      color: [
        Math.min(1, Math.max(0, finiteOr(x?.color?.[0], 1))),
        Math.min(1, Math.max(0, finiteOr(x?.color?.[1], 1))),
        Math.min(1, Math.max(0, finiteOr(x?.color?.[2], 1))),
      ],
      intensity: Math.min(40, Math.max(0, finiteOr(x?.intensity, 1))),
      angularRadius: Math.min(0.5, Math.max(0.002, finiteOr(x?.angularRadius, 0.05))),
    };
  };
  return { key: fix(l.key), fill: fix(l.fill), rim: fix(l.rim), ambient: normalizeAmbient(l.ambient) };
}

function normalizeAmbient(a: [number, number, number]): [number, number, number] {
  return [
    Math.min(2, Math.max(0, finiteOr(a?.[0], 0.02))),
    Math.min(2, Math.max(0, finiteOr(a?.[1], 0.02))),
    Math.min(2, Math.max(0, finiteOr(a?.[2], 0.025))),
  ];
}

export function validateExposure(e: ExposureState): ExposureState {
  return { stops: Math.min(6, Math.max(-6, finiteOr(e?.stops, 0))) };
}

function finiteOr(x: number | undefined, fallback: number): number {
  return typeof x === 'number' && Number.isFinite(x) ? x : fallback;
}

export const DEFAULT_CAMERA: CameraState = {
  yaw: 0.28,
  pitch: 0.16,
  distance: 2.05,
  target: [0.10, 0.22, 0.28],
  fovY: 0.68,
};

// Authored with human-readable decimals, then normalised once at load so the GPU,
// the CPU reference and preset comparison all agree on unit light directions.
const RAW_DEFAULT_LIGHTING: LightingState = {
  key: { direction: [-0.42, 0.62, 0.66], color: [1.0, 0.97, 0.92], intensity: 4.2, angularRadius: 0.045 },
  fill: { direction: [0.72, 0.18, 0.42], color: [0.78, 0.85, 1.0], intensity: 1.15, angularRadius: 0.10 },
  rim: { direction: [0.10, 0.30, -0.95], color: [0.92, 0.95, 1.0], intensity: 1.8, angularRadius: 0.06 },
  ambient: [0.028, 0.030, 0.035],
};

export const DEFAULT_LIGHTING: LightingState = validateLighting(RAW_DEFAULT_LIGHTING);

export const DEFAULT_EXPOSURE: ExposureState = { stops: 0.45 };
