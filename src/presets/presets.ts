// Presets store the complete reproduction state: camera, lighting, exposure,
// swatch + material parameters. Loading validates and clamps every field, and the
// UI then re-uploads textures and GPU uniforms so the same frame is reproduced.

import {
  DEFAULT_CAMERA,
  DEFAULT_EXPOSURE,
  DEFAULT_LIGHTING,
  clampParams,
  validateCamera,
  validateExposure,
  validateLighting,
  SWATCHES,
  type CameraState,
  type ExposureState,
  type LightingState,
  type MaterialParams,
  type PresetState,
  type SwatchId,
} from '../scene/material.ts';

export const PRESET_STORAGE_KEY = 'aiml.presets.v1';

export interface StoredPreset {
  id: string;
  name: string;
  createdAt: string;
  state: PresetState;
}

export interface PresetDocument {
  version: 1;
  presets: StoredPreset[];
}

export const BUILTIN_PRESETS: StoredPreset[] = [
  {
    id: 'builtin-studio-leather',
    name: '演播室 · 黑色打孔皮革',
    createdAt: '2026-01-01T00:00:00.000Z',
    state: {
      camera: { yaw: 0.34, pitch: 0.22, distance: 2.15, target: [0.12, 0.24, 0.30], fovY: 0.6 },
      lighting: DEFAULT_LIGHTING,
      exposure: { stops: 0 },
      swatch: 'leather-black',
      material: SWATCHES[0].defaults,
    },
  },
  {
    id: 'builtin-gloss-detail',
    name: '高光特写 · 钢琴黑',
    createdAt: '2026-01-01T00:00:00.000Z',
    state: {
      camera: { yaw: 0.62, pitch: 0.30, distance: 1.55, target: [0.5, 0.16, 0.32], fovY: 0.52 },
      lighting: {
        key: { direction: [-0.55, 0.35, 0.75], color: [1.0, 0.98, 0.95], intensity: 5.2, angularRadius: 0.03 },
        fill: { direction: [0.65, 0.05, 0.30], color: [0.72, 0.80, 1.0], intensity: 0.7, angularRadius: 0.12 },
        rim: { direction: [-0.10, 0.55, -0.80], color: [0.9, 0.94, 1.0], intensity: 2.4, angularRadius: 0.06 },
        ambient: [0.010, 0.011, 0.014],
      },
      exposure: { stops: -0.6 },
      swatch: 'piano-gloss',
      material: { baseColor: '#0a0a0b', roughness: 0.06, metallic: 0.0, normalStrength: 0.15, normalEnabled: true, uvScale: 2.0 },
    },
  },
  {
    id: 'builtin-bench-diffuse',
    name: '基准 · 纯漫反射',
    createdAt: '2026-01-01T00:00:00.000Z',
    state: {
      camera: { yaw: 0, pitch: 0, distance: 2.0, target: [0, 0, 0], fovY: 0.6 },
      lighting: DEFAULT_LIGHTING,
      exposure: { stops: 0 },
      swatch: 'flatscreen-void',
      material: { baseColor: '#ffffff', roughness: 1.0, metallic: 0.0, normalStrength: 0.0, normalEnabled: false, uvScale: 1.0 },
    },
  },
];

export function loadPresetDocument(raw: string | null): PresetDocument {
  if (!raw) return { version: 1, presets: [] };
  try {
    const parsed = JSON.parse(raw) as Partial<PresetDocument>;
    if (!parsed || parsed.version !== 1 || !Array.isArray(parsed.presets)) {
      return { version: 1, presets: [] };
    }
    const presets = parsed.presets.filter(isStoredPreset).map((p) => ({
      id: String(p.id),
      name: String(p.name).slice(0, 80),
      createdAt: String(p.createdAt),
      state: normalizeState(p.state),
    }));
    return { version: 1, presets };
  } catch {
    return { version: 1, presets: [] };
  }
}

function isStoredPreset(p: unknown): p is StoredPreset {
  if (!p || typeof p !== 'object') return false;
  const o = p as Record<string, unknown>;
  return typeof o.id === 'string' && typeof o.name === 'string' && typeof o.state === 'object';
}

export function normalizeState(s: Partial<PresetState> | undefined): PresetState {
  const fallbackSwatch: SwatchId = 'leather-black';
  const swatch = (SWATCHES.find((x) => x.id === (s as { swatch?: string } | undefined)?.swatch)?.id ?? fallbackSwatch);
  return {
    camera: validateCamera(s?.camera ?? DEFAULT_CAMERA),
    lighting: validateLighting(s?.lighting ?? DEFAULT_LIGHTING),
    exposure: validateExposure(s?.exposure ?? DEFAULT_EXPOSURE),
    swatch,
    material: clampParams(s?.material ?? SWATCHES.find((x) => x.id === swatch)!.defaults),
  };
}

export function serializePresets(presets: StoredPreset[]): string {
  return JSON.stringify({ version: 1, presets } satisfies PresetDocument, null, 2);
}

// The restore path returns the exact state the UI should apply. Round-tripping
// through JSON means numbers survive reload byte-for-byte, so the same frame is
// reproduced rather than an approximation.
export function restorePreset(preset: StoredPreset): PresetState {
  return normalizeState(JSON.parse(JSON.stringify(preset.state)) as Partial<PresetState>);
}

export interface PresetDiff {
  equal: boolean;
  fields: string[];
}

export function comparePresets(a: PresetState, b: PresetState): PresetDiff {
  const diffs: string[] = [];
  const cam = a.camera, cb = b.camera;
  if (cam.yaw !== cb.yaw) diffs.push('camera.yaw');
  if (cam.pitch !== cb.pitch) diffs.push('camera.pitch');
  if (cam.distance !== cb.distance) diffs.push('camera.distance');
  if (cam.fovY !== cb.fovY) diffs.push('camera.fovY');
  if (cam.target.some((v, i) => v !== cb.target[i])) diffs.push('camera.target');
  if (a.exposure.stops !== b.exposure.stops) diffs.push('exposure.stops');
  if (a.swatch !== b.swatch) diffs.push('swatch');
  const keys = ['key', 'fill', 'rim'] as const;
  for (const k of keys) {
    if (a.lighting[k].intensity !== b.lighting[k].intensity) diffs.push(`lighting.${k}.intensity`);
    if (a.lighting[k].direction.some((v, i) => v !== b.lighting[k].direction[i])) diffs.push(`lighting.${k}.direction`);
    if (a.lighting[k].color.some((v, i) => v !== b.lighting[k].color[i])) diffs.push(`lighting.${k}.color`);
  }
  if (a.lighting.ambient.some((v, i) => v !== b.lighting.ambient[i])) diffs.push('lighting.ambient');
  const keys2 = ['baseColor', 'roughness', 'metallic', 'normalStrength', 'normalEnabled', 'uvScale'] as const;
  for (const k of keys2) {
    if (a.material[k] !== b.material[k]) diffs.push(`material.${k}`);
  }
  return { equal: diffs.length === 0, fields: diffs };
}

export function makePreset(name: string, state: PresetState, now = new Date()): StoredPreset {
  return {
    id: `preset-${now.getTime()}-${Math.floor(Math.random() * 1000)}`,
    name: name.slice(0, 80) || '未命名预设',
    createdAt: now.toISOString(),
    state: normalizeState(state),
  };
}

export interface CameraPresetCamera {
  camera: CameraState;
  lighting: LightingState;
  exposure: ExposureState;
  material: MaterialParams;
}
