// Interior material kernel. Every material/mutation command goes through here;
// controls are rendered from the kernel snapshot and commands that break the
// documented bounds are rejected instead of silently clamped. A rejection leaves
// the previous material untouched and is surfaced back to the interior material
// panel, so the UI never shows a half-applied state.

import {
  MATERIAL_BOUNDS,
  SWATCHES,
  clampParams,
  swatchById,
  type MaterialParams,
  type PresetState,
  type SwatchId,
} from './material.ts';

export type ParamKey = keyof MaterialParams;

export interface SpecimenState {
  /** Rotation of the selected specimen about its own centre, radians. */
  yaw: number;
  pitch: number;
}

export interface KernelFailure {
  code: 'unknown-swatch' | 'param-out-of-range' | 'invalid-value' | 'unknown-command' | 'specimen-out-of-range';
  command: string;
  message: string;
  at: number;
}

export interface KernelSnapshot {
  swatch: SwatchId;
  params: MaterialParams;
  specimen: SpecimenState;
  revision: number;
  failure: KernelFailure | null;
}

export interface KernelInitial {
  swatch: SwatchId;
  params: MaterialParams;
  specimen?: SpecimenState;
}

type Listener = (snapshot: KernelSnapshot) => void;

export const SPECIMEN_BOUNDS = {
  yaw: [-Math.PI, Math.PI] as [number, number],
  pitch: [-0.6, 0.6] as [number, number],
};

const HEX = /^#[0-9a-fA-F]{6}$/;

export class MaterialKernel {
  private swatch: SwatchId;
  private params: MaterialParams;
  private specimen: SpecimenState;
  private revision = 0;
  private failure: KernelFailure | null = null;
  private readonly listeners = new Set<Listener>();

  constructor(initial: KernelInitial) {
    const known = SWATCHES.find((s) => s.id === initial.swatch);
    if (!known) throw new Error(`unknown initial swatch ${initial.swatch}`);
    this.swatch = initial.swatch;
    this.params = clampParams(initial.params);
    this.specimen = {
      yaw: clampRange(initial.specimen?.yaw ?? 0, SPECIMEN_BOUNDS.yaw),
      pitch: clampRange(initial.specimen?.pitch ?? 0, SPECIMEN_BOUNDS.pitch),
    };
  }

  get snapshot(): KernelSnapshot {
    return {
      swatch: this.swatch,
      params: { ...this.params },
      specimen: { ...this.specimen },
      revision: this.revision,
      failure: this.failure ? { ...this.failure } : null,
    };
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  setSwatch(id: SwatchId): boolean {
    const entry = SWATCHES.find((s) => s.id === id);
    if (!entry) {
      return this.reject('unknown-swatch', 'setSwatch', `未知样块 “${String(id)}”，材质保持不变`);
    }
    this.swatch = entry.id;
    this.params = clampParams(entry.defaults);
    this.commit('setSwatch');
    return true;
  }

  setParam<K extends ParamKey>(key: K, value: MaterialParams[K]): boolean {
    if (!(key in this.params)) {
      return this.reject('unknown-command', 'setParam', `未知材质参数 “${String(key)}”`);
    }
    const error = validateParam(key, value);
    if (error) return this.reject('param-out-of-range', `setParam.${String(key)}`, error);
    this.params = { ...this.params, [key]: value };
    this.commit(`setParam.${String(key)}`);
    return true;
  }

  setSpecimenYaw(yaw: number): boolean {
    if (!Number.isFinite(yaw) || yaw < SPECIMEN_BOUNDS.yaw[0] || yaw > SPECIMEN_BOUNDS.yaw[1]) {
      return this.reject('specimen-out-of-range', 'setSpecimenYaw', `样件偏航 ${yaw} 超出 [${SPECIMEN_BOUNDS.yaw[0]}, ${SPECIMEN_BOUNDS.yaw[1]}]`);
    }
    this.specimen = { ...this.specimen, yaw };
    this.commit('setSpecimenYaw');
    return true;
  }

  setSpecimenPitch(pitch: number): boolean {
    if (!Number.isFinite(pitch) || pitch < SPECIMEN_BOUNDS.pitch[0] || pitch > SPECIMEN_BOUNDS.pitch[1]) {
      return this.reject('specimen-out-of-range', 'setSpecimenPitch', `样件俯仰 ${pitch} 超出 [${SPECIMEN_BOUNDS.pitch[0]}, ${SPECIMEN_BOUNDS.pitch[1]}]`);
    }
    this.specimen = { ...this.specimen, pitch };
    this.commit('setSpecimenPitch');
    return true;
  }

  // Replays the original preset flow: camera/lighting/exposure are applied by the
  // caller, the kernel owns swatch + material. Invalid presets are rejected whole.
  applyPreset(state: PresetState): boolean {
    const entry = SWATCHES.find((s) => s.id === state.swatch);
    if (!entry) {
      return this.reject('unknown-swatch', 'applyPreset', `预设引用了未知样块 “${String(state.swatch)}”，预设整体不应用`);
    }
    for (const key of Object.keys(state.material) as ParamKey[]) {
      const error = validateParam(key, state.material[key]);
      if (error) {
        return this.reject('invalid-value', `applyPreset.material.${String(key)}`, `预设材质字段无效：${error}`);
      }
    }
    this.swatch = entry.id;
    this.params = { ...state.material };
    this.commit('applyPreset');
    return true;
  }

  clearFailure(): void {
    if (!this.failure) return;
    this.failure = null;
    this.revision++;
    this.emit();
  }

  private reject(code: KernelFailure['code'], command: string, message: string): false {
    this.failure = { code, command, message, at: Date.now() };
    this.emit();
    return false;
  }

  private commit(command: string): void {
    this.failure = null;
    this.revision++;
    this.emit(command);
  }

  private emit(_command?: string): void {
    const snap = this.snapshot;
    for (const listener of [...this.listeners]) listener(snap);
  }
}

function validateParam(key: ParamKey, value: unknown): string | null {
  if (key === 'baseColor') {
    return typeof value === 'string' && HEX.test(value) ? null : `基础色必须是 #rrggbb，收到 “${String(value)}”`;
  }
  if (key === 'normalEnabled') {
    return typeof value === 'boolean' ? null : 'normalEnabled 必须是布尔值';
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return `${String(key)} 必须是有限数值，收到 “${String(value)}”`;
  }
  const bounds = MATERIAL_BOUNDS[key as Exclude<ParamKey, 'baseColor' | 'normalEnabled'>];
  if (!bounds) return `未知材质参数 “${String(key)}”`;
  if (value < bounds[0] || value > bounds[1]) {
    return `${String(key)}=${value} 超出 [${bounds[0]}, ${bounds[1]}]`;
  }
  return null;
}

function clampRange(x: number, r: [number, number]): number {
  if (!Number.isFinite(x)) return 0;
  return Math.min(r[1], Math.max(r[0], x));
}

export function swatchLabel(id: SwatchId): string {
  return swatchById(id).label;
}
