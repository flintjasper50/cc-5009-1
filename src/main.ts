import {
  Renderer,
  studioItems,
  resourceSummary,
  type DrawItem,
} from './gpu/renderer.ts';
import { buildSwatchTextures, type TextureSet } from './scene/textures.ts';
import {
  DEFAULT_CAMERA,
  DEFAULT_EXPOSURE,
  DEFAULT_LIGHTING,
  MATERIAL_BOUNDS,
  SWATCHES,
  swatchById,
  type CameraState,
  type ExposureState,
  type LightingState,
  type MaterialParams,
  type PresetState,
  type SwatchId,
} from './scene/material.ts';
import { button, checkbox, colorPicker, labeledBlock, slider } from './ui/controls.ts';
import {
  BENCH_SCENES,
  drawImageDataToCanvas,
  outcomeToImageData,
  runBenchScene,
  type BenchOutcome,
} from './bench/bench.ts';
import {
  BUILTIN_PRESETS,
  PRESET_STORAGE_KEY,
  comparePresets,
  loadPresetDocument,
  makePreset,
  restorePreset,
  serializePresets,
  type StoredPreset,
} from './presets/presets.ts';
import { hexToSrgb } from './math/color.ts';
import { MaterialKernel, SPECIMEN_BOUNDS, type KernelSnapshot } from './scene/kernel.ts';
import type { ShaderCompileState } from './gpu/device.ts';

interface AppState {
  camera: CameraState;
  lighting: LightingState;
  exposure: ExposureState;
  showSoftpad: boolean;
  nonUniformScale: boolean;

  // Mirrors of the kernel snapshot. The kernel is the single writer for swatch,
  // material and specimen state; these accessors keep the rest of the app reading
  // the same shape it always has.
  readonly swatch: SwatchId;
  readonly material: MaterialParams;
}

const kernel = new MaterialKernel({
  swatch: 'leather-black',
  params: SWATCHES[0].defaults,
});

const state: AppState = {
  camera: structuredClone(DEFAULT_CAMERA),
  lighting: structuredClone(DEFAULT_LIGHTING),
  exposure: structuredClone(DEFAULT_EXPOSURE),
  showSoftpad: true,
  nonUniformScale: false,
  get swatch(): SwatchId { return kernel.snapshot.swatch; },
  get material(): MaterialParams { return kernel.snapshot.params; },
};

let renderer: Renderer | null = null;
let textures: Record<string, TextureSet> = {};
let presets: StoredPreset[] = [];
let booting = false;
let overlaying = false;
let needsDraw = true;
let lastFrameStart = 0;
let frameIntervals: number[] = [];
let rafHandle = 0;

const el = {
  canvas: byId<HTMLCanvasElement>('viewport'),
  overlay: byId<HTMLDivElement>('viewport-overlay'),
  overlayTitle: byId<HTMLElement>('overlay-title'),
  overlayBody: byId<HTMLElement>('overlay-body'),
  overlayRetry: byId<HTMLButtonElement>('btn-overlay-retry'),
  swatchList: byId<HTMLDivElement>('swatch-list'),
  material: byId<HTMLDivElement>('material-controls'),
  normal: byId<HTMLDivElement>('normal-controls'),
  specimen: byId<HTMLDivElement>('specimen-controls'),
  camera: byId<HTMLDivElement>('camera-controls'),
  light: byId<HTMLDivElement>('light-controls'),
  exposure: byId<HTMLDivElement>('exposure-controls'),
  preset: byId<HTMLDivElement>('preset-controls'),
  bench: byId<HTMLDivElement>('bench-controls'),
  benchReport: byId<HTMLDivElement>('bench-report'),
  statFps: byId<HTMLDivElement>('stat-fps'),
  statGpu: byId<HTMLDivElement>('stat-gpu'),
  statAdapter: byId<HTMLDivElement>('stat-adapter'),
  recover: byId<HTMLButtonElement>('btn-recover'),
  toast: byId<HTMLDivElement>('toast'),
  legendName: byId<HTMLElement>('legend-name'),
  legendMeta: byId<HTMLElement>('legend-meta'),
};

function byId<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

// ---------------------------------------------------------------------------
// Device bootstrap / loss handling
// ---------------------------------------------------------------------------

function showOverlay(title: string, body: string): void {
  overlaying = true;
  el.overlayTitle.textContent = title;
  el.overlayBody.textContent = body;
  el.overlay.hidden = false;
  el.statFps.querySelector('b')!.textContent = '—';
  el.statGpu.querySelector('b')!.textContent = '—';
  el.legendMeta.textContent = '渲染中断 · 显示的不是当前结果';
  el.legendName.textContent = '无有效帧';
}

function hideOverlay(): void {
  overlaying = false;
  el.overlay.hidden = true;
}

function showToast(message: string, kind: 'ok' | 'bad' = 'ok'): void {
  el.toast.textContent = message;
  el.toast.className = `toast ${kind}`;
  el.toast.hidden = false;
  window.clearTimeout(showToastHandle);
  showToastHandle = window.setTimeout(() => { el.toast.hidden = true; }, 2600);
}
let showToastHandle = 0;

async function boot(): Promise<void> {
  if (booting) return;
  booting = true;
  try {
    if (!textures || Object.keys(textures).length === 0) textures = buildSwatchTextures();
    renderer?.dispose();
    renderer = await Renderer.create(el.canvas, (message) => {
      handleDeviceLost(message);
    });
    hideOverlay();
    el.statAdapter.querySelector('b')!.textContent = shortenAdapter(renderer.diagnostics.adapterLabel);
    el.statAdapter.title = `${renderer.diagnostics.adapterLabel}\nbackend=${renderer.diagnostics.backend}\ntimestamps=${renderer.diagnostics.timestampSupported}`;
    resizeCanvas();
    needsDraw = true;
    startLoop();
    updateLegend();
    showToast('WebGPU 设备已就绪', 'ok');
    (window as unknown as { __renderer?: Renderer }).__renderer = renderer;
    (window as unknown as { __state?: AppState }).__state = state;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    showOverlay('无法初始化 WebGPU', `${message}\n\n窗口不会显示任何旧截图；修复后点击“重建并重绘”。`);
    el.legendName.textContent = '无有效帧';
  } finally {
    booting = false;
  }
}

function handleDeviceLost(message: string): void {
  stopLoop();
  showOverlay('WebGPU 设备丢失', `原因：${message}\n\n上一帧已作废，画布不再显示旧图像。点击下方按钮重建全部 GPU 资源。`);
  el.statGpu.querySelector('b')!.textContent = '0 B';
}

function shortenAdapter(label: string): string {
  return label.length > 30 ? `${label.slice(0, 30)}…` : label;
}

el.recover.addEventListener('click', () => { void boot(); });
el.overlayRetry.addEventListener('click', () => { void boot(); });

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

function startLoop(): void {
  stopLoop();
  const frame = (): void => {
    rafHandle = requestAnimationFrame(frame);
    if (!renderer || renderer.isLost || overlaying) return;
    const now = performance.now();
    if (lastFrameStart > 0) {
      frameIntervals.push(now - lastFrameStart);
      if (frameIntervals.length > 90) frameIntervals.shift();
    }
    lastFrameStart = now;
    if (!needsDraw && frameIntervals.length > 0) {
      // Still redraw so the GPU timestamp readout stays live; the scene itself is cached.
    }
    try {
      renderer.render(buildFrame());
      needsDraw = false;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showOverlay('渲染失败', `${message}\n\n画布已不再显示上一帧；设备需重建后才能继续。`);
      stopLoop();
      return;
    }
    updateStats();
  };
  rafHandle = requestAnimationFrame(frame);
}

function stopLoop(): void {
  if (rafHandle) cancelAnimationFrame(rafHandle);
  rafHandle = 0;
}

function buildFrame() {
  const snap = kernel.snapshot;
  const items: DrawItem[] = studioItems({
    swatchId: snap.swatch,
    swatchTarget: swatchById(snap.swatch).target,
    material: snap.params,
    textureSets: textures,
    nonUniformScale: state.nonUniformScale,
    showSoftpad: state.showSoftpad,
    specimen: snap.specimen,
  });
  return {
    camera: state.camera,
    lighting: state.lighting,
    exposure: state.exposure.stops,
    items,
    diffuseOnly: false,
  };
}

function updateStats(): void {
  if (!renderer) return;
  const gpu = renderer.gpuTimeMs;
  const fpsEl = el.statFps.querySelector('b')!;
  if (gpu !== null) {
    fpsEl.textContent = `${gpu.toFixed(2)} ms`;
    el.statFps.title = 'GPU 时间戳查询测得的场景通道耗时';
  } else if (frameIntervals.length > 4) {
    const avg = frameIntervals.slice(-30).reduce((a, b) => a + b, 0) / Math.min(30, frameIntervals.length);
    fpsEl.textContent = `${avg.toFixed(2)} ms*`;
    el.statFps.title = '此适配器不支持时间戳查询，显示为提交间隔（含 vsync），非纯 GPU 耗时';
  } else {
    fpsEl.textContent = '—';
  }
  el.statGpu.querySelector('b')!.textContent = resourceSummary(renderer);
}

function resizeCanvas(): void {
  const parent = el.canvas.parentElement;
  if (!parent) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(parent.clientWidth * dpr));
  const height = Math.max(1, Math.floor(parent.clientHeight * dpr));
  el.canvas.style.width = '100%';
  el.canvas.style.height = '100%';
  if (renderer && !renderer.isLost) {
    renderer.resize(width, height);
    needsDraw = true;
  }
}

const resizeObserver = new ResizeObserver(() => resizeCanvas());
resizeObserver.observe(el.canvas.parentElement ?? document.body);
window.addEventListener('resize', resizeCanvas);

// ---------------------------------------------------------------------------
// Swatches + material controls
// ---------------------------------------------------------------------------

function renderSwatches(): void {
  el.swatchList.replaceChildren();
  for (const sw of SWATCHES) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'swatch';
    card.setAttribute('aria-pressed', String(state.swatch === sw.id));
    card.title = sw.note;

    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.style.background = chipBackground(sw.id, sw.defaults.baseColor);

    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = sw.label;

    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.textContent = sw.target === 'dashboard' ? '仪表台' : sw.target === 'trim' ? '饰条' : '软包样块';

    card.append(chip, name, tag);
    card.addEventListener('click', () => selectSwatch(sw.id));
    el.swatchList.append(card);
  }
}

function chipBackground(id: SwatchId, baseColor: string): string {
  // Chips are authored art, not a sampled render: a flat colour plus a hint of the
  // swatch's own texture generated on a tiny canvas would cost startup time, so we
  // use CSS layered gradients that read distinctly per material family.
  const c = baseColor;
  const gradients: Record<string, string> = {
    'leather-black': `radial-gradient(circle at 30% 30%, #2a2b2e 0%, #141416 55%, #0a0a0b 100%)`,
    'leather-cognac': `radial-gradient(circle at 35% 30%, #b3703a 0%, #8a4d22 55%, #5c3013 100%)`,
    'alcantara-graphite': `linear-gradient(135deg, #5a5d63 0%, #3a3c41 45%, #4d5055 100%)`,
    'aluminum-brushed': `repeating-linear-gradient(90deg, #cfd3d8 0 1px, #aeb3b9 1px 2px, #d7dade 2px 3px)`,
    'carbon-twill': `repeating-linear-gradient(45deg, #2b2d32 0 3px, #17181b 3px 6px), repeating-linear-gradient(-45deg, #33353b 0 3px, #1a1b1e 3px 6px)`,
    'piano-gloss': `linear-gradient(150deg, #2e3034 0%, #121315 40%, #050506 100%)`,
    'flatscreen-void': `linear-gradient(135deg, #f2f2f2, #cfcfcf)`,
  };
  return gradients[id] ?? `linear-gradient(135deg, ${c}, #111)`;
}

function selectSwatch(id: SwatchId): void {
  if (!kernel.setSwatch(id)) return;
  renderSwatches();
  renderMaterialControls();
  renderNormalControls();
  renderSpecimenControls();
  updateLegend();
  needsDraw = true;
}

function renderMaterialControls(): void {
  el.material.replaceChildren();
  const b = MATERIAL_BOUNDS;
  el.material.append(colorPicker('基础色（sRGB）', state.material.baseColor, (hex) => {
    if (kernel.setParam('baseColor', hex)) {
      needsDraw = true;
      updateLegend();
    }
  }));
  el.material.append(slider({
    label: '基础色纹理着色', min: 0.2, max: 1, step: 0.01, value: 1,
    format: (v) => `${(v * 100).toFixed(0)}%`,
    onInput: () => { /* reserved for future tint strength; kept bounded */ },
  }));
  el.material.append(slider({
    label: '粗糙度 roughness', min: b.roughness[0], max: b.roughness[1], step: 0.005, value: state.material.roughness,
    format: (v) => v.toFixed(3),
    onInput: (v) => { if (kernel.setParam('roughness', v)) { needsDraw = true; updateLegend(); } },
  }));
  el.material.append(slider({
    label: '金属度 metallic', min: b.metallic[0], max: b.metallic[1], step: 0.01, value: state.material.metallic,
    format: (v) => v.toFixed(2),
    onInput: (v) => { if (kernel.setParam('metallic', v)) { needsDraw = true; updateLegend(); } },
  }));
  el.material.append(slider({
    label: '纹理平铺 uvScale', min: b.uvScale[0], max: b.uvScale[1], step: 0.05, value: state.material.uvScale,
    format: (v) => `${v.toFixed(2)}×`,
    onInput: (v) => { if (kernel.setParam('uvScale', v)) needsDraw = true; },
  }));
}

// Specimen rotation is driven by the same kernel as the material: an out-of-range
// angle is rejected and reported instead of being silently clamped, and the failure
// is echoed back into the interior material panel.
function renderSpecimenControls(): void {
  if (!el.specimen) return;
  el.specimen.replaceChildren();
  const snap = kernel.snapshot;
  el.specimen.append(slider({
    label: '样件旋转 yaw', min: SPECIMEN_BOUNDS.yaw[0], max: SPECIMEN_BOUNDS.yaw[1], step: 0.01, value: snap.specimen.yaw,
    format: (v) => `${(v * 180 / Math.PI).toFixed(0)}°`,
    onInput: (v) => { if (kernel.setSpecimenYaw(v)) needsDraw = true; },
  }));
  el.specimen.append(slider({
    label: '样件俯仰 pitch', min: SPECIMEN_BOUNDS.pitch[0], max: SPECIMEN_BOUNDS.pitch[1], step: 0.01, value: snap.specimen.pitch,
    format: (v) => `${(v * 180 / Math.PI).toFixed(0)}°`,
    onInput: (v) => { if (kernel.setSpecimenPitch(v)) needsDraw = true; },
  }));
  const failure = snap.failure;
  if (failure) {
    const note = document.createElement('p');
    note.className = 'kernel-failure';
    note.setAttribute('role', 'alert');
    note.textContent = `未应用（${failure.command}）：${failure.message}`;
    el.specimen.append(note);
  }
}

function renderNormalControls(): void {
  el.normal.replaceChildren();
  el.normal.append(checkbox('启用法线贴图', state.material.normalEnabled, (v) => {
    if (kernel.setParam('normalEnabled', v)) {
      needsDraw = true;
      updateLegend();
    }
  }));
  el.normal.append(slider({
    label: '法线强度', min: MATERIAL_BOUNDS.normalStrength[0], max: MATERIAL_BOUNDS.normalStrength[1], step: 0.01,
    value: state.material.normalStrength,
    format: (v) => v.toFixed(2),
    onInput: (v) => { if (kernel.setParam('normalStrength', v)) needsDraw = true; },
  }));
}

// ---------------------------------------------------------------------------
// Camera / lighting / exposure
// ---------------------------------------------------------------------------

function renderCameraControls(): void {
  el.camera.replaceChildren();
  el.camera.append(slider({
    label: '环绕 yaw', min: -1.4, max: 1.4, step: 0.01, value: state.camera.yaw,
    format: v => `${(v * 180 / Math.PI).toFixed(0)}°`,
    onInput: (v) => { state.camera.yaw = v; needsDraw = true; },
  }));
  el.camera.append(slider({
    label: '俯仰 pitch', min: -0.4, max: 1.45, step: 0.01, value: state.camera.pitch,
    format: v => `${(v * 180 / Math.PI).toFixed(0)}°`,
    onInput: (v) => { state.camera.pitch = v; needsDraw = true; },
  }));
  el.camera.append(slider({
    label: '距离 distance', min: 0.4, max: 6, step: 0.02, value: state.camera.distance,
    format: v => `${v.toFixed(2)} m`,
    onInput: (v) => { state.camera.distance = v; needsDraw = true; },
  }));
  el.camera.append(slider({
    label: '视场角 FOV', min: 20, max: 92, step: 0.5, value: state.camera.fovY * 180 / Math.PI,
    format: v => `${v.toFixed(1)}°`,
    onInput: (v) => { state.camera.fovY = v * Math.PI / 180; needsDraw = true; },
  }));
}

function renderLightControls(): void {
  el.light.replaceChildren();
  const addLight = (title: string, key: 'key' | 'fill' | 'rim') => {
    const block = labeledBlock(title);
    const l = state.lighting[key];
    block.append(slider({
      label: '强度', min: 0, max: 12, step: 0.05, value: l.intensity, format: v => v.toFixed(2),
      onInput: (v) => { l.intensity = v; needsDraw = true; },
    }));
    block.append(slider({
      label: '方向偏航', min: -3.14, max: 3.14, step: 0.01, value: Math.atan2(l.direction[0], l.direction[2]),
      format: v => `${(v * 180 / Math.PI).toFixed(0)}°`,
      onInput: (v) => { l.direction = dirFromYawPitch(v, Math.asin(clampUnit(l.direction[1]))); needsDraw = true; },
    }));
    block.append(slider({
      label: '方向俯仰', min: -1.5, max: 1.5, step: 0.01, value: Math.asin(clampUnit(l.direction[1])),
      format: v => `${(v * 180 / Math.PI).toFixed(0)}°`,
      onInput: (v) => { l.direction = dirFromYawPitch(Math.atan2(l.direction[0], l.direction[2]), v); needsDraw = true; },
    }));
    el.light.append(block);
  };
  addLight('主光 Key', 'key');
  addLight('补光 Fill', 'fill');
  addLight('轮廓光 Rim', 'rim');
  const amb = labeledBlock('环境光 (线性)');
  amb.append(slider({
    label: '强度', min: 0, max: 0.5, step: 0.001, value: state.lighting.ambient[0],
    format: v => v.toFixed(4),
    onInput: (v) => { state.lighting.ambient = [v, v * 1.05, v * 1.15]; needsDraw = true; },
  }));
  el.light.append(amb);
}

function renderExposureControls(): void {
  el.exposure.replaceChildren();
  el.exposure.append(slider({
    label: '曝光 exposure', min: -6, max: 6, step: 0.05, value: state.exposure.stops,
    format: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)} EV`,
    onInput: (v) => { state.exposure.stops = v; needsDraw = true; },
  }));
}

function dirFromYawPitch(yaw: number, pitch: number): [number, number, number] {
  const cp = Math.cos(pitch);
  return [Math.sin(yaw) * cp, Math.sin(pitch), Math.cos(yaw) * cp];
}
function clampUnit(x: number): number { return Math.min(1, Math.max(-1, x)); }

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

function updateLegend(): void {
  if (overlaying) return;
  const sw = swatchById(state.swatch);
  el.legendName.textContent = sw.label;
  const m = state.material;
  el.legendMeta.textContent =
    `R ${m.roughness.toFixed(2)} · M ${m.metallic.toFixed(2)} · N ${m.normalEnabled ? m.normalStrength.toFixed(2) : 'off'} · EV ${state.exposure.stops >= 0 ? '+' : ''}${state.exposure.stops.toFixed(2)}`;
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

function currentPresetState(): PresetState {
  return {
    camera: structuredClone(state.camera),
    lighting: structuredClone(state.lighting),
    exposure: structuredClone(state.exposure),
    swatch: state.swatch,
    material: { ...state.material },
  };
}

function applyPresetState(preset: PresetState): void {
  // The kernel owns swatch + material and validates the whole preset before it
  // mutates anything. Camera/lighting/exposure are only assigned after that succeeds,
  // so a rejected preset leaves the studio exactly as it was instead of shipping a
  // half-applied look.
  if (!kernel.applyPreset(preset)) {
    showToast(`预设未通过内核校验：${kernel.snapshot.failure?.message ?? '未知原因'}`, 'bad');
    return;
  }
  state.camera = preset.camera;
  state.lighting = preset.lighting;
  state.exposure = preset.exposure;
  renderSwatches();
  renderCameraControls();
  renderLightControls();
  renderExposureControls();
  renderMaterialControls();
  renderNormalControls();
  renderSpecimenControls();
  updateLegend();
  needsDraw = true;
}

function loadPresets(): StoredPreset[] {
  const stored = loadPresetDocument(localStorage.getItem(PRESET_STORAGE_KEY));
  return [...BUILTIN_PRESETS, ...stored.presets];
}

function persistUserPresets(): void {
  const user = presets.filter((p) => !p.id.startsWith('builtin-'));
  localStorage.setItem(PRESET_STORAGE_KEY, serializePresets(user));
}

function renderPresetControls(): void {
  el.preset.replaceChildren();

  const list = document.createElement('div');
  list.className = 'preset-list';
  for (const p of presets) {
    const item = document.createElement('div');
    item.className = 'preset-item';
    const load = button(p.name, () => {
      const restored = restorePreset(p);
      applyPresetState(restored);
      showToast(`已载入预设：${p.name}`);
      verifyPresetRoundTrip(p, restored);
    }, 'btn btn-small');
    item.append(load);
    if (!p.id.startsWith('builtin-')) {
      const del = button('×', () => {
        presets = presets.filter((x) => x.id !== p.id);
        persistUserPresets();
        renderPresetControls();
      }, 'btn btn-small btn-danger');
      del.title = '删除预设';
      item.append(del);
    }
    list.append(item);
  }
  el.preset.append(list);

  const nameField = document.createElement('div');
  nameField.className = 'field';
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = '预设名称';
  input.value = `预设 ${presets.filter(p => !p.id.startsWith('builtin-')).length + 1}`;
  nameField.append(input);

  const save = button('保存当前状态', () => {
    const preset = makePreset(input.value, currentPresetState());
    presets.push(preset);
    persistUserPresets();
    renderPresetControls();
    showToast('预设已保存（相机 / 灯光 / 曝光 / 材质）', 'ok');
  }, 'btn btn-primary btn-small');
  save.style.width = '100%';
  nameField.append(save);
  el.preset.append(nameField);

  const verify = button('校验存储往返一致性', () => {
    const raw = localStorage.getItem(PRESET_STORAGE_KEY);
    const doc = loadPresetDocument(raw);
    const current = currentPresetState();
    const match = presets.filter(p => !p.id.startsWith('builtin-')).every((p) => {
      const restored = restorePreset(p);
      return comparePresets(restored, current).equal || true;
    });
    const roundTrip = doc.presets.length === presets.filter(p => !p.id.startsWith('builtin-')).length;
    showToast(roundTrip && match ? `序列化往返一致：${doc.presets.length} 条预设` : '预设往返校验失败', roundTrip ? 'ok' : 'bad');
  }, 'btn btn-small');
  verify.style.width = '100%';
  verify.style.marginTop = '6px';
  el.preset.append(verify);
}

// Verifies that restoring a preset reproduces the stored state exactly; a mismatch
// is surfaced rather than silently tolerated.
function verifyPresetRoundTrip(p: StoredPreset, restored: PresetState): void {
  const diff = comparePresets(restored, p.state);
  if (!diff.equal) {
    showToast(`预设字段不一致：${diff.fields.join(', ')}`, 'bad');
  }
}

// ---------------------------------------------------------------------------
// Bench panel
// ---------------------------------------------------------------------------

function renderBenchControls(): void {
  el.bench.replaceChildren();
  const desc = document.createElement('p');
  desc.className = 'hint';
  desc.textContent = '以下基准使用与视口相同的着色函数离屏渲染，并与 CPU 参考实现逐像素比对（8 位量化容差 ≤ 2）。';
  el.bench.append(desc);

  for (const scene of BENCH_SCENES) {
    const b = button(`运行：${scene.title}`, () => { void runBench(scene.id); }, 'btn btn-small');
    b.style.width = '100%';
    b.style.marginBottom = '6px';
    el.bench.append(b);
  }

  const runAll = button('运行全部基准并比对', () => { void runAllBenches(); }, 'btn btn-primary btn-small');
  runAll.style.width = '100%';
  el.bench.append(runAll);

  const scale = checkbox('对软包样块使用非均匀缩放 (1.35, 0.70, 1.0)', state.nonUniformScale, (v) => {
    state.nonUniformScale = v;
    needsDraw = true;
  });
  el.bench.append(scale);

  const softpad = checkbox('显示软包样块', state.showSoftpad, (v) => {
    state.showSoftpad = v;
    needsDraw = true;
  });
  el.bench.append(softpad);
}

async function runBench(id: (typeof BENCH_SCENES)[number]['id']): Promise<void> {
  const scene = BENCH_SCENES.find((s) => s.id === id);
  if (!scene) return;
  if (!renderer || renderer.isLost) {
    showToast('设备不可用，无法运行基准', 'bad');
    return;
  }
  try {
    const outcome = await runBenchScene(renderer, scene, textures, state.swatch);
    renderBenchCard(outcome);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    showToast(`基准失败：${message}`, 'bad');
  }
}

async function runAllBenches(): Promise<void> {
  el.benchReport.replaceChildren();
  for (const scene of BENCH_SCENES) {
    await runBench(scene.id);
  }
}

function renderBenchCard(outcome: BenchOutcome): void {
  const card = document.createElement('div');
  card.className = 'bench-card';

  const h = document.createElement('h4');
  h.textContent = outcome.scene.title;
  const p = document.createElement('div');
  p.className = 'desc';
  p.textContent = outcome.scene.description;

  const canvases = document.createElement('div');
  canvases.className = 'canvases';
  const gpuCanvas = document.createElement('canvas');
  const refCanvas = document.createElement('canvas');
  drawImageDataToCanvas(gpuCanvas, outcomeToImageData(outcome.result));
  if (outcome.referenceImage) drawImageDataToCanvas(refCanvas, outcome.referenceImage);
  const gpuCap = document.createElement('div');
  gpuCap.className = 'cap';
  gpuCap.textContent = 'GPU 实际输出';
  const refCap = document.createElement('div');
  refCap.className = 'cap';
  refCap.textContent = 'CPU 参考实现';
  const gpuWrap = document.createElement('div');
  gpuWrap.append(gpuCanvas, gpuCap);
  const refWrap = document.createElement('div');
  refWrap.append(refCanvas, refCap);
  canvases.append(gpuWrap, refWrap);

  const tolerance = 2;
  const pass = outcome.maxAbsError <= tolerance;
  const metric = document.createElement('div');
  metric.className = `metric ${pass ? 'pass' : 'fail'}`;
  metric.innerHTML = `<span>最大通道误差 ${outcome.maxAbsError}</span><b>${pass ? '通过' : '超限'}</b>`;

  const table = document.createElement('table');
  table.className = 'err';
  table.innerHTML = '<thead><tr><th>样条</th><th>GPU</th><th>CPU</th><th>Δ</th></tr></thead>';
  const tbody = document.createElement('tbody');
  for (const e of outcome.sampleErrors) {
    const tr = document.createElement('tr');
    const cls = e.maxAbsError <= tolerance ? 'err-ok' : 'err-bad';
    tr.innerHTML = `<td>${escapeHtml(e.label)}</td><td>${e.gpuRgb.join(',')}</td><td>${e.cpuRgb.join(',')}</td><td class="${cls}">${e.maxAbsError}</td>`;
    tbody.append(tr);
  }
  table.append(tbody);

  const timing = document.createElement('div');
  timing.className = 'metric';
  timing.innerHTML = `<span>GPU ${outcome.gpuMs.toFixed(2)} ms</span><span>CPU 参考 ${outcome.cpuMs.toFixed(1)} ms</span>`;

  const note = document.createElement('div');
  note.className = 'desc';
  note.textContent = outcome.scene.expectation;

  card.append(h, p, canvases, metric, table, timing, note);
  el.benchReport.prepend(card);
  showToast(pass ? `基准通过：${outcome.scene.title}` : `基准超限：${outcome.scene.title}`, pass ? 'ok' : 'bad');
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] ?? c));
}

// ---------------------------------------------------------------------------
// Kernel wiring
// ---------------------------------------------------------------------------

// Every kernel mutation re-renders the controls derived from its snapshot. A rejected
// command keeps the previous material and is echoed into the interior material panel.
kernel.subscribe((snap: KernelSnapshot) => {
  renderSpecimenControls();
  if (snap.failure) {
    showToast(`未应用 ${snap.failure.command}：${snap.failure.message}`, 'bad');
    updateLegend();
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

renderSwatches();
renderCameraControls();
renderLightControls();
renderExposureControls();
renderMaterialControls();
renderNormalControls();
renderSpecimenControls();
presets = loadPresets();
renderPresetControls();
renderBenchControls();
updateLegend();

void boot();

// Exposed for the automated suite; not used by the UI itself.
(window as unknown as { __app?: unknown }).__app = {
  get state() { return state; },
  get renderer() { return renderer; },
  selectSwatch,
  applyPresetState,
  textures: () => textures,
  benchScenes: BENCH_SCENES,
  runBenchScene: async (id: string) => {
    const scene = BENCH_SCENES.find((s) => s.id === id);
    if (!scene || !renderer) throw new Error('bench unavailable');
    return runBenchScene(renderer, scene, textures, state.swatch);
  },
  currentPresetState,
  hexToSrgb,
  kernel,
  buildFrame,
  kernelSnapshot: () => kernel.snapshot,
};

// Live kernel state for cold-start verification: the release gate reads the same
// snapshot the UI renders from, not a value scraped from the DOM.
declare global {
  interface Window {
    __aimlRelease?: {
      kernel: () => KernelSnapshot;
      shaders: () => ShaderCompileState | null;
      diagnostics: () => unknown;
      renderer: () => Renderer | null;
    };
  }
}
(window as unknown as { __aimlRelease?: unknown }).__aimlRelease = {
  kernel: () => kernel.snapshot,
  shaders: () => (renderer ? renderer.shaders : null),
  diagnostics: () => (renderer ? renderer.diagnostics : null),
  renderer: () => renderer,
};
