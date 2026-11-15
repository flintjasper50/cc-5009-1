import { SCENE_WGSL, COMPOSITE_WGSL } from './shaders.ts';
import {
  createGpuContext,
  formatBytes,
  textureBytes,
  ResourceLedger,
  type GpuContext,
  type GpuDiagnostics,
  type ResourceHandle,
  type ShaderCompileInfo,
  type ShaderCompileState,
} from './device.ts';
import {
  buildSceneGeometry,
  type MeshData,
} from '../scene/geometry.ts';
import type { TextureSet } from '../scene/textures.ts';
import {
  clampParams,
  MATERIAL_BOUNDS,
  type CameraState,
  type LightingState,
  type MaterialParams,
  type SwatchId,
} from '../scene/material.ts';
import {
  mat4LookAt,
  mat4Multiply,
  mat4Perspective,
  mat4Identity,
  mat4FromTRS,
  normalMatrixFromModel,
  type Mat4,
} from '../math/linalg.ts';
import { hexToSrgb, srgbToLinear, type RGB } from '../math/color.ts';

export type MeshKey = 'dashboard' | 'trim' | 'softpad' | 'bench';

export interface DrawItem {
  mesh: MeshKey;
  model: Mat4;
  params: MaterialParams;
  /** Linear base tint multiplied with the albedo texture. */
  tint: RGB;
  /** Swatch whose textures feed this item. Never inferred from another item. */
  swatchId: SwatchId;
  textureSet: TextureSet;
  /** When true the albedo texture is skipped (benchmark solid-colour path). */
  solid?: boolean;
}

export interface FrameState {
  camera: CameraState;
  lighting: LightingState;
  exposure: number;
  items: DrawItem[];
  diffuseOnly?: boolean;
}

export interface GpuMesh {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
  bytes: number;
}

interface MaterialSlot {
  albedo: GPUTexture;
  normal: GPUTexture;
  bindGroup: GPUBindGroup;
  bytes: number;
  handle: ResourceHandle;
}

interface ItemBuffers {
  buffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  handle: ResourceHandle;
}

export interface BenchSample {
  roughness: number;
  metallic: number;
  /** sRGB hex of the solid base colour; empty when textured. */
  baseColorHex: string;
  xOffset: number;
  xScale: number;
  label: string;
}

export interface BenchRequest {
  kind: 'diffuse' | 'flat-normal' | 'roughness' | 'boundaries';
  width: number;
  height: number;
  lighting: LightingState;
  exposure: number;
  baseColorHex: string;
  normalStrength: number;
  roughness: number;
  metallic: number;
  diffuseOnly: boolean;
  normalEnabled: boolean;
  /** solid = skip colour texture, use tint only. */
  solid: boolean;
  swatchId: SwatchId;
  textureSet: TextureSet;
}

export interface BenchResult {
  width: number;
  height: number;
  pixels: Uint8Array;
  samples: BenchSample[];
  gpuMs: number;
}

const FRAME_UNIFORM_BYTES = 224;
const MODEL_UNIFORM_BYTES = 160;
const MAX_ITEMS = 8;

export class Renderer {
  private ctx: GpuContext;
  private ledger = new ResourceLedger();
  private canvas: HTMLCanvasElement;
  private canvasContext: GPUCanvasContext;
  private lost = false;
  private disposed = false;
  private onLost: (message: string) => void = () => {};

  private frameLayout!: GPUBindGroupLayout;
  private modelLayout!: GPUBindGroupLayout;
  private texLayout!: GPUBindGroupLayout;
  private compositeLayout!: GPUBindGroupLayout;
  private frameBuffer!: GPUBuffer;
  private frameHandle!: ResourceHandle;
  private frameBindGroup!: GPUBindGroup;
  private compositeBuffer!: GPUBuffer;
  private compositeBindGroup!: GPUBindGroup;
  private sampler!: GPUSampler;

  private studioPipeline!: GPURenderPipeline;
  private benchPipeline!: GPURenderPipeline;
  private compositePipeline!: GPURenderPipeline;

  private meshes = new Map<MeshKey, GpuMesh>();
  private slots = new Map<SwatchId, MaterialSlot>();
  private retired: { slot: MaterialSlot; frames: number }[] = [];
  private items = new Map<number, ItemBuffers>();

  private hdrTexture: GPUTexture | null = null;
  private hdrHandle: ResourceHandle | null = null;
  private depthTexture: GPUTexture | null = null;
  private depthHandle: ResourceHandle | null = null;
  private width = 0;
  private height = 0;

  private querySet: GPUQuerySet | null = null;
  private queryResolve: GPUBuffer | null = null;
  private timestampPending = false;
  private gpuMs: number | null = null;

  private compileState: ShaderCompileState = { ready: false, modules: [], pipelines: [], ok: false, unsupported: false };
  private compilePromise: Promise<ShaderCompileState> | null = null;
  private resolveCompile: ((state: ShaderCompileState) => void) | null = null;

  static async create(canvas: HTMLCanvasElement, onLost: (message: string) => void): Promise<Renderer> {
    const ctx = await createGpuContext();
    const renderer = new Renderer(canvas, ctx);
    await renderer.init(onLost);
    return renderer;
  }

  private constructor(canvas: HTMLCanvasElement, ctx: GpuContext) {
    this.canvas = canvas;
    this.ctx = ctx;
    const ctx2d = canvas.getContext('webgpu');
    if (!ctx2d) throw new Error('无法获取 WebGPU 画布上下文');
    this.canvasContext = ctx2d;
  }

  get diagnostics(): GpuDiagnostics { return this.ctx.diagnostics; }
  get gpuTimeMs(): number | null { return this.gpuMs; }
  get shaders(): ShaderCompileState { return this.compileState; }
  get resourceBytes(): number { return this.ledger.total(); }
  get resourceCounts(): Record<string, number> { return this.ledger.counts(); }
  get isLost(): boolean { return this.lost; }
  get isDisposed(): boolean { return this.disposed; }
  get size(): { width: number; height: number } { return { width: this.width, height: this.height }; }

  private async init(onLost: (message: string) => void): Promise<void> {
    this.onLost = onLost;
    const { device } = this.ctx;
    const geometry = buildSceneGeometry();
    this.uploadMesh('dashboard', geometry.dashboard);
    this.uploadMesh('trim', geometry.trim);
    this.uploadMesh('softpad', geometry.softpad);
    this.uploadMesh('bench', geometry.bench);

    this.frameLayout = device.createBindGroupLayout({
      label: 'frame-layout',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    this.modelLayout = device.createBindGroupLayout({
      label: 'model-layout',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    this.texLayout = device.createBindGroupLayout({
      label: 'texture-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.compositeLayout = device.createBindGroupLayout({
      label: 'composite-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });

    const sceneModule = device.createShaderModule({ label: 'scene-shader', code: SCENE_WGSL });
    const compositeModule = device.createShaderModule({ label: 'composite-shader', code: COMPOSITE_WGSL });

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 48,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x4' },
        { shaderLocation: 3, offset: 40, format: 'float32x2' },
      ],
    };

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.modelLayout, this.texLayout] });

    const studioPipeline = device.createRenderPipelineAsync({
      label: 'studio-hdr',
      layout: pipelineLayout,
      vertex: { module: sceneModule, entryPoint: 'vs_main', buffers: [vertexLayout] },
      fragment: { module: sceneModule, entryPoint: 'fs_main', targets: [{ format: this.ctx.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    const benchPipeline = device.createRenderPipelineAsync({
      label: 'bench-rgba8',
      layout: pipelineLayout,
      vertex: { module: sceneModule, entryPoint: 'vs_main', buffers: [vertexLayout] },
      fragment: { module: sceneModule, entryPoint: 'fs_bench', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    const compositePipeline = device.createRenderPipelineAsync({
      label: 'composite',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] }),
      vertex: { module: compositeModule, entryPoint: 'vs_fullscreen' },
      fragment: { module: compositeModule, entryPoint: 'fs_composite', targets: [{ format: this.ctx.canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    });

    // A rejected pipeline promise must still record a failed compile state rather than
    // aborting init before the state exists.
    const settled = await Promise.allSettled([studioPipeline, benchPipeline, compositePipeline]);
    const labels = ['studio-hdr', 'bench-rgba8', 'composite'] as const;
    const failures: { label: string; ok: boolean; error?: string }[] = [];
    settled.forEach((result, i) => {
      if (result.status === 'fulfilled') {
        const pipeline = result.value;
        if (labels[i] === 'studio-hdr') this.studioPipeline = pipeline;
        else if (labels[i] === 'bench-rgba8') this.benchPipeline = pipeline;
        else this.compositePipeline = pipeline;
      } else {
        failures.push({ label: labels[i], ok: false, error: result.reason instanceof Error ? result.reason.message : String(result.reason) });
      }
    });
    void this.collectCompilation(
      [
        { label: 'scene-shader', module: sceneModule },
        { label: 'composite-shader', module: compositeModule },
      ],
      labels.map((label, i) => ({ label, result: settled[i] })),
      failures,
    );

    this.sampler = device.createSampler({
      label: 'material-sampler',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    this.frameBuffer = device.createBuffer({
      label: 'frame-uniform',
      size: FRAME_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.frameHandle = this.ledger.track('uniform', FRAME_UNIFORM_BYTES, () => this.frameBuffer.destroy());
    this.frameBindGroup = device.createBindGroup({
      layout: this.frameLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameBuffer } }],
    });

    this.compositeBuffer = device.createBuffer({
      label: 'composite-uniform',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.ledger.track('uniform', 16, () => this.compositeBuffer.destroy());

    if (this.ctx.diagnostics.timestampSupported) {
      this.querySet = device.createQuerySet({ label: 'frame-timestamps', type: 'timestamp', count: 2 });
      this.queryResolve = device.createBuffer({
        label: 'timestamp-resolve',
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.ledger.track('timestamp-query', 16, () => this.querySet?.destroy());
    }

    this.canvasContext.configure({
      device,
      format: this.ctx.canvasFormat,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.ctx.device.lost.then((info) => {
      this.lost = true;
      onLost(`${info.reason}: ${info.message || '（无详细信息）'}`);
    });
  }

  private finishCompilation(state: ShaderCompileState): void {
    this.compileState = state;
    this.resolveCompile?.(state);
    this.resolveCompile = null;
  }

  async awaitCompilation(): Promise<ShaderCompileState> {
    if (this.compileState.ready) return this.compileState;
    if (!this.compilePromise) {
      this.compilePromise = new Promise<ShaderCompileState>((resolve) => { this.resolveCompile = resolve; });
    }
    return this.compilePromise;
  }

  // Compilation is an explicit build gate: a module that fails to compile or a
  // pipeline that fails to create keeps the release unverified instead of letting
  // the app present a blank frame as if it had rendered.
  private async collectCompilation(
    modules: { label: string; module: GPUShaderModule }[],
    pipelines: { label: string; result: PromiseSettledResult<GPURenderPipeline> }[],
    preFailures: { label: string; ok: boolean; error?: string }[] = [],
  ): Promise<void> {
    let unsupported = false;
    const reports: ShaderCompileInfo[] = [];
    for (const entry of modules) {
      let info: GPUCompilationInfo;
      try {
        info = await entry.module.getCompilationInfo();
      } catch (error) {
        if (error instanceof TypeError) {
          unsupported = true;
          reports.push({ label: entry.label, ok: true, messages: [{ type: 'info', message: '浏览器未实现 getCompilationInfo', lineNum: 0, linePos: 0 }] });
          continue;
        }
        reports.push({
          label: entry.label,
          ok: false,
          messages: [{ type: 'error', message: `getCompilationInfo 失败：${error instanceof Error ? error.message : String(error)}`, lineNum: 0, linePos: 0 }],
        });
        continue;
      }
      const messages = info.messages.map((m) => ({
        type: m.type,
        message: m.message,
        lineNum: m.lineNum,
        linePos: m.linePos,
      }));
      reports.push({ label: entry.label, ok: !messages.some((m) => m.type === 'error'), messages });
    }

    const pipeResults: { label: string; ok: boolean; error?: string }[] = [...preFailures];
    for (const entry of pipelines) {
      if (entry.result.status === 'fulfilled') {
        pipeResults.push({ label: entry.label, ok: true });
      } else if (!pipeResults.some((p) => p.label === entry.label)) {
        const reason = entry.result.reason;
        pipeResults.push({ label: entry.label, ok: false, error: reason instanceof Error ? reason.message : String(reason) });
      }
    }

    const ok = reports.every((r) => r.ok) && pipeResults.every((p) => p.ok);
    this.finishCompilation({ ready: true, modules: reports, pipelines: pipeResults, ok, unsupported });
  }

  private uploadMesh(key: MeshKey, mesh: MeshData): void {
    const { device } = this.ctx;
    const interleaved = new Float32Array(mesh.vertexCount * 12);
    for (let i = 0; i < mesh.vertexCount; i++) {
      const o = i * 12;
      interleaved[o] = mesh.positions[i * 3];
      interleaved[o + 1] = mesh.positions[i * 3 + 1];
      interleaved[o + 2] = mesh.positions[i * 3 + 2];
      interleaved[o + 3] = mesh.normals[i * 3];
      interleaved[o + 4] = mesh.normals[i * 3 + 1];
      interleaved[o + 5] = mesh.normals[i * 3 + 2];
      interleaved[o + 6] = mesh.tangents[i * 4];
      interleaved[o + 7] = mesh.tangents[i * 4 + 1];
      interleaved[o + 8] = mesh.tangents[i * 4 + 2];
      interleaved[o + 9] = mesh.tangents[i * 4 + 3];
      interleaved[o + 10] = mesh.uvs[i * 2];
      interleaved[o + 11] = mesh.uvs[i * 2 + 1];
    }
    const vertexBuffer = device.createBuffer({
      label: `${key}-vertices`,
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, interleaved);
    const indexBuffer = device.createBuffer({
      label: `${key}-indices`,
      size: mesh.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, mesh.indices);
    const bytes = interleaved.byteLength + mesh.indices.byteLength;
    this.ledger.track('vertex+index', bytes, () => { vertexBuffer.destroy(); indexBuffer.destroy(); });
    this.meshes.set(key, { vertexBuffer, indexBuffer, indexCount: mesh.indexCount, bytes });
  }

  // Material slots are keyed by swatch id and hold their own immutable textures.
  // No draw can pick up another swatch's data because every item carries its swatch id.
  private slotFor(item: DrawItem): MaterialSlot {
    const existing = this.slots.get(item.swatchId);
    if (existing) return existing;
    const { device } = this.ctx;
    const t = item.textureSet;

    const mipLevelCount = t.albedo.levels.length;
    const albedo = device.createTexture({
      label: `${item.swatchId}-albedo`,
      size: [t.albedo.width, t.albedo.height],
      mipLevelCount,
      format: 'rgba8unorm',
      viewFormats: ['rgba8unorm-srgb'],
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const normal = device.createTexture({
      label: `${item.swatchId}-normal`,
      size: [t.normal.width, t.normal.height],
      mipLevelCount: t.normal.levels.length,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // Upload each mip explicitly; GPU-side mip generation for normals would average
    // encoded directions as if they were colours, which is wrong for direction data.
    for (let level = 0; level < t.albedo.levels.length; level++) {
      const lw = Math.max(1, t.albedo.width >> level);
      const lh = Math.max(1, t.albedo.height >> level);
      device.queue.writeTexture(
        { texture: albedo, mipLevel: level },
        t.albedo.levels[level],
        { bytesPerRow: lw * 4, rowsPerImage: lh },
        [lw, lh, 1],
      );
    }
    for (let level = 0; level < t.normal.levels.length; level++) {
      const lw = Math.max(1, t.normal.width >> level);
      const lh = Math.max(1, t.normal.height >> level);
      device.queue.writeTexture(
        { texture: normal, mipLevel: level },
        t.normal.levels[level],
        { bytesPerRow: lw * 4, rowsPerImage: lh },
        [lw, lh, 1],
      );
    }

    // Albedo uses an sRGB view so hardware decodes it to linear light. The normal
    // map is sampled through a plain unorm view: direction bytes, not colour.
    const bindGroup = device.createBindGroup({
      layout: this.texLayout,
      entries: [
        { binding: 0, resource: albedo.createView({ format: 'rgba8unorm-srgb' }) },
        { binding: 1, resource: normal.createView({ format: 'rgba8unorm' }) },
        { binding: 2, resource: this.sampler },
      ],
    });

    const bytes = textureBytes(t.albedo.width, t.albedo.height, 'rgba8unorm', t.albedo.levels.length) +
      textureBytes(t.normal.width, t.normal.height, 'rgba8unorm', t.normal.levels.length);
    const handle = this.ledger.track('swatch-texture', bytes, () => { albedo.destroy(); normal.destroy(); });
    const slot: MaterialSlot = { albedo, normal, bindGroup, bytes, handle };
    this.slots.set(item.swatchId, slot);
    return slot;
  }

  // Slots not referenced by the current frame are retired after two frames so in-flight
  // command buffers still sampling them finish before the textures are destroyed.
  private retainSlots(active: Set<SwatchId>): void {
    for (const [id, slot] of [...this.slots]) {
      if (active.has(id)) continue;
      this.slots.delete(id);
      this.retired.push({ slot, frames: 2 });
    }
  }

  private itemBuffers(index: number): ItemBuffers {
    const existing = this.items.get(index);
    if (existing) return existing;
    const { device } = this.ctx;
    const buffer = device.createBuffer({
      label: `model-uniform-${index}`,
      size: MODEL_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      layout: this.modelLayout,
      entries: [{ binding: 0, resource: { buffer } }],
    });
    const handle = this.ledger.track('uniform', MODEL_UNIFORM_BYTES, () => buffer.destroy());
    const rec: ItemBuffers = { buffer, bindGroup, handle };
    this.items.set(index, rec);
    return rec;
  }

  private ensureTargets(width: number, height: number): void {
    if (this.hdrTexture && this.width === width && this.height === height) return;
    const { device } = this.ctx;
    this.hdrHandle?.release();
    this.depthHandle?.release();
    this.hdrTexture = device.createTexture({
      label: 'hdr-target',
      size: [width, height],
      format: this.ctx.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthTexture = device.createTexture({
      label: 'depth-target',
      size: [width, height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.hdrHandle = this.ledger.track('hdr-target', textureBytes(width, height, this.ctx.format), () => this.hdrTexture?.destroy());
    this.depthHandle = this.ledger.track('depth-target', textureBytes(width, height, 'depth24plus'), () => this.depthTexture?.destroy());
    this.compositeBindGroup = device.createBindGroup({
      layout: this.compositeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.compositeBuffer } },
        { binding: 1, resource: this.hdrTexture.createView() },
      ],
    });
    this.width = width;
    this.height = height;
  }

  resize(width: number, height: number): void {
    if (this.lost || this.disposed) return;
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    this.canvas.width = w;
    this.canvas.height = h;
    this.ensureTargets(w, h);
  }

  render(state: FrameState): void {
    if (this.lost || this.disposed) return;
    if (this.width === 0 || this.height === 0) this.resize(this.canvas.clientWidth || this.canvas.width, this.canvas.clientHeight || this.canvas.height);
    if (!this.hdrTexture || !this.depthTexture || !this.compositeBindGroup) return;
    const { device } = this.ctx;
    const items = state.items.slice(0, MAX_ITEMS);
    const active = new Set(items.map((i) => i.swatchId));
    this.retainSlots(active);

    for (const item of items) this.slotFor(item);

    this.writeFrameUniform(state);
    for (let i = 0; i < items.length; i++) this.writeModelUniform(this.itemBuffers(i), items[i], state.diffuseOnly === true);
    device.queue.writeBuffer(this.compositeBuffer, 0, new Uint32Array(new Float32Array([
      state.exposure, this.width, this.height, 0,
    ]).buffer));

    const encoder = device.createCommandEncoder({ label: 'studio' });
    const pass = encoder.beginRenderPass({
      label: 'scene-pass',
      colorAttachments: [{
        view: this.hdrTexture.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1,
      },
      timestampWrites: this.querySet ? { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined,
    });
    pass.setPipeline(this.studioPipeline);
    pass.setBindGroup(0, this.frameBindGroup);
    for (let i = 0; i < items.length; i++) {
      const mesh = this.meshes.get(items[i].mesh);
      if (!mesh) continue;
      pass.setBindGroup(1, this.itemBuffers(i).bindGroup);
      pass.setBindGroup(2, this.slotFor(items[i]).bindGroup);
      pass.setVertexBuffer(0, mesh.vertexBuffer);
      pass.setIndexBuffer(mesh.indexBuffer, 'uint32');
      pass.drawIndexed(mesh.indexCount);
    }
    pass.end();

    const composite = encoder.beginRenderPass({
      label: 'composite-pass',
      colorAttachments: [{
        view: this.canvasContext.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    composite.setPipeline(this.compositePipeline);
    composite.setBindGroup(0, this.compositeBindGroup);
    composite.draw(3);
    composite.end();

    device.queue.submit([encoder.finish()]);

    this.drainRetired();
    this.readTimestamps();
  }

  private drainRetired(): void {
    const keep: typeof this.retired = [];
    for (const r of this.retired) {
      r.frames--;
      if (r.frames <= 0) r.slot.handle.release();
      else keep.push(r);
    }
    this.retired = keep;
  }

  private readTimestamps(): void {
    if (!this.querySet || !this.queryResolve || this.timestampPending) return;
    const { device } = this.ctx;
    const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
    enc.copyBufferToBuffer(this.queryResolve, 0, readback, 0, 16);
    device.queue.submit([enc.finish()]);
    this.timestampPending = true;
    readback.mapAsync(GPUMapMode.READ).then(() => {
      const data = new BigUint64Array(readback.getMappedRange());
      const ns = Number(data[1] - data[0]);
      if (ns > 0) this.gpuMs = ns / 1e6;
      readback.unmap();
      readback.destroy();
      this.timestampPending = false;
    }).catch(() => {
      this.timestampPending = false;
      readback.destroy();
    });
  }

  private writeFrameUniform(state: FrameState): void {
    const bytes = new ArrayBuffer(FRAME_UNIFORM_BYTES);
    const f = new Float32Array(bytes);
    const aspect = Math.max(1, this.width) / Math.max(1, this.height);
    const eye = cameraEye(state.camera);
    const vp = mat4Multiply(
      mat4Perspective(state.camera.fovY, aspect, 0.03, 60),
      mat4LookAt(eye, state.camera.target, [0, 1, 0]),
    );
    f.set(vp, 0);
    f[16] = eye[0]; f[17] = eye[1]; f[18] = eye[2]; f[19] = 0;
    f[20] = state.lighting.ambient[0];
    f[21] = state.lighting.ambient[1];
    f[22] = state.lighting.ambient[2];
    f[23] = 0;
    f[24] = state.exposure; f[25] = 0; f[26] = 0; f[27] = 0;
    let o = 28;
    for (const l of [state.lighting.key, state.lighting.fill, state.lighting.rim]) {
      f[o] = l.direction[0]; f[o + 1] = l.direction[1]; f[o + 2] = l.direction[2]; f[o + 3] = l.intensity;
      f[o + 4] = l.color[0]; f[o + 5] = l.color[1]; f[o + 6] = l.color[2]; f[o + 7] = l.angularRadius;
      o += 8;
    }
    f[o] = state.diffuseOnly ? 1 : 0;
    this.ctx.device.queue.writeBuffer(this.frameBuffer, 0, bytes);
  }

  private writeModelUniform(rec: ItemBuffers, item: DrawItem, diffuseOnly: boolean): void {
    const bytes = new ArrayBuffer(MODEL_UNIFORM_BYTES);
    const f = new Float32Array(bytes);
    f.set(item.model, 0);
    const nm = normalMatrixFromModel(item.model);
    // WGSL mat3x3f(c0, c1, c2) takes columns; normalMatrixFromModel returns the
    // transpose(inverse) in row-major order, so pack its columns here.
    f[16] = nm[0]; f[17] = nm[3]; f[18] = nm[6]; f[19] = 0;
    f[20] = nm[1]; f[21] = nm[4]; f[22] = nm[7]; f[23] = 0;
    f[24] = nm[2]; f[25] = nm[5]; f[26] = nm[8]; f[27] = 0;
    f[28] = item.tint[0]; f[29] = item.tint[1]; f[30] = item.tint[2]; f[31] = 0;
    const p = clampParams(item.params);
    f[32] = p.roughness;
    f[33] = p.metallic;
    f[34] = p.normalStrength;
    f[35] = (p.normalEnabled && !diffuseOnly) ? 1 : 0;
    f[36] = Math.min(MATERIAL_BOUNDS.uvScale[1], Math.max(MATERIAL_BOUNDS.uvScale[0], p.uvScale));
    f[37] = item.solid ? 1 : 0;
    this.ctx.device.queue.writeBuffer(rec.buffer, 0, bytes);
  }

  // --- Benchmark path -------------------------------------------------------------
  // Renders the same shade() into rgba8unorm after the display transform, so real
  // pixels can be read back and compared numerically with the CPU reference.
  async renderBench(req: BenchRequest): Promise<BenchResult> {
    if (this.lost || this.disposed) throw new Error('设备已丢失，无法执行基准渲染');
    const { device } = this.ctx;
    const width = Math.max(1, Math.floor(req.width));
    const height = Math.max(1, Math.floor(req.height));

    const strips = benchSamples(req);
    const items = benchItems(req, strips);
    const active = new Set(items.map((i) => i.swatchId));
    this.retainSlots(active);
    for (const it of items) this.slotFor(it);

    this.writeBenchFrameUniform(req);
    for (let i = 0; i < items.length; i++) this.writeModelUniform(this.itemBuffers(i), items[i], req.diffuseOnly);

    const target = device.createTexture({
      label: 'bench-target',
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depth = device.createTexture({
      label: 'bench-depth',
      size: [width, height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const readback = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const start = performance.now();
    const encoder = device.createCommandEncoder({ label: 'bench' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: target.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      depthStencilAttachment: {
        view: depth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1,
      },
    });
    pass.setPipeline(this.benchPipeline);
    pass.setBindGroup(0, this.frameBindGroup);
    for (let i = 0; i < items.length; i++) {
      const mesh = this.meshes.get(items[i].mesh);
      const slot = this.slotFor(items[i]);
      if (!mesh) continue;
      pass.setBindGroup(1, this.itemBuffers(i).bindGroup);
      pass.setBindGroup(2, slot.bindGroup);
      pass.setVertexBuffer(0, mesh.vertexBuffer);
      pass.setIndexBuffer(mesh.indexBuffer, 'uint32');
      pass.drawIndexed(mesh.indexCount);
    }
    pass.end();
    encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow }, [width, height, 1]);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(readback.getMappedRange());
    const pixels = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      pixels.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
    }
    readback.unmap();
    readback.destroy();
    target.destroy();
    depth.destroy();
    const gpuMs = performance.now() - start;
    this.drainRetired();
    return { width, height, pixels, samples: strips, gpuMs };
  }

  private writeBenchFrameUniform(req: BenchRequest): void {
    const bytes = new ArrayBuffer(FRAME_UNIFORM_BYTES);
    const f = new Float32Array(bytes);
    // Identity view-projection: clip space equals world space, so each pixel maps to
    // an unambiguous world position for the CPU reference computation.
    f.set(mat4Identity(), 0);
    f[16] = 0; f[17] = 0; f[18] = 1.5; f[19] = 0;
    f[20] = req.lighting.ambient[0]; f[21] = req.lighting.ambient[1]; f[22] = req.lighting.ambient[2]; f[23] = 0;
    f[24] = req.exposure; f[25] = 0; f[26] = 0; f[27] = 0;
    let o = 28;
    for (const l of [req.lighting.key, req.lighting.fill, req.lighting.rim]) {
      f[o] = l.direction[0]; f[o + 1] = l.direction[1]; f[o + 2] = l.direction[2]; f[o + 3] = l.intensity;
      f[o + 4] = l.color[0]; f[o + 5] = l.color[1]; f[o + 6] = l.color[2]; f[o + 7] = l.angularRadius;
      o += 8;
    }
    f[o] = req.diffuseOnly ? 1 : 0;
    this.ctx.device.queue.writeBuffer(this.frameBuffer, 0, bytes);
  }

  // Deliberately kills the device so the app exercises its recovery path. The loss is
  // reported synchronously: `device.lost` resolves on the GPU process's schedule, and a
  // frame readout or overlay that waits for it would keep showing stale values.
  simulateDeviceLoss(): void {
    if (this.lost) return;
    this.lost = true;
    this.ctx.device.destroy();
    this.onLost('destroyed: 模拟设备丢失');
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.ledger.clear();
    this.slots.clear();
    this.retired = [];
    this.items.clear();
    this.meshes.clear();
    try { this.ctx.device.destroy(); } catch { /* already lost */ }
  }
}

export function cameraEye(camera: CameraState): [number, number, number] {
  const cp = Math.cos(camera.pitch);
  return [
    camera.target[0] + camera.distance * cp * Math.sin(camera.yaw),
    camera.target[1] + camera.distance * Math.sin(camera.pitch),
    camera.target[2] + camera.distance * cp * Math.cos(camera.yaw),
  ];
}

export function benchSamples(req: BenchRequest): BenchSample[] {
  if (req.kind === 'roughness') {
    return [0.05, 0.25, 0.55, 0.9].map((roughness, i) => ({
      roughness,
      metallic: req.metallic,
      baseColorHex: req.baseColorHex,
      xOffset: -0.72 + i * 0.48,
      xScale: 0.07,
      label: `R=${roughness.toFixed(2)} / M=${req.metallic.toFixed(2)}`,
    }));
  }
  if (req.kind === 'boundaries') {
    return [
      { roughness: 1.0, metallic: 0.0, baseColorHex: '#000000', xOffset: -0.6, xScale: 0.09, label: '黑色 R=1.00 M=0' },
      { roughness: 0.02, metallic: 1.0, baseColorHex: '#e8ecf0', xOffset: 0.0, xScale: 0.09, label: '近镜面 R=0.02 M=1' },
      { roughness: 0.02, metallic: 1.0, baseColorHex: '#050505', xOffset: 0.6, xScale: 0.09, label: '近镜面黑 R=0.02 M=1' },
    ];
  }
  return [{
    roughness: req.roughness,
    metallic: req.metallic,
    baseColorHex: req.baseColorHex,
    xOffset: 0,
    xScale: 1,
    label: req.kind === 'flat-normal' ? '平面法线' : '纯漫反射',
  }];
}

function benchItems(req: BenchRequest, strips: BenchSample[]): DrawItem[] {
  return strips.map((s) => {
    const model = mat4Identity();
    model[0] = s.xScale;
    // Strip height matches the plane's full height; only x is compressed.
    model[12] = s.xOffset;
    const tint = srgbToLinear(hexToSrgb(s.baseColorHex));
    const params = clampParams({
      baseColor: s.baseColorHex,
      roughness: s.roughness,
      metallic: s.metallic,
      normalStrength: req.normalStrength,
      normalEnabled: req.normalEnabled,
      uvScale: 1,
    });
    return {
      mesh: 'bench',
      model,
      params,
      tint,
      swatchId: req.swatchId,
      textureSet: req.textureSet,
      solid: req.solid,
    };
  });
}

// Builds the studio scene drawers. Only the selected swatch's target mesh uses the
// live material; the other meshes keep fixed, documented materials so switching a
// swatch can never leak its textures or parameters into unrelated geometry.
export const FIXED_DASH: { swatchId: SwatchId; params: MaterialParams } = {
  swatchId: 'leather-black',
  params: { baseColor: '#222326', roughness: 0.6, metallic: 0, normalStrength: 1, normalEnabled: true, uvScale: 3.2 },
};
export const FIXED_TRIM: { swatchId: SwatchId; params: MaterialParams } = {
  swatchId: 'aluminum-brushed',
  params: { baseColor: '#c0c4c9', roughness: 0.3, metallic: 1, normalStrength: 0.5, normalEnabled: true, uvScale: 8 },
};
export const FIXED_PAD: { swatchId: SwatchId; params: MaterialParams } = {
  swatchId: 'piano-gloss',
  params: { baseColor: '#313235', roughness: 0.9, metallic: 0, normalStrength: 1.2, normalEnabled: true, uvScale: 4 },
};

export function studioItems(opts: {
  swatchId: SwatchId;
  swatchTarget: 'dashboard' | 'trim' | 'softpad';
  material: MaterialParams;
  textureSets: Record<string, TextureSet>;
  nonUniformScale: boolean;
  showSoftpad: boolean;
  specimen?: { yaw: number; pitch: number };
}): DrawItem[] {
  const make = (mesh: MeshKey, params: MaterialParams, swatchId: SwatchId, model: Mat4): DrawItem => ({
    mesh,
    model,
    params,
    tint: srgbToLinear(hexToSrgb(params.baseColor)),
    swatchId,
    textureSet: opts.textureSets[swatchId],
  });

  const dashParams = opts.swatchTarget === 'dashboard' ? opts.material : FIXED_DASH.params;
  const dashId = opts.swatchTarget === 'dashboard' ? opts.swatchId : FIXED_DASH.swatchId;
  const trimParams = opts.swatchTarget === 'trim' ? opts.material : FIXED_TRIM.params;
  const trimId = opts.swatchTarget === 'trim' ? opts.swatchId : FIXED_TRIM.swatchId;
  const padParams = opts.swatchTarget === 'softpad' ? opts.material : FIXED_PAD.params;
  const padId = opts.swatchTarget === 'softpad' ? opts.swatchId : FIXED_PAD.swatchId;

  // Non-uniform scale is applied around the cushion's own centre so the sample stays
  // seated on the dash instead of drifting when the scale changes.
  const padScale = opts.nonUniformScale ? [1.35, 0.7, 1.0] : [1, 1, 1];
  const trimModel = mat4Identity();
  trimModel[14] = 0.004; // lift the bezel off the dash surface to avoid z-fighting

  const list: DrawItem[] = [
    make('dashboard', dashParams, dashId, mat4Identity()),
    make('trim', trimParams, trimId, trimModel),
  ];
  if (opts.showSoftpad) list.push(make('softpad', padParams, padId, centeredScaleModel([0.54, 0.10, 0.295], padScale)));

  // The selected specimen rotates about its own centre; every other mesh keeps its
  // authored transform so rotating the sample cannot shift the surrounding scene.
  if (opts.specimen && (opts.specimen.yaw !== 0 || opts.specimen.pitch !== 0)) {
    const targetIndex = opts.swatchTarget === 'dashboard' ? 0 : opts.swatchTarget === 'trim' ? 1 : opts.showSoftpad ? 2 : -1;
    if (targetIndex >= 0) {
      const pivot = opts.swatchTarget === 'softpad' ? [0.54, 0.10, 0.295] : opts.swatchTarget === 'trim' ? [0, 0.24, 0.28] : [0, 0.17, 0.31];
      const rotation = mat4FromTRS([0, 0, 0], opts.specimen.yaw, opts.specimen.pitch, [1, 1, 1]);
      list[targetIndex] = { ...list[targetIndex], model: rotationAboutPivot(rotation, pivot) };
    }
  }
  return list;
}

// T(p) * R * T(-p): rotate a specimen about its own centre while it stays seated in
// the studio. The model matrix keeps its non-uniform scale on the cushion because the
// rotation is applied outside the existing scaled transform.
function rotationAboutPivot(rotation: Mat4, pivot: number[]): Mat4 {
  const toOrigin = mat4Identity();
  toOrigin[12] = -pivot[0]; toOrigin[13] = -pivot[1]; toOrigin[14] = -pivot[2];
  const back = mat4Identity();
  back[12] = pivot[0]; back[13] = pivot[1]; back[14] = pivot[2];
  return mat4Multiply(back, mat4Multiply(rotation, toOrigin));
}

// Scale about a pivot: T(p) * S * T(-p). Keeps the object anchored while making the
// model matrix genuinely non-uniform, which is what exercises the normal matrix.
function centeredScaleModel(pivot: number[], scale: number[]): Mat4 {
  const m = mat4Identity();
  m[0] = scale[0]; m[5] = scale[1]; m[10] = scale[2];
  m[12] = pivot[0] * (1 - scale[0]);
  m[13] = pivot[1] * (1 - scale[1]);
  m[14] = pivot[2] * (1 - scale[2]);
  return m;
}

export function resourceSummary(r: Renderer): string {
  const counts = Object.entries(r.resourceCounts).map(([k, v]) => `${k}:${v}`).join(' ');
  return `${formatBytes(r.resourceBytes)}${counts ? ` · ${counts}` : ''}`;
}

// Local re-export so the app can reference mesh keys without importing geometry.
export type { MeshData };
