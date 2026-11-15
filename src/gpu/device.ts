// GPU device lifecycle: creation, capability probing, byte accounting and explicit
// teardown. Every buffer/texture in the app goes through ResourceLedger so the
// "资源占用" readout reflects real allocations rather than an estimate.

export interface GpuDiagnostics {
  adapterLabel: string;
  backend: string;
  isFallback: boolean;
  timestampSupported: boolean;
  maxTextureDimension2D: number;
  maxBufferSize: number;
  /** Whether the probe ran against a software rasteriser rather than real hardware. */
  softwareRasterizer: boolean;
  vendor: string;
  architecture: string;
}

export interface ShaderCompileInfo {
  label: string;
  ok: boolean;
  messages: { type: string; message: string; lineNum: number; linePos: number }[];
}

export interface ShaderCompileState {
  ready: boolean;
  modules: ShaderCompileInfo[];
  pipelines: { label: string; ok: boolean; error?: string }[];
  ok: boolean;
  /** True when the browser refused to report compilation info at all. */
  unsupported: boolean;
}

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** Linear HDR render target format. */
  format: GPUTextureFormat;
  canvasFormat: GPUTextureFormat;
  diagnostics: GpuDiagnostics;
}

export interface ResourceHandle {
  readonly kind: string;
  readonly bytes: number;
  release(): void;
}

export class ResourceLedger {
  private bytes = 0;
  private readonly live = new Set<ResourceHandle & { destroy: () => void }>();

  track(kind: string, bytes: number, destroy: () => void): ResourceHandle {
    let released = false;
    const rec = {
      kind,
      bytes,
      destroy,
      release: () => {
        if (released) return;
        released = true;
        this.live.delete(rec);
        this.bytes -= bytes;
        destroy();
      },
    };
    this.live.add(rec);
    this.bytes += bytes;
    return rec;
  }

  total(): number { return this.bytes; }

  counts(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const r of this.live) out[r.kind] = (out[r.kind] ?? 0) + 1;
    return out;
  }

  clear(): void {
    for (const r of [...this.live]) r.release();
    this.bytes = 0;
  }
}

export function textureBytes(width: number, height: number, format: GPUTextureFormat, mipLevels = 1): number {
  const bpp = formatBytesPerPixel(format);
  let total = 0;
  let w = width, h = height;
  for (let i = 0; i < mipLevels; i++) {
    total += w * h * bpp;
    w = Math.max(1, Math.floor(w / 2));
    h = Math.max(1, Math.floor(h / 2));
  }
  return total;
}

export function formatBytesPerPixel(format: GPUTextureFormat): number {
  switch (format) {
    case 'r8unorm':
      return 1;
    case 'rg8unorm':
      return 2;
    case 'rgba8unorm':
    case 'rgba8unorm-srgb':
    case 'bgra8unorm':
    case 'bgra8unorm-srgb':
    case 'depth24plus':
    case 'depth32float':
      return 4;
    case 'rgba16float':
      return 8;
    case 'rgba32float':
      return 16;
    default:
      return 8;
  }
}

export async function createGpuContext(): Promise<GpuContext> {
  if (!navigator.gpu) {
    throw new Error('此浏览器未暴露 navigator.gpu。请使用支持 WebGPU 的 Chromium（需 --enable-unsafe-webgpu）。');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('未能获取 WebGPU 适配器。');
  }

  const requiredFeatures: GPUFeatureName[] = [];
  const timestampSupported = adapter.features.has('timestamp-query');
  if (timestampSupported) requiredFeatures.push('timestamp-query');

  const requiredLimits: Record<string, number> = {};
  const maxTex = Math.min(2048, adapter.limits.maxTextureDimension2D);
  requiredLimits.maxTextureDimension2D = maxTex;

  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });

  const info = adapter.info as GPUAdapterInfo | undefined;
  const vendor = info?.vendor ?? '';
  const architecture = info?.architecture ?? '';
  const diagnostics: GpuDiagnostics = {
    adapterLabel: describeAdapter(info),
    backend: (info as { backend?: string } | undefined)?.backend ?? 'webgpu',
    isFallback: (adapter as { isFallbackAdapter?: boolean }).isFallbackAdapter ?? false,
    timestampSupported,
    maxTextureDimension2D: device.limits.maxTextureDimension2D,
    maxBufferSize: device.limits.maxBufferSize,
    softwareRasterizer: isSoftwareAdapter(info),
    vendor,
    architecture,
  };

  return { adapter, device, format: 'rgba16float', canvasFormat: navigator.gpu.getPreferredCanvasFormat(), diagnostics };
}

// A software rasteriser reports itself through the adapter architecture
// (SwiftShader / llvmpipe / lavapipe). The release gate uses this to distinguish a
// real hardware frame from a fallback image, which must never be presented as verified.
export function isSoftwareAdapter(info: GPUAdapterInfo | undefined): boolean {
  const haystack = `${info?.description ?? ''} ${info?.vendor ?? ''} ${info?.architecture ?? ''}`.toLowerCase();
  return /swiftshader|llvmpipe|softpipe|lavapipe|software/.test(haystack);
}

function describeAdapter(info: GPUAdapterInfo | undefined): string {
  if (!info) return 'WebGPU adapter';
  const parts = [info.description, info.vendor, info.architecture].filter((x): x is string => !!x && x.length > 0);
  if (parts.length > 0) return parts.join(' · ');
  const backend = (info as { backend?: string }).backend;
  return backend ? `WebGPU (${backend})` : 'WebGPU adapter';
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}
