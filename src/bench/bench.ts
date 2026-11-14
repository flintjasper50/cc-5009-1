// Benchmark + reference validation. Four scenes:
//  1. diffuse     — pure Lambert, flat normal, solid albedo, one key light
//  2. flat-normal — normal map disabled (geometry normal only)
//  3. roughness   — four strips at fixed roughness, metallic 1
//  4. boundaries  — black+rough, near-mirror metal, black near-mirror
// The GPU renders the exact shader path into rgba8unorm; the CPU reference replays
// the same stated model per pixel. We report per-channel absolute error, and the
// maximum is asserted against a tolerance that reflects 8-bit quantisation.

import { evaluateShading, encodeBenchPixel, type CpuShading } from '../math/brdf.ts';
import { hexToSrgb, srgbToLinear, type RGB } from '../math/color.ts';
import { benchSamples, cameraEye, Renderer, type BenchRequest, type BenchResult } from '../gpu/renderer.ts';
import { buildSwatchTextures, type TextureSet } from '../scene/textures.ts';
import type { LightingState, SwatchId } from '../scene/material.ts';

export interface BenchSceneSpec {
  id: 'diffuse' | 'flat-normal' | 'roughness' | 'boundaries';
  title: string;
  description: string;
  expectation: string;
  request: Omit<BenchRequest, 'textureSet' | 'swatchId'>;
}

const BENCH_LIGHTING: LightingState = {
  key: { direction: [-0.42, 0.62, 0.66], color: [1.0, 0.97, 0.92], intensity: 3.4, angularRadius: 0.12 },
  fill: { direction: [0.72, 0.18, 0.42], color: [0.78, 0.85, 1.0], intensity: 0.9, angularRadius: 0.18 },
  rim: { direction: [0.10, 0.30, -0.95], color: [0.92, 0.95, 1.0], intensity: 1.5, angularRadius: 0.15 },
  ambient: [0.020, 0.022, 0.026],
};

const BOUNDARY_LIGHTING: LightingState = {
  key: { direction: [0.0, 0.18, 0.98], color: [1.0, 0.98, 0.95], intensity: 5.0, angularRadius: 0.16 },
  fill: { direction: [-0.30, -0.10, 0.95], color: [0.80, 0.86, 1.0], intensity: 1.2, angularRadius: 0.2 },
  rim: { direction: [0.35, 0.30, 0.88], color: [0.95, 0.95, 1.0], intensity: 1.4, angularRadius: 0.2 },
  ambient: [0.012, 0.013, 0.016],
};

export const BENCH_SCENES: BenchSceneSpec[] = [
  {
    id: 'diffuse',
    title: '纯漫反射基准',
    description: '单一主光 + 平面法线 + 纯白反照率，关闭高光与法线贴图。',
    expectation: '每个像素应为 sum_i(kd·albedo/π · I_i · max(0, N·L_i)) + ambient·albedo，且与 GPU 输出在 8 位量化内一致。',
    request: {
      kind: 'diffuse',
      width: 512,
      height: 320,
      lighting: BENCH_LIGHTING,
      exposure: 0,
      baseColorHex: '#b9bcc2',
      normalStrength: 0,
      roughness: 1,
      metallic: 0,
      diffuseOnly: true,
      normalEnabled: false,
      solid: true,
    },
  },
  {
    id: 'flat-normal',
    title: '平面法线基准',
    description: '保留 GGX 高光与粗糙度，但法线贴图关闭，用于确认着色只依赖几何法线。',
    expectation: '与“法线贴图开启”的场景相比，平面上的 N·H 应均匀，颜色只随视角与光照方向变化。',
    request: {
      kind: 'flat-normal',
      width: 512,
      height: 320,
      lighting: BENCH_LIGHTING,
      exposure: 0,
      baseColorHex: '#b9bcc2',
      normalStrength: 0,
      roughness: 0.35,
      metallic: 0,
      diffuseOnly: false,
      normalEnabled: false,
      solid: true,
    },
  },
  {
    id: 'roughness',
    title: '粗糙度序列',
    description: '四条同色金属样条，粗糙度 0.05 / 0.25 / 0.55 / 0.90。',
    expectation: '粗糙度越低，高光越集中、峰值越亮；R=0.05 时反射接近镜面，R=0.9 时高光铺开。',
    request: {
      kind: 'roughness',
      width: 768,
      height: 256,
      lighting: BENCH_LIGHTING,
      exposure: 0,
      baseColorHex: '#d5d9de',
      normalStrength: 0,
      roughness: 0.5,
      metallic: 1,
      diffuseOnly: false,
      normalEnabled: false,
      solid: true,
    },
  },
  {
    id: 'boundaries',
    title: '边界条件',
    description: '黑色极粗糙 (R=1,M=0)、近镜面金属 (R=0.02,M=1)、黑色近镜面 (R=0.02,M=1)。',
    expectation: '黑色极粗糙只留漫反射近零与微弱 F0=0.04 高光；近镜面金属高光峰值高且集中在主光方向；黑色近镜面暗部但边缘 Fresnel 明显。',
    request: {
      kind: 'boundaries',
      width: 768,
      height: 256,
      lighting: BOUNDARY_LIGHTING,
      exposure: 0,
      baseColorHex: '#ffffff',
      normalStrength: 0,
      roughness: 0.5,
      metallic: 0,
      diffuseOnly: false,
      normalEnabled: false,
      solid: true,
    },
  },
];

export interface PixelError {
  label: string;
  sampleRects: { x0: number; y0: number; x1: number; y1: number }[];
  gpuRgb: [number, number, number];
  cpuRgb: [number, number, number];
  maxAbsError: number;
}

export interface BenchOutcome {
  scene: BenchSceneSpec;
  result: BenchResult;
  sampleErrors: PixelError[];
  maxAbsError: number;
  gpuMs: number;
  pixelsCompared: number;
  cpuMs: number;
  referenceImage: ImageData | null;
}

// CPU reference for a bench pixel. The bench plane sits at z=0 with an identity
// view-projection, so NDC x/y in [-1,1] map linearly to world x/y and the camera sits
// at (0,0,1.5). U/V are fixed so flat-normal and diffuse compare against plain maths.
export function cpuReferencePixel(
  x: number,
  y: number,
  width: number,
  height: number,
  req: BenchRequest,
  sample: ReturnType<typeof benchSamples>[number],
): RGB {
  // Identity view-projection on the bench path means NDC == world position on z=0,
  // so the pixel's world x/y are recovered directly from its NDC coordinates.
  const worldX = (x + 0.5) / width * 2 - 1;
  const worldY = (y + 0.5) / height * 2 - 1;
  const cam: [number, number, number] = [0, 0, 1.5];
  const view: [number, number, number] = normalize3([cam[0] - worldX, cam[1] - worldY, cam[2] - 0]);
  const n: [number, number, number] = [0, 0, 1];
  const t: [number, number, number] = [1, 0, 0];
  const b: [number, number, number] = [0, 1, 0];
  const albedo = req.solid ? srgbToLinear(hexToSrgb(sample.baseColorHex)) : srgbToLinear(hexToSrgb(sample.baseColorHex));
  const shading: CpuShading = {
    n, t, b, view, albedo,
    roughness: sample.roughness,
    metallic: sample.metallic,
    ambient: req.lighting.ambient,
    lights: [
      { direction: req.lighting.key.direction, color: req.lighting.key.color, intensity: req.lighting.key.intensity, angularRadius: req.lighting.key.angularRadius },
      { direction: req.lighting.fill.direction, color: req.lighting.fill.color, intensity: req.lighting.fill.intensity, angularRadius: req.lighting.fill.angularRadius },
      { direction: req.lighting.rim.direction, color: req.lighting.rim.color, intensity: req.lighting.rim.intensity, angularRadius: req.lighting.rim.angularRadius },
    ],
    diffuseOnly: req.diffuseOnly,
  };
  return evaluateShading(shading);
}

function normalize3(v: [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

export async function runBenchScene(
  renderer: Renderer,
  spec: BenchSceneSpec,
  textureSets: Record<string, TextureSet>,
  swatchId: SwatchId,
): Promise<BenchOutcome> {
  const req: BenchRequest = { ...spec.request, textureSet: textureSets[swatchId], swatchId };
  const result = await renderer.renderBench(req);
  const samples = benchSamples(req);
  const errors: PixelError[] = [];
  let maxAbs = 0;
  let compared = 0;

  // Sample the centre of each strip (screen-space NDC), not the full plane.
  samples.forEach((s, si) => {
    const cx = Math.floor(((s.xOffset * 0.5) + 0.5) * result.width);
    const cy = Math.floor(result.height * 0.5);
    const gpu = readPixel(result, cx, cy);
    const linear = cpuReferencePixel(cx, cy, result.width, result.height, req, s);
    const cpu = encodeBenchPixel(linear, req.exposure);
    const err = Math.max(Math.abs(gpu[0] - cpu[0]), Math.abs(gpu[1] - cpu[1]), Math.abs(gpu[2] - cpu[2]));
    maxAbs = Math.max(maxAbs, err);
    compared++;
    errors.push({
      label: s.label,
      sampleRects: [{ x0: cx, y0: cy, x1: cx, y1: cy }],
      gpuRgb: gpu,
      cpuRgb: cpu,
      maxAbsError: err,
    });
    void si;
  });

  const cpuStart = performance.now();
  const referenceImage = buildReferenceImage(result.width, result.height, req, samples);
  const cpuMs = performance.now() - cpuStart;

  return {
    scene: spec,
    result,
    sampleErrors: errors,
    maxAbsError: maxAbs,
    gpuMs: result.gpuMs,
    pixelsCompared: compared,
    cpuMs,
    referenceImage,
  };
}

function readPixel(result: BenchResult, x: number, y: number): [number, number, number] {
  const i = (y * result.width + x) * 4;
  return [result.pixels[i], result.pixels[i + 1], result.pixels[i + 2]];
}

// Reference image from the CPU model, for side-by-side comparison in the UI.
export function buildReferenceImage(width: number, height: number, req: BenchRequest, samples: ReturnType<typeof benchSamples>): ImageData {
  const img = new ImageData(width, height);
  // Strip half-width in NDC is 2*xScale (bench plane spans -2..2 world units).
  const sampleXs = samples.map((s) => ({ s, cx: (s.xOffset * 0.5 + 0.5) * width, halfNdc: 2 * s.xScale }));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ndc = (x + 0.5) / width * 2 - 1;
      const hit = sampleXs.find((s) => Math.abs(ndc - s.s.xOffset) <= s.halfNdc);
      const rgb: [number, number, number] = hit
        ? encodeBenchPixel(cpuReferencePixel(x, y, width, height, req, hit.s), req.exposure)
        : [10, 10, 10];
      const o = (y * width + x) * 4;
      img.data[o] = rgb[0];
      img.data[o + 1] = rgb[1];
      img.data[o + 2] = rgb[2];
      img.data[o + 3] = 255;
    }
  }
  return img;
}

export function outcomeToImageData(result: BenchResult): ImageData {
  const img = new ImageData(result.width, result.height);
  img.data.set(result.pixels);
  return img;
}

export function drawImageDataToCanvas(canvas: HTMLCanvasElement, img: ImageData): void {
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.putImageData(img, 0, 0);
}

export function imageDataPixels(img: ImageData): Uint8Array {
  return new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength);
}

// Convenience for the automated Playwright suite: runs all four scenes and returns
// the numeric summary it asserts on.
export async function runAllBenchScenes(renderer: Renderer, swatchId: SwatchId): Promise<BenchOutcome[]> {
  const textureSets = buildSwatchTextures();
  const out: BenchOutcome[] = [];
  for (const spec of BENCH_SCENES) {
    out.push(await runBenchScene(renderer, spec, textureSets, swatchId));
  }
  return out;
}

export { cameraEye };
