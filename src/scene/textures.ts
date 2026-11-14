// Procedural texture generation. Colour maps are authored in sRGB-encoded bytes and
// uploaded through an srgb view so the GPU decodes to linear on sample. Normal maps
// carry directions encoded as n*0.5+0.5 in plain unorm bytes and MUST NOT be decoded
// as colour; the shader does the 2n-1 decode explicitly.

export interface MipChain {
  readonly width: number;
  readonly height: number;
  /** One Uint8Array per mip level, RGBA8. */
  readonly levels: readonly Uint8Array[];
}

export interface TextureSet {
  readonly albedo: MipChain;
  readonly normal: MipChain;
  readonly label: string;
  readonly bytes: number;
}

export interface SwatchTextureSpec {
  id: string;
  label: string;
  size: number;
  /** Height field in [0,1], sampled on a periodic domain. */
  height: (u: number, v: number) => number;
  /** Linear-space surface colour in [0,1]. */
  color: (u: number, v: number) => [number, number, number];
  /** Optional weave/thread mask controlling where height derivatives apply. */
  normalStrengthScale?: number;
}

function hash2(x: number, y: number, seed: number): number {
  // Deterministic integer hash; avoids Math.random so presets reproduce exactly.
  let h = (x * 374761393 + y * 668265263 + seed * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967295;
}

function valueNoise(x: number, y: number, seed: number): number {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi, seed);
  const b = hash2(xi + 1, yi, seed);
  const c = hash2(xi, yi + 1, seed);
  const d = hash2(xi + 1, yi + 1, seed);
  return a * (1 - u) * (1 - v) + b * u * (1 - v) + c * (1 - u) * v + d * u * v;
}

// Tileable fBm: lattice wraps on the period so the texture repeats without seams.
export function tileableFbm(u: number, v: number, period: number, octaves: number, seed: number): number {
  let amp = 0.5, freq = period, sum = 0, norm = 0;
  for (let o = 0; o < octaves; o++) {
    const x = u * freq, y = v * freq;
    // wrap lattice coordinates to the period to make it seamless
    const xi = ((Math.floor(x) % freq) + freq) % freq;
    const yi = ((Math.floor(y) % freq) + freq) % freq;
    const xf = x - Math.floor(x), yf = y - Math.floor(y);
    const sx = xf * xf * (3 - 2 * xf);
    const sy = yf * yf * (3 - 2 * yf);
    const a = hash2(xi, yi, seed + o);
    const b = hash2((xi + 1) % freq, yi, seed + o);
    const c = hash2(xi, (yi + 1) % freq, seed + o);
    const d = hash2((xi + 1) % freq, (yi + 1) % freq, seed + o);
    const n = a * (1 - sx) * (1 - sy) + b * sx * (1 - sy) + c * (1 - sx) * sy + d * sx * sy;
    sum += n * amp;
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / norm;
}

function linearToSrgb(c: number): number {
  const v = Math.max(0, Math.min(1, c));
  return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}

function wrapNoise(x: number, y: number, period: number, seed: number): number {
  const xi = ((Math.floor(x) % period) + period) % period;
  const yi = ((Math.floor(y) % period) + period) % period;
  const xf = x - Math.floor(x), yf = y - Math.floor(y);
  const sx = xf * xf * (3 - 2 * xf), sy = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi, seed);
  const b = hash2((xi + 1) % period, yi, seed);
  const c = hash2(xi, (yi + 1) % period, seed);
  const d = hash2((xi + 1) % period, (yi + 1) % period, seed);
  return a * (1 - sx) * (1 - sy) + b * sx * (1 - sy) + c * (1 - sx) * sy + d * sx * sy;
}

// Downsamples an RGBA8 image by 2x with a box filter. Normal maps are re-normalised
// after averaging because averaging directions and then normalising is the standard
// mip construction for tangent-space maps.
function downsample(src: Uint8Array, w: number, h: number, isNormal: boolean): Uint8Array {
  const dw = Math.max(1, w >> 1);
  const dh = Math.max(1, h >> 1);
  const out = new Uint8Array(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const sx = Math.min(w - 1, x * 2 + dx);
          const sy = Math.min(h - 1, y * 2 + dy);
          const i = (sy * w + sx) * 4;
          r += src[i]; g += src[i + 1]; b += src[i + 2]; a += src[i + 3];
        }
      }
      const o = (y * dw + x) * 4;
      if (isNormal) {
        // Average decoded directions, renormalise, re-encode. Never blend encodings as
        // if they were colours.
        const nx = (r / 4) / 255 * 2 - 1;
        const ny = (g / 4) / 255 * 2 - 1;
        const nz = (b / 4) / 255 * 2 - 1;
        const len = Math.hypot(nx, ny, nz) || 1;
        out[o] = Math.round(((nx / len) * 0.5 + 0.5) * 255);
        out[o + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255);
        out[o + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255);
      } else {
        out[o] = Math.round(r / 4);
        out[o + 1] = Math.round(g / 4);
        out[o + 2] = Math.round(b / 4);
      }
      out[o + 3] = 255;
    }
  }
  return out;
}

function buildMips(base: Uint8Array, size: number, isNormal: boolean): MipChain {
  const levels: Uint8Array[] = [base];
  let w = size, h = size;
  let src = base;
  while (w > 1 || h > 1) {
    const next = downsample(src, w, h, isNormal);
    levels.push(next);
    src = next;
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
  }
  return { width: size, height: size, levels };
}

export function buildTextureSet(spec: SwatchTextureSpec): TextureSet {
  const { size } = spec;
  const albedo = new Uint8Array(size * size * 4);
  const normal = new Uint8Array(size * size * 4);
  const heights = new Float32Array(size * size);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      heights[y * size + x] = spec.height(x / size, y / size);
    }
  }

  // Derivative scale is in texels; dividing by size keeps slopes resolution-independent
  // so a 512 map and a 256 map of the same pattern shade alike.
  const strength = spec.normalStrengthScale ?? 1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const v = y / size;
      const idx = y * size + x;
      const c = spec.color(u, v);
      albedo[idx * 4] = Math.round(linearToSrgb(c[0]) * 255);
      albedo[idx * 4 + 1] = Math.round(linearToSrgb(c[1]) * 255);
      albedo[idx * 4 + 2] = Math.round(linearToSrgb(c[2]) * 255);
      albedo[idx * 4 + 3] = 255;

      const xm = (x - 1 + size) % size, xp = (x + 1) % size;
      const ym = (y - 1 + size) % size, yp = (y + 1) % size;
      // d(height)/d(uv); multiply by 2 for the central-difference span.
      const dhdu = (heights[y * size + xp] - heights[y * size + xm]) * 0.5;
      const dhdv = (heights[yp * size + x] - heights[ym * size + x]) * 0.5;
      let nx = -dhdu * strength;
      let ny = -dhdv * strength;
      let nz = 1;
      const len = Math.hypot(nx, ny, nz);
      nx /= len; ny /= len; nz /= len;
      normal[idx * 4] = Math.round((nx * 0.5 + 0.5) * 255);
      normal[idx * 4 + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      normal[idx * 4 + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      normal[idx * 4 + 3] = 255;
    }
  }

  const albedoChain = buildMips(albedo, size, false);
  const normalChain = buildMips(normal, size, true);
  const bytes = albedoChain.levels.reduce((n, l) => n + l.byteLength, 0) +
    normalChain.levels.reduce((n, l) => n + l.byteLength, 0);
  return { albedo: albedoChain, normal: normalChain, label: spec.label, bytes };
}

// --- Swatch definitions -----------------------------------------------------

function leatherColor(base: [number, number, number], grain: number): (u: number, v: number) => [number, number, number] {
  return (u, v) => {
    const n1 = tileableFbm(u, v, 16, 3, 11);
    const n2 = tileableFbm(u, v, 48, 2, 23);
    const variation = 0.82 + 0.30 * n1 + 0.18 * (n2 - 0.5);
    const tint = 0.94 + 0.12 * (n2 - 0.5) * grain;
    return [base[0] * variation * tint, base[1] * variation, base[2] * variation * (2 - tint)];
  };
}

export const SWATCH_SPECS: SwatchTextureSpec[] = [
  {
    id: 'leather-black',
    label: '黑色打孔皮革',
    size: 512,
    normalStrengthScale: 0.35,
    height: (u, v) => {
      const grain = tileableFbm(u * 6, v * 6, 64, 3, 31);
      const pores = tileableFbm(u * 12, v * 12, 96, 2, 41);
      const punch = perforation(u, v);
      return grain * 0.5 + (pores > 0.78 ? 0.15 : 0.0) - punch * 0.7;
    },
    color: (u, v) => {
      const c = leatherColor([0.010, 0.010, 0.011], 1)(u, v);
      const punch = perforation(u, v);
      const dark = 1 - 0.55 * punch;
      return [c[0] * dark, c[1] * dark, c[2] * dark];
    },
  },
  {
    id: 'leather-cognac',
    label: '干邑色软皮',
    size: 512,
    normalStrengthScale: 0.30,
    height: (u, v) => {
      const grain = tileableFbm(u * 5, v * 5, 64, 3, 17);
      const creases = Math.abs(tileableFbm(u * 2, v * 2, 32, 3, 19) - 0.5);
      return grain * 0.55 + Math.pow(1 - creases * 2, 3) * 0.18;
    },
    color: leatherColor([0.265, 0.115, 0.045], 1),
  },
  {
    id: 'alcantara-graphite',
    label: '石墨灰翻毛皮',
    size: 512,
    normalStrengthScale: 0.55,
    height: (u, v) => {
      // Fine directional fibre: anisotropic streaks, kept below the texel Nyquist limit.
      const f = tileableFbm(u * 48, v * 8, 64, 3, 5);
      const g = tileableFbm(u * 8, v * 8, 32, 3, 7);
      return f * 0.3 + g * 0.35;
    },
    color: (u, v) => {
      const f = tileableFbm(u * 48, v * 8, 64, 2, 5);
      const g = tileableFbm(u * 16, v * 16, 32, 2, 9);
      const sheen = 0.72 + 0.35 * f;
      const base: [number, number, number] = [0.075, 0.078, 0.085];
      const tint = 0.9 + 0.2 * g;
      return [base[0] * sheen * tint, base[1] * sheen * tint, base[2] * sheen * tint];
    },
  },
  {
    id: 'aluminum-brushed',
    label: '拉丝铝饰条',
    size: 512,
    normalStrengthScale: 0.18,
    height: (u, v) => {
      const brush = tileableFbm(u * 96, v * 4, 96, 3, 29);
      return brush * 0.45;
    },
    color: (u, v) => {
      const brush = tileableFbm(u * 96, v * 4, 96, 2, 29);
      const base: [number, number, number] = [0.55, 0.57, 0.60];
      const k = 0.88 + 0.24 * brush;
      return [base[0] * k, base[1] * k, base[2] * k];
    },
  },
  {
    id: 'carbon-twill',
    label: '碳纤维斜纹',
    size: 512,
    normalStrengthScale: 0.45,
    height: (u, v) => {
      const weave = twillWeave(u, v);
      const fibre = tileableFbm(u * 24, v * 24, 48, 2, 61);
      return weave * 0.5 + fibre * 0.12;
    },
    color: (u, v) => {
      const weave = twillWeave(u, v);
      const fibre = tileableFbm(u * 24, v * 24, 48, 2, 61);
      const base: [number, number, number] = [0.035, 0.037, 0.041];
      const k = 0.6 + weave * 0.9 + fibre * 0.2;
      return [base[0] * k, base[1] * k, base[2] * k];
    },
  },
  {
    id: 'piano-gloss',
    label: '钢琴黑高光',
    size: 512,
    normalStrengthScale: 0.05,
    height: (u, v) => tileableFbm(u, v, 8, 2, 71) * 0.05,
    color: (u, v) => {
      const n = tileableFbm(u, v, 24, 2, 71);
      const v2 = 0.004 + 0.003 * n;
      return [v2, v2, v2 * 1.1];
    },
  },
  {
    id: 'flatscreen-void',
    label: '纯漫反射基准 (无纹理)',
    size: 4,
    height: () => 0,
    color: () => [1, 1, 1],
  },
];

function perforation(u: number, v: number): number {
  const nx = 48, ny = 48;
  const fx = (u * nx) % 1, fy = (v * ny) % 1;
  const cx = fx - 0.5, cy = fy - 0.5;
  const stagger = Math.floor(u * nx + v * ny) % 2;
  const d = Math.hypot((cx + stagger * 0.5) % 1 - 0.5, cy);
  return d < 0.16 ? 1 - d / 0.16 : 0;
}

function twillWeave(u: number, v: number): number {
  const nx = 16, ny = 16;
  const ix = Math.floor(u * nx);
  const iy = Math.floor(v * ny);
  const fx = (u * nx) % 1;
  const fy = (v * ny) % 1;
  // 2/2 twill: warp visible when ((ix+iy) mod 4) < 2
  const over = ((ix + iy) % 4) < 2;
  const profile = over ? Math.sin(fx * Math.PI) : Math.sin(fy * Math.PI);
  return over ? 0.35 + 0.65 * profile : 0.15 + 0.5 * profile;
}

export function buildSwatchTextures(): Record<string, TextureSet> {
  const out: Record<string, TextureSet> = {};
  for (const spec of SWATCH_SPECS) {
    out[spec.id] = buildTextureSet(spec);
  }
  return out;
}
