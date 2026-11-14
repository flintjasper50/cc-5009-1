// Color-texture encoding is kept strictly separate from the linear values used in
// shading. sRGB textures are decoded by the GPU on sample (rgb8unorm-srgb view); this
// module provides the identical transfer function on the CPU so that UI swatches,
// reference renders and history readbacks agree with the shader.

export type RGB = [number, number, number];

export function srgbToLinearChannel(c: number): number {
  // IEC 61966-2-1. Exact piecewise curve, not the 2.2 power approximation.
  const v = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  return v;
}

export function linearToSrgbChannel(c: number): number {
  const clamped = Math.max(0, Math.min(1, c));
  const v = clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055;
  return v;
}

export function srgbToLinear(color: RGB): RGB {
  return [srgbToLinearChannel(color[0]), srgbToLinearChannel(color[1]), srgbToLinearChannel(color[2])];
}

export function linearToSrgb(color: RGB): RGB {
  return [linearToSrgbChannel(color[0]), linearToSrgbChannel(color[1]), linearToSrgbChannel(color[2])];
}

export function hexToSrgb(hex: string): RGB {
  const s = hex.replace('#', '');
  const v = s.length === 3
    ? s.split('').map((c) => parseInt(c + c, 16))
    : [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
  return [v[0] / 255, v[1] / 255, v[2] / 255];
}

export function srgbToHex(color: RGB): string {
  const to = (c: number) => Math.round(Math.max(0, Math.min(1, c)) * 255).toString(16).padStart(2, '0');
  return '#' + to(color[0]) + to(color[1]) + to(color[2]);
}

// Normal-map bytes are direction data, not color. Decoding is (b/255)*2-1 and must
// never go through srgbToLinear. Encoding mirrors it exactly for generated maps.
export function normalMapEncode(n: RGB): RGB {
  return [(n[0] * 0.5 + 0.5), (n[1] * 0.5 + 0.5), (n[2] * 0.5 + 0.5)];
}

export function normalMapDecode(encoded: RGB): RGB {
  const n: RGB = [encoded[0] * 2 - 1, encoded[1] * 2 - 1, encoded[2] * 2 - 1];
  const len = Math.hypot(n[0], n[1], n[2]) || 1;
  return [n[0] / len, n[1] / len, n[2] / len];
}

// Display transform used by the composite pass. Exposure is applied in linear light,
// then the sRGB OETF (the canvas is configured with an srgb view format).
export function exposureScale(stops: number): number {
  return Math.pow(2, stops);
}

export function linearToDisplaySrgb(linear: RGB, exposureStops: number): RGB {
  const k = exposureScale(exposureStops);
  return linearToSrgb([linear[0] * k, linear[1] * k, linear[2] * k]);
}

export function luminance(linear: RGB): number {
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}
