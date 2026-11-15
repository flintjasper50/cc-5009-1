// PNG comparison helpers for the release gate. Uses pngjs from the project's own
// node_modules so the gate does not depend on ad-hoc /tmp installs.

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { PNG } = require('pngjs');

export function readPng(buffer) {
  return PNG.sync.read(buffer);
}

export function summarizePng(buffer) {
  const png = PNG.sync.read(buffer);
  let sum = 0;
  let max = 0;
  let opaque = 0;
  const pixels = png.width * png.height;
  for (let i = 0; i < png.data.length; i += 4) {
    const luma = 0.2126 * png.data[i] + 0.7152 * png.data[i + 1] + 0.0722 * png.data[i + 2];
    sum += luma;
    if (png.data[i + 3] > 0) opaque++;
    if (luma > max) max = luma;
  }
  return {
    width: png.width,
    height: png.height,
    mean: sum / pixels,
    max,
    pixels,
    opaqueRatio: pixels === 0 ? 0 : opaque / pixels,
  };
}

// Ratio of pixels that changed by more than a small per-channel threshold, plus the
// mean absolute delta. A blank or unchanged frame yields a near-zero ratio, which is
// what the "material change must alter the frame" gate refuses to accept.
export function diffPng(aBuffer, bBuffer, threshold = 6) {
  const a = PNG.sync.read(aBuffer);
  const b = PNG.sync.read(bBuffer);
  const width = Math.min(a.width, b.width);
  const height = Math.min(a.height, b.height);
  let changed = 0;
  let totalDelta = 0;
  const total = width * height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ao = (y * a.width + x) * 4;
      const bo = (y * b.width + x) * 4;
      const d = Math.max(
        Math.abs(a.data[ao] - b.data[bo]),
        Math.abs(a.data[ao + 1] - b.data[bo + 1]),
        Math.abs(a.data[ao + 2] - b.data[bo + 2]),
      );
      totalDelta += d;
      if (d > threshold) changed++;
    }
  }
  return {
    width,
    height,
    changedPixels: changed,
    changedRatio: total === 0 ? 0 : changed / total,
    meanDelta: total === 0 ? 0 : totalDelta / total,
  };
}
