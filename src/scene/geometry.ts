import { normalize3, cross3, type Vec3 } from '../math/linalg.ts';

export interface MeshData {
  positions: Float32Array;
  normals: Float32Array;
  tangents: Float32Array; // vec4: xyz + handedness
  uvs: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
  triangleCount: number;
}

export type ParamSurface = (u: number, v: number) => Vec3;

interface GridOptions {
  nu: number;
  nv: number;
  wrapU?: boolean;
  wrapV?: boolean;
  uvScaleU?: number;
  uvScaleV?: number;
  // Planar UV projection in world XY. Used for the cushion where the parametric
  // u/v direction is not aligned with the visible weave pattern.
  planarUV?: { center: Vec3; sizeU: number; sizeV: number; uvScaleU: number; uvScaleV: number };
}

// Builds a triangle mesh from a parametric surface P(u,v). Normals come from the
// analytic-ish finite difference of the surface, so every generated model has
// correct geometry normals without relying on imported assets.
export function buildParamGrid(surface: ParamSurface, opts: GridOptions): MeshData {
  const { nu, nv, wrapU = false, wrapV = false } = opts;
  const cellU = 1 / nu;
  const cellV = 1 / nv;
  const cols = wrapU ? nu : nu + 1;
  const rows = wrapV ? nv : nv + 1;
  const vertexCount = cols * rows;

  const pos = new Float32Array(vertexCount * 3);
  const nor = new Float32Array(vertexCount * 3);
  const tan = new Float32Array(vertexCount * 4);
  const uv = new Float32Array(vertexCount * 2);

  const P = (u: number, v: number): Vec3 => {
    const uu = wrapU ? ((u % 1) + 1) % 1 : Math.min(1, Math.max(0, u));
    const vv = wrapV ? ((v % 1) + 1) % 1 : Math.min(1, Math.max(0, v));
    return surface(uu, vv);
  };

  const hu = 0.5 * cellU;
  const hv = 0.5 * cellV;
  const planar = opts.planarUV;

  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const u = (i / nu) % (wrapU ? 1 : 2);
      const v = (j / nv) % (wrapV ? 1 : 2);
      const idx = j * cols + i;
      const p = P(u, v);
      const du = subtract(P(u + hu, v), P(u - hu, v));
      const dv = subtract(P(u, v + hv), P(u, v - hv));
      let n = normalize3(cross3(du, dv));
      // Degenerate rows (poles) fall back to outward orientation.
      if (!Number.isFinite(n[0]) || lengthSq(n) < 1e-12) {
        n = normalize3(p);
      }
      let t: Vec3;
      if (planar) {
        // Project +X into the tangent plane, keep it orthogonal to N.
        const xAxis: Vec3 = [1, 0, 0];
        const proj = subtract(xAxis, scale(n, dot(n, xAxis)));
        t = normalize3(proj);
        if (lengthSq(t) < 1e-12) t = normalize3(du);
      } else {
        t = normalize3(du);
      }
      const bitangent = cross3(n, t);
      const handedness = dot(bitangent, dv) >= 0 ? 1 : -1;
      t = subtract(t, scale(n, dot(n, t)));
      t = normalize3(t);

      pos[idx * 3] = p[0]; pos[idx * 3 + 1] = p[1]; pos[idx * 3 + 2] = p[2];
      nor[idx * 3] = n[0]; nor[idx * 3 + 1] = n[1]; nor[idx * 3 + 2] = n[2];
      tan[idx * 4] = t[0]; tan[idx * 4 + 1] = t[1]; tan[idx * 4 + 2] = t[2]; tan[idx * 4 + 3] = handedness;
      if (planar) {
        uv[idx * 2] = (p[0] - planar.center[0]) / planar.sizeU * planar.uvScaleU;
        uv[idx * 2 + 1] = (p[1] - planar.center[1]) / planar.sizeV * planar.uvScaleV;
      } else {
        uv[idx * 2] = u * (opts.uvScaleU ?? 1);
        uv[idx * 2 + 1] = v * (opts.uvScaleV ?? 1);
      }
    }
  }

  const indexCount = (wrapU ? nu : nu) * (wrapV ? nv : nv) * 6;
  const indices = new Uint32Array(indexCount);
  let k = 0;
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const i0 = i;
      const i1 = wrapU ? (i + 1) % nu : i + 1;
      const j0 = j;
      const j1 = wrapV ? (j + 1) % nv : j + 1;
      const a = j0 * cols + i0;
      const b = j0 * cols + i1;
      const c = j1 * cols + i0;
      const d = j1 * cols + i1;
      indices[k++] = a; indices[k++] = b; indices[k++] = c;
      indices[k++] = b; indices[k++] = d; indices[k++] = c;
    }
  }

  return { positions: pos, normals: nor, tangents: tan, uvs: uv, indices, vertexCount, indexCount, triangleCount: indexCount / 3 };
}

// Cushion built as a closed superellipsoid. Front and back faces are flat-ish with
// rounded corners, so the visible front reads as a padded soft-touch sample.
export function buildCushion(a: number, b: number, c: number, e1: number, e2: number, nu: number, nv: number,
  uvScaleU: number, uvScaleV: number): MeshData {
  const sgnPow = (x: number, e: number): number => Math.sign(x) * Math.pow(Math.abs(x), e);
  const surface: ParamSurface = (u, v) => {
    const theta = u * Math.PI * 2;
    const phi = (v - 0.5) * Math.PI;
    const cv = sgnPow(Math.cos(phi), e1);
    const sv = sgnPow(Math.sin(phi), e1);
    return [
      a * cv * sgnPow(Math.cos(theta), e2),
      b * cv * sgnPow(Math.sin(theta), e2),
      c * sv,
    ];
  };

  const V = (u: number, v: number): Vec3 => {
    const theta = u * Math.PI * 2;
    const phi = (v - 0.5) * Math.PI * 0.995;
    const cv = sgnPow(Math.cos(phi), e1);
    const sv = sgnPow(Math.sin(phi), e1);
    return [
      a * cv * sgnPow(Math.cos(theta), e2),
      b * cv * sgnPow(Math.sin(theta), e2),
      c * sv,
    ];
  };

  // Use an inset latitude range plus explicit pole vertices so the caps are watertight
  // and normals at the poles remain well defined.
  const rows = nv + 1;
  const cols = nu;
  const poleFront = cols * rows;
  const poleBack = poleFront + 1;
  const vertexCount = poleBack + 1;

  const pos = new Float32Array(vertexCount * 3);
  const nor = new Float32Array(vertexCount * 3);
  const tan = new Float32Array(vertexCount * 4);
  const uv = new Float32Array(vertexCount * 2);

  for (let j = 0; j < rows; j++) {
    const v = j / nv;
    for (let i = 0; i < cols; i++) {
      const u = i / nu;
      const idx = j * cols + i;
      const p = V(u, v);
      const hu = 1 / (nu * 2);
      const hv = 1 / (nv * 2);
      const du = subtract(V(u + hu, v), V(u - hu, v));
      const dv = subtract(V(u, v + hv), V(u, v - hv));
      let n = normalize3(cross3(du, dv));
      const t = normalize3(subtract([1, 0, 0], scale(n, dot(n, [1, 0, 0]))));
      const handedness = dot(cross3(n, t), [0, 1, 0]) >= 0 ? 1 : -1;
      pos[idx * 3] = p[0]; pos[idx * 3 + 1] = p[1]; pos[idx * 3 + 2] = p[2];
      nor[idx * 3] = n[0]; nor[idx * 3 + 1] = n[1]; nor[idx * 3 + 2] = n[2];
      tan[idx * 4] = t[0]; tan[idx * 4 + 1] = t[1]; tan[idx * 4 + 2] = t[2]; tan[idx * 4 + 3] = handedness;
      uv[idx * 2] = (p[0] / a * 0.5 + 0.5) * uvScaleU;
      uv[idx * 2 + 1] = (p[1] / b * 0.5 + 0.5) * uvScaleV;
    }
  }

  const setVertex = (idx: number, p: Vec3, n: Vec3, uvx: number, uvy: number) => {
    pos[idx * 3] = p[0]; pos[idx * 3 + 1] = p[1]; pos[idx * 3 + 2] = p[2];
    nor[idx * 3] = n[0]; nor[idx * 3 + 1] = n[1]; nor[idx * 3 + 2] = n[2];
    tan[idx * 4] = 1; tan[idx * 4 + 1] = 0; tan[idx * 4 + 2] = 0; tan[idx * 4 + 3] = 1;
    uv[idx * 2] = uvx; uv[idx * 2 + 1] = uvy;
  };
  setVertex(poleFront, [0, 0, c], [0, 0, 1], 0.5 * uvScaleU, 0.5 * uvScaleV);
  setVertex(poleBack, [0, 0, -c], [0, 0, -1], 0.5 * uvScaleU, 0.5 * uvScaleV);

  const idxList: number[] = [];
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const i0 = i;
      const i1 = (i + 1) % nu;
      const a0 = j * cols + i0, b0 = j * cols + i1;
      const c0 = (j + 1) * cols + i0, d0 = (j + 1) * cols + i1;
      idxList.push(a0, b0, c0, b0, d0, c0);
    }
  }
  // Caps: fan around the pole vertices. Front pole connects the last ring (v≈max).
  const lastRing = nv * cols;
  for (let i = 0; i < nu; i++) {
    const i1 = (i + 1) % nu;
    idxList.push(poleFront, lastRing + i, lastRing + i1);
    idxList.push(poleBack, i1, i);
  }

  const indices = new Uint32Array(idxList);
  return { positions: pos, normals: nor, tangents: tan, uvs: uv, indices, vertexCount, indexCount: indices.length, triangleCount: indices.length / 3 };
}

export function buildBenchPlane(size = 4): MeshData {
  // Fronto-parallel plane at z=0 facing +Z. Used for numeric benchmark comparisons.
  const positions = new Float32Array([
    -size / 2, -size / 2, 0,
    size / 2, -size / 2, 0,
    size / 2, size / 2, 0,
    -size / 2, size / 2, 0,
  ]);
  const normals = new Float32Array([
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  ]);
  const tangents = new Float32Array([
    1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
  ]);
  const uvs = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return { positions, normals, tangents, uvs, indices, vertexCount: 4, indexCount: 6, triangleCount: 2 };
}

function subtract(a: Vec3, b: Vec3): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale(a: Vec3, s: number): Vec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot(a: Vec3, b: Vec3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function lengthSq(a: Vec3): number { return dot(a, a); }

export interface SceneGeometry {
  dashboard: MeshData;
  trim: MeshData;
  softpad: MeshData;
  bench: MeshData;
}

export function buildSceneGeometry(): SceneGeometry {
  const dashSurface: ParamSurface = (u, v) => {
    const x = -1.35 + 2.7 * u;
    const y = -0.24 + 0.82 * smoothstep(0, 1, v);
    // Cockpit brow: bulges forward across the top, wings recede at the sides.
    const wing = Math.pow(Math.abs(2 * u - 1), 2.2);
    const brow = 0.09 * Math.pow(v, 5);
    const centerLift = 0.075 * Math.exp(-Math.pow((u - 0.52) / 0.22, 2)) * smoothstep(0.2, 0.7, v);
    const z = 0.10 + 0.36 * Math.pow(v, 1.25) - 0.10 * wing + brow + centerLift;
    return [x, y, z];
  };

  const trimSurface: ParamSurface = (u, v) => {
    const x = -1.26 + 2.52 * u;
    const yc = 0.235 + 0.035 * Math.sin(u * Math.PI * 2 + 0.6) * (1 - u * 0.4);
    const y = yc + (v - 0.5) * 0.075;
    const wing = Math.pow(Math.abs(2 * u - 1), 2.2);
    const dashZ = 0.10 + 0.36 * Math.pow((yc + 0.24) / 0.82, 1.25) - 0.10 * wing;
    const z = dashZ + 0.045 + 0.012 * Math.sin(v * Math.PI) + 0.006 * Math.cos(u * Math.PI * 3);
    return [x, y, z];
  };

  const dashboard = buildParamGrid(dashSurface, { nu: 192, nv: 128, uvScaleU: 3.2, uvScaleV: 2.0 });
  const trim = buildParamGrid(trimSurface, { nu: 256, nv: 24, uvScaleU: 8.0, uvScaleV: 0.30 });
  const cushion = buildCushion(0.34, 0.24, 0.068, 0.42, 0.40, 120, 72, 1.7, 1.3);
  // Seat the padded sample on the passenger side. The dash surface at (u=0.70,v=0.45)
  // sits at z ~= 0.24, so the cushion centre is placed just proud of it and pitched
  // slightly back to match the surface slope.
  rotateMesh(cushion, -0.08, -0.10);
  translateMesh(cushion, [0.54, 0.10, 0.295]);
  const bench = buildBenchPlane(4);
  return { dashboard, trim, softpad: cushion, bench };
}

function translateMesh(mesh: MeshData, t: Vec3): void {
  for (let i = 0; i < mesh.vertexCount; i++) {
    mesh.positions[i * 3] += t[0];
    mesh.positions[i * 3 + 1] += t[1];
    mesh.positions[i * 3 + 2] += t[2];
  }
}

function rotateMesh(mesh: MeshData, yaw: number, pitch: number): void {
  const rot = rotationMatrix(yaw, pitch);
  for (let i = 0; i < mesh.vertexCount; i++) {
    const p: Vec3 = [mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]];
    const n: Vec3 = [mesh.normals[i * 3], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]];
    const tg: Vec3 = [mesh.tangents[i * 4], mesh.tangents[i * 4 + 1], mesh.tangents[i * 4 + 2]];
    const rp = mul3(rot, p);
    const rn = mul3(rot, n);
    const rt = mul3(rot, tg);
    mesh.positions[i * 3] = rp[0];
    mesh.positions[i * 3 + 1] = rp[1];
    mesh.positions[i * 3 + 2] = rp[2];
    mesh.normals[i * 3] = rn[0]; mesh.normals[i * 3 + 1] = rn[1]; mesh.normals[i * 3 + 2] = rn[2];
    mesh.tangents[i * 4] = rt[0]; mesh.tangents[i * 4 + 1] = rt[1]; mesh.tangents[i * 4 + 2] = rt[2];
  }
}

function rotationMatrix(yaw: number, pitch: number): number[] {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  return [
    cy, sy * sp, sy * cp,
    0, cp, -sp,
    -sy, cy * sp, cy * cp,
  ];
}

function mul3(m: number[], v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
