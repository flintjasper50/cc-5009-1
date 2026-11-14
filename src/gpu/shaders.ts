// WGSL for the material preview. Two entry points share one shading function:
//  - fs_main  : studio path, writes linear radiance into an rgba16float HDR target
//  - fs_bench : benchmark path, applies exposure + sRGB OETF and writes rgba8unorm
// One shared `shade()` guarantees benchmarked pixels come from the exact maths
// the interactive viewport uses.

export const SCENE_WGSL = /* wgsl */ `
struct Light {
  dirIntensity: vec4f,   // xyz = unit direction toward the light (world), w = radiant intensity
  color: vec4f,          // rgb = linear light colour, w = angular radius (radians)
};

struct Frame {
  viewProj: mat4x4f,
  cameraPos: vec4f,
  ambient: vec4f,
  exposure: vec4f,
  lights: array<Light, 3>,
  flags: vec4f,          // bit0 = diffuse-only
};

struct Model {
  model: mat4x4f,
  nm0: vec4f,
  nm1: vec4f,
  nm2: vec4f,
  baseColor: vec4f,
  params: vec4f,         // roughness, metallic, normalStrength, normalEnabled
  misc: vec4f,           // uvScale, solidAlbedo, 0, 0
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> model: Model;
@group(2) @binding(0) var albedoTex: texture_2d<f32>;
@group(2) @binding(1) var normalTex: texture_2d<f32>;
@group(2) @binding(2) var samp: sampler;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) nWorld: vec3f,
  @location(2) tWorld: vec3f,
  @location(3) handed: f32,
  @location(4) uv: vec2f,
};

@vertex
fn vs_main(
  @location(0) pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) tangent: vec4f,
  @location(3) uv: vec2f,
) -> VOut {
  var out: VOut;
  let wp = model.model * vec4f(pos, 1.0);
  out.clip = frame.viewProj * wp;
  out.worldPos = wp.xyz;

  // Normals need the inverse-transpose of the model matrix. With non-uniform scale
  // the model matrix itself would shear them off the true surface normal. Tangents
  // are surface directions and use the model matrix directly.
  let nm = mat3x3f(model.nm0.xyz, model.nm1.xyz, model.nm2.xyz);
  out.nWorld = nm * normal;
  out.tWorld = (model.model * vec4f(tangent.xyz, 0.0)).xyz;
  out.handed = tangent.w;
  out.uv = uv;
  return out;
}

const PI = 3.141592653589793;

fn safeNormalize(v: vec3f, fallback: vec3f) -> vec3f {
  let l = length(v);
  return select(fallback, v / l, l > 1e-7);
}

fn distributionGGX(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Height-correlated Smith visibility (Heitz). Contains the 1/(4 nv nl) denominator.
fn visibilitySmithGGX(nDotV: f32, nDotL: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let gv = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
  let gl = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-6);
}

// Effective GGX roughness for a finite disc light: convolve the surface slope
// distribution with the light's angular radius. A punctual light (radius 0) would make
// a mirror infinitely bright at exact alignment and black otherwise, which is neither
// renderable nor comparable; the finite disc keeps specular peaks bounded and smooth.
fn effectiveAlpha(alpha: f32, angularRadius: f32) -> f32 {
  let r = max(2.0 * angularRadius, 0.001);
  return sqrt(alpha * alpha + r * r);
}

fn fresnelSchlick(u: f32, f0: vec3f) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow(clamp(1.0 - u, 0.0, 1.0), 5.0);
}

// Exact IEC 61966-2-1 sRGB OETF. Values are clamped to [0,1] after exposure before
// the piecewise curve so the transfer stays defined; this is the reference encode.
fn srgbOETF(c: f32) -> f32 {
  let v = clamp(c, 0.0, 1.0);
  return select(1.055 * pow(v, 1.0 / 2.4) - 0.055, v * 12.92, v <= 0.0031308);
}

fn displayEncode(linear: vec3f, stops: f32) -> vec3f {
  let e = linear * exp2(stops);
  return vec3f(srgbOETF(e.r), srgbOETF(e.g), srgbOETF(e.b));
}

// Tangent-space normal map decode. Bytes store a direction as n*0.5+0.5 with no colour
// transfer; decoding is 2n-1 and must never go through sRGB logic.
fn decodeNormalMap(uv: vec2f, strength: f32) -> vec3f {
  let enc = textureSample(normalTex, samp, uv).xyz;
  let decoded = enc * 2.0 - vec3f(1.0);
  return normalize(vec3f(decoded.xy * strength, max(decoded.z, 0.15)));
}

fn shade(in: VOut, front: bool) -> vec3f {
  // --- Space conversion -------------------------------------------------------
  var nW = normalize(in.nWorld);
  if (!front) {
    nW = -nW;
  }
  // Interpolation does not preserve orthogonality; rebuild an orthonormal basis.
  let tW = safeNormalize(in.tWorld - nW * dot(in.tWorld, nW), vec3f(1.0, 0.0, 0.0));
  let bW = normalize(cross(nW, tW)) * select(-1.0, 1.0, in.handed >= 0.0);

  let uv = in.uv * model.misc.x;
  var nTS = vec3f(0.0, 0.0, 1.0);
  if (model.params.w > 0.5) {
    nTS = decodeNormalMap(uv, model.params.z);
  }

  let viewWorld = safeNormalize(frame.cameraPos.xyz - in.worldPos, vec3f(0.0, 0.0, 1.0));
  // World -> tangent. The BRDF runs in the tangent frame, so the decoded map normal
  // is simply the shading normal there.
  let vTS = normalize(vec3f(dot(viewWorld, tW), dot(viewWorld, bW), dot(viewWorld, nW)));

  let diffuseOnly = (u32(frame.flags.x) & 1u) != 0u;
  let solidAlbedo = model.misc.y > 0.5;

  var albedo = model.baseColor.rgb;
  if (!solidAlbedo) {
    // Albedo is stored sRGB-encoded; the srgb texture view decodes on sample, so this
    // value is already linear light. The base tint is linear too (uploaded pre-decoded).
    albedo = clamp(model.baseColor.rgb * textureSample(albedoTex, samp, uv).rgb, vec3f(0.0), vec3f(1.0));
  }

  let rough = clamp(model.params.x, 0.02, 1.0);
  let metal = clamp(model.params.y, 0.0, 1.0);
  let alpha = max(rough * rough, 0.0005);
  let f0 = mix(vec3f(0.04), albedo, metal);

  var radiance = vec3f(0.0);
  for (var i = 0u; i < 3u; i++) {
    let light = frame.lights[i];
    let lW = safeNormalize(light.dirIntensity.xyz, vec3f(0.0, 0.0, 1.0));
    let lTS = normalize(vec3f(dot(lW, tW), dot(lW, bW), dot(lW, nW)));
    let hTS = normalize(vTS + lTS);
    let nDotL = max(dot(nTS, lTS), 0.0);
    if (nDotL <= 0.0) {
      continue;
    }
    let nDotV = max(dot(nTS, vTS), 1e-4);
    let nDotH = max(dot(nTS, hTS), 0.0);
    let vDotH = max(dot(vTS, hTS), 0.0);

    let f = fresnelSchlick(vDotH, f0);
    // Diffuse-only mode is textbook Lambert: no specular lobe and no Fresnel energy
    // compensation, so the benchmark can be checked against albedo/PI * sum(...).
    var kd = (vec3f(1.0) - f) * (1.0 - metal);
    if (diffuseOnly) {
      kd = vec3f(1.0 - metal);
    }
    let diffuse = kd * albedo / PI;

    var specular = vec3f(0.0);
    if (!diffuseOnly) {
      let alphaEff = effectiveAlpha(alpha, light.color.w);
      let d = distributionGGX(nDotH, alphaEff);
      let vis = visibilitySmithGGX(nDotV, nDotL, alphaEff);
      specular = d * vis * f;
    }

    radiance += (diffuse + specular) * light.color.rgb * light.dirIntensity.w * nDotL;
  }

  // Explicit constant Lambertian ambient term; no implicit environment probe.
  radiance += frame.ambient.rgb * albedo * (1.0 - metal);
  return radiance;
}

@fragment
fn fs_main(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  return vec4f(shade(in, front), 1.0);
}

@fragment
fn fs_bench(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  return vec4f(displayEncode(shade(in, front), frame.exposure.x), 1.0);
}
`;

// Fullscreen composite: exposure in linear light, then sRGB OETF, then write to the
// non-srgb swapchain. Doing the transfer explicitly keeps CPU reference and GPU path
// identical.
export const COMPOSITE_WGSL = /* wgsl */ `
struct Composite {
  exposure: vec4f,   // x = exposure stops scale, yz = viewport size in pixels
};

@group(0) @binding(0) var<uniform> comp: Composite;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;

struct VOut {
  @builtin(position) clip: vec4f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VOut {
  var out: VOut;
  var p = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  out.clip = vec4f(p[i], 0.0, 1.0);
  return out;
}

// Exact IEC 61966-2-1 sRGB OETF. Values are clamped to [0,1] after exposure before
// the piecewise curve so the transfer stays defined; this is the reference encode.
fn srgbOETF(c: f32) -> f32 {
  let v = clamp(c, 0.0, 1.0);
  return select(1.055 * pow(v, 1.0 / 2.4) - 0.055, v * 12.92, v <= 0.0031308);
}

// Studio backdrop: the scene pass clears to alpha 0, geometry writes alpha 1. Uncovered
// pixels get an explicit linear-light gradient instead of clearing to black, so the
// viewport always shows a rendered backdrop rather than an ambiguous black frame.
fn studioBackdrop(uv: vec2f) -> vec3f {
  let d = distance(uv, vec2f(0.5, 0.34));
  let glow = exp(-d * 3.4) * 0.050;
  let horizon = smoothstep(0.15, 1.0, uv.y);
  let base = mix(vec3f(0.030, 0.033, 0.040), vec3f(0.011, 0.012, 0.015), horizon);
  return base + vec3f(glow, glow * 0.96, glow * 0.88);
}

@fragment
fn fs_composite(in: VOut) -> @location(0) vec4f {
  let pixel = vec2i(in.clip.xy);
  let sample = textureLoad(hdrTex, pixel, 0);
  let uv = vec2f((f32(pixel.x) + 0.5) / comp.exposure.y, (f32(pixel.y) + 0.5) / comp.exposure.z);
  let linear = select(studioBackdrop(uv), sample.rgb, sample.a > 0.5);
  let exposed = linear * exp2(comp.exposure.x);
  return vec4f(srgbOETF(exposed.r), srgbOETF(exposed.g), srgbOETF(exposed.b), 1.0);
}
`;
