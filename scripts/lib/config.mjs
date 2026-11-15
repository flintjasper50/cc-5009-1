// Release config loader. Every field the gate relies on is required: a missing or
// malformed value is a hard error, never a default. That is what keeps the material
// lab's release config from "succeeding" by falling back to something that was not
// reviewed.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');

export class ConfigError extends Error {
  constructor(problems) {
    super(`发布配置无效：\n  - ${problems.join('\n  - ')}`);
    this.name = 'ConfigError';
    this.problems = problems;
  }
}

export function configPath() {
  return path.join(root, 'release', 'release.config.json');
}

export function repoRoot() {
  return root;
}

export function loadReleaseConfig(file = configPath()) {
  let raw;
  try {
    raw = fs.readFileSync(file, 'utf8');
  } catch (error) {
    throw new ConfigError([`无法读取 ${file}：${error instanceof Error ? error.message : String(error)}`]);
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new ConfigError([`${file} 不是合法 JSON：${error instanceof Error ? error.message : String(error)}`]);
  }
  return assertConfig(parsed);
}

// Strict structural validation. Undeclared keys are allowed, missing required keys are
// not: the gate must never infer a release decision from an absent setting.
export function assertConfig(value) {
  const problems = [];
  const isObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v);
  const needObject = (obj, key) => {
    const v = obj[key];
    if (!isObject(v)) { problems.push(`缺少对象字段 ${key}`); return {}; }
    return v;
  };
  const needString = (obj, key, label = key) => {
    const v = obj[key];
    if (typeof v !== 'string' || v.length === 0) { problems.push(`${label} 必须是非空字符串`); return ''; }
    return v;
  };
  const needNumber = (obj, key, label = key) => {
    const v = obj[key];
    if (typeof v !== 'number' || !Number.isFinite(v)) { problems.push(`${label} 必须是有限数字`); return 0; }
    return v;
  };
  const needBoolean = (obj, key, label = key) => {
    const v = obj[key];
    if (typeof v !== 'boolean') { problems.push(`${label} 必须是布尔值`); return false; }
    return v;
  };
  const needStringArray = (obj, key, label = key) => {
    const v = obj[key];
    if (!Array.isArray(v) || v.length === 0 || !v.every((x) => typeof x === 'string')) {
      problems.push(`${label} 必须是非空字符串数组`); return [];
    }
    return v;
  };

  if (!isObject(value)) throw new ConfigError(['配置根必须是对象']);
  if (value.schema !== 'aiml.release.config/v1') problems.push('schema 必须是 aiml.release.config/v1');

  const app = needObject(value, 'app');
  const appOut = {
    name: needString(app, 'name'),
    entry: needString(app, 'entry'),
    expectedTitleIncludes: needString(app, 'expectedTitleIncludes'),
  };

  const build = needObject(value, 'build');
  const buildOut = {
    command: needStringArray(build, 'command', 'build.command'),
    outputDir: needString(build, 'outputDir'),
    mustExist: needStringArray(build, 'mustExist'),
    forbidPaths: needStringArray(build, 'forbidPaths'),
    minMainJsBytes: needNumber(build, 'minMainJsBytes'),
  };

  const render = needObject(value, 'render');
  const limitsRaw = needObject(render, 'requiredLimits');
  const limits = {};
  for (const [key, raw] of Object.entries(limitsRaw)) {
    if (typeof raw !== 'number' || !Number.isFinite(raw)) { problems.push(`render.requiredLimits.${key} 必须是有限数字`); continue; }
    limits[key] = raw;
  }
  if (Object.keys(limits).length === 0) problems.push('render.requiredLimits 至少需要一项');
  const renderOut = {
    backend: needString(render, 'backend'),
    requiredAdapterFeature: needString(render, 'requiredAdapterFeature'),
    requiredLimits: limits,
    requiredTextureFormats: needStringArray(render, 'requiredTextureFormats'),
    requiredFloatTargets: needStringArray(render, 'requiredFloatTargets'),
    allowSoftwareRasterizer: needBoolean(render, 'allowSoftwareRasterizer'),
  };

  const fixtures = needObject(value, 'fixtures');
  const cameraRaw = needObject(fixtures, 'camera');
  const target = cameraRaw.target;
  if (!Array.isArray(target) || target.length !== 3 || !target.every((x) => typeof x === 'number' && Number.isFinite(x))) {
    problems.push('fixtures.camera.target 必须是三个有限数字');
  }
  const fixturesOut = {
    swatch: needString(fixtures, 'swatch'),
    preset: needString(fixtures, 'preset'),
    camera: {
      yaw: needNumber(cameraRaw, 'yaw', 'fixtures.camera.yaw'),
      pitch: needNumber(cameraRaw, 'pitch', 'fixtures.camera.pitch'),
      distance: needNumber(cameraRaw, 'distance', 'fixtures.camera.distance'),
      target: Array.isArray(target) && target.length === 3 ? target : [0, 0, 0],
      fovY: needNumber(cameraRaw, 'fovY', 'fixtures.camera.fovY'),
    },
    roughness: needNumber(fixtures, 'roughness'),
    roughnessAlt: needNumber(fixtures, 'roughnessAlt'),
    rotatedYaw: needNumber(fixtures, 'rotatedYaw'),
    referenceFrame: needString(fixtures, 'referenceFrame'),
  };

  const gates = needObject(value, 'gates');
  const gatesOut = {
    shadersMustCompile: needBoolean(gates, 'shadersMustCompile'),
    frameMustBeNonBlank: needBoolean(gates, 'frameMustBeNonBlank'),
    materialChangeMustAlterFrame: needBoolean(gates, 'materialChangeMustAlterFrame'),
    rotationMustAlterFrame: needBoolean(gates, 'rotationMustAlterFrame'),
    benchMaxAbsError: needNumber(gates, 'benchMaxAbsError'),
    minFrameMeanLuma: needNumber(gates, 'minFrameMeanLuma'),
    minChangedPixelRatio: needNumber(gates, 'minChangedPixelRatio'),
    noHalfAppliedMaterial: needBoolean(gates, 'noHalfAppliedMaterial'),
  };

  const artifacts = needObject(value, 'artifacts');
  const artifactsOut = {
    dir: needString(artifacts, 'dir'),
    framesDir: needString(artifacts, 'framesDir'),
    evidenceFile: needString(artifacts, 'evidenceFile'),
  };

  const launch = needObject(value, 'launch');
  const launchOut = {
    args: needStringArray(launch, 'args'),
    env: needObject(launch, 'env'),
  };

  if (problems.length > 0) throw new ConfigError(problems);

  return {
    schema: 'aiml.release.config/v1',
    app: appOut,
    build: buildOut,
    render: renderOut,
    fixtures: fixturesOut,
    gates: gatesOut,
    artifacts: artifactsOut,
    launch: launchOut,
  };
}

export function resolvePath(rel) {
  return path.isAbsolute(rel) ? rel : path.join(root, rel);
}
