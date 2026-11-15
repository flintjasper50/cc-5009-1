// Browser runtime discovery. A clean environment must be able to find a supported
// Chromium and the SwiftShader Vulkan ICD without relying on a developer's local
// cache. Every path is reported so the evidence file says exactly what ran.

import fs from 'node:fs';
import path from 'node:path';
import { repoRoot } from './config.mjs';

export class EnvironmentError extends Error {
  constructor(problems) {
    super(`渲染环境不完整：\n  - ${problems.join('\n  - ')}`);
    this.name = 'EnvironmentError';
    this.problems = problems;
  }
}

function firstExisting(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return null;
}

// Playwright's browsers live either in the env-provided cache or the default per-user
// cache. We accept both, and also an explicit AIML_CHROME override.
export function findChrome() {
  const env = process.env;
  if (env.AIML_CHROME) {
    if (!fs.existsSync(env.AIML_CHROME)) throw new EnvironmentError([`AIML_CHROME 指向的文件不存在：${env.AIML_CHROME}`]);
    return env.AIML_CHROME;
  }
  const caches = [
    env.PLAYWRIGHT_BROWSERS_PATH,
    path.join(env.HOME ?? '', '.cache/ms-playwright'),
    path.join(repoRoot(), '.browsers'),
  ].filter(Boolean);
  const candidates = [];
  for (const cache of caches) {
    if (!fs.existsSync(cache)) continue;
    for (const entry of fs.readdirSync(cache)) {
      if (!/^chromium-\d+$/.test(entry)) continue;
      candidates.push(path.join(cache, entry, 'chrome-linux', 'chrome'));
      candidates.push(path.join(cache, entry, 'chrome-headless-shell-linux64', 'chrome-headless-shell'));
    }
  }
  const found = firstExisting(candidates);
  if (!found) {
    throw new EnvironmentError([
      '未找到 Chromium。请在干净环境中运行 `node node_modules/playwright-core/cli.js install chromium`，或设置 AIML_CHROME。',
    ]);
  }
  return found;
}

// Returns the browser's bundled SwiftShader ICD plus any sysroot library paths the
// operator prepared. Missing ICD is not fatal here: the capability probe decides
// whether hardware or software is actually available and the gate refuses mismatch.
export function findVulkanIcd(chromePath) {
  const env = process.env;
  if (env.AIML_VK_ICD) {
    if (!fs.existsSync(env.AIML_VK_ICD)) throw new EnvironmentError([`AIML_VK_ICD 指向的文件不存在：${env.AIML_VK_ICD}`]);
    return env.AIML_VK_ICD;
  }
  const bundled = path.join(path.dirname(chromePath), 'vk_swiftshader_icd.json');
  if (fs.existsSync(bundled)) return bundled;
  const sysroot = env.AIML_SYSROOT;
  if (sysroot) {
    const candidates = [
      path.join(sysroot, 'usr/share/vulkan/icd.d/lvp_icd.aarch64.json'),
      path.join(sysroot, 'usr/share/vulkan/icd.d/lvp_icd.x86_64.json'),
    ];
    const found = firstExisting(candidates);
    if (found) return found;
  }
  return null;
}

export function libraryPaths(chromePath) {
  const env = process.env;
  if (env.AIML_LD_LIBRARY_PATH) return env.AIML_LD_LIBRARY_PATH;
  const parts = [];
  const sysroot = env.AIML_SYSROOT;
  if (sysroot) {
    const archDir = process.arch === 'arm64' ? 'aarch64-linux-gnu' : 'x86_64-linux-gnu';
    parts.push(path.join(sysroot, 'usr/lib', archDir), path.join(sysroot, 'lib', archDir));
  }
  const bundled = path.dirname(chromePath);
  if (fs.existsSync(bundled)) parts.push(bundled);
  return parts.join(':');
}

// Launch arguments come from release/release.config.json. The fallback below exists
// only so an operator can launch a browser by hand; the gate always passes the config.
export function chromiumLaunchArgs(config) {
  if (config?.launch?.args) return [...config.launch.args];
  return [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--use-angle=swiftshader',
    '--use-webgpu-adapter=swiftshader',
    '--disable-gpu-sandbox',
    '--disable-vulkan-surface',
  ];
}

// Expands the configured environment template. `$ICD` and `$LIBRARY_PATH` are the
// values discovered from this clean environment, so the config names the variables
// while the host supplies the actual paths.
export function launchEnvironment(config, { icd, libraryPath }) {
  const env = { ...process.env };
  const template = config?.launch?.env ?? { VK_ICD_FILENAMES: '$ICD', VK_DRIVER_FILES: '$ICD', LD_LIBRARY_PATH: '$LIBRARY_PATH' };
  for (const [key, value] of Object.entries(template)) {
    if (typeof value !== 'string') continue;
    const expanded = value.replace(/\$ICD/g, icd ?? '').replace(/\$LIBRARY_PATH/g, libraryPath ?? '');
    if (expanded !== '') env[key] = expanded;
  }
  return env;
}
