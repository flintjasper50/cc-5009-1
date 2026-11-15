// Release gate for the automotive interior material lab.
//
// What this proves, in order, and blocks on:
//   1. A reproducible build from a clean tree produces a runnable dist/.
//   2. A real WebGPU adapter exposes every required texture format and float target.
//   3. The shaders actually compile (per-module compilation info + pipeline creation).
//   4. The interior material kernel drives the controls; invalid input is rejected
//      whole, never half-applied.
//   5. The original sample preset replays through the kernel and drives the frame.
//   6. Rotating the specimen and changing roughness alter real rendered pixels.
//   7. The offscreen benchmark still matches its CPU reference.
//
// The gate never treats "the process is alive" or "the device exists" as acceptance.
// A software rasteriser frame is labelled as such in the evidence and cannot satisfy
// the hardware-path requirement.

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { chromium } from 'playwright-core';
import { loadReleaseConfig, resolvePath, repoRoot } from './lib/config.mjs';
import { EnvironmentError, chromiumLaunchArgs, findChrome, findVulkanIcd, launchEnvironment, libraryPaths } from './lib/environment.mjs';
import { probeCapabilities } from './lib/capability.mjs';
import { startStaticServer } from './lib/server.mjs';
import { readPng, summarizePng, diffPng } from './lib/png.mjs';

const args = process.argv.slice(2);
const skipBuild = args.includes('--skip-build');
const keepServer = args.includes('--keep-server');
const allowSoftware = args.includes('--allow-software');
const missingFormat = flagValue('--missing-format');

function flagValue(name) {
  const hit = args.find((a) => a.startsWith(`${name}=`));
  return hit ? hit.slice(name.length + 1) : null;
}

const checks = [];
function record(name, pass, detail, blocking = true) {
  checks.push({ name, pass, detail, blocking });
  const mark = pass ? 'PASS' : blocking ? 'BLOCK' : 'WARN';
  console.log(`${mark} ${name}${detail ? ` — ${detail}` : ''}`);
}

class ReleaseBlocked extends Error {
  constructor(message) { super(message); this.name = 'ReleaseBlocked'; }
}

function commandLabel(command) {
  return command.join(' ');
}

async function main() {
  const startedAt = new Date().toISOString();
  const config = loadReleaseConfig();
  const root = repoRoot();
  const evidence = {
    schema: 'aiml.release.evidence/v1',
    startedAt,
    configFile: 'release/release.config.json',
    configSchema: config.schema,
    environment: {},
    build: null,
    capabilities: null,
    compile: null,
    kernel: null,
    frames: [],
    bench: [],
    checks: [],
    verdict: 'unknown',
  };

  // --- 1. Build ---------------------------------------------------------------
  if (!skipBuild) {
    const pre = gitStatus(root);
    if (pre.dirtyBuildInputs.length > 0) {
      record('构建输入干净', false, `存在未提交的构建输入：${pre.dirtyBuildInputs.slice(0, 5).join(', ')}`, false);
    } else {
      record('构建输入干净', true, 'git 工作树中的 src/、release/、index.html 无未提交改动', false);
    }
    const build = spawnSync(config.build.command[0], config.build.command.slice(1), {
      cwd: root,
      encoding: 'utf8',
      env: { ...process.env, CI: '1' },
    });
    evidence.build = {
      command: commandLabel(config.build.command),
      status: build.status,
      stdoutTail: tail(build.stdout, 20),
      stderrTail: tail(build.stderr, 20),
    };
    if (build.status !== 0) {
      record('可复现构建成功', false, `命令 ${commandLabel(config.build.command)} 退出码 ${build.status}`);
      return finish(evidence, config, root, startedAt);
    }
    record('可复现构建成功', true, commandLabel(config.build.command));
  } else {
    evidence.build = { command: '(skipped)', status: 0 };
    record('可复现构建成功', true, '本次跳过构建，使用现有 dist/', false);
  }

  const distDir = resolvePath(config.build.outputDir);
  for (const rel of config.build.mustExist) {
    const exists = fs.existsSync(path.join(distDir, rel));
    record(`产物包含 ${rel}`, exists, path.join(config.build.outputDir, rel));
    if (!exists) return finish(evidence, config, root, startedAt);
  }
  for (const rel of config.build.forbidPaths) {
    const exists = fs.existsSync(path.join(distDir, rel));
    record(`产物不含源目录 ${rel}`, !exists, exists ? '构建目录内出现了源码目录' : 'ok', false);
  }
  const mainJs = largestEntry(path.join(distDir, 'assets'));
  if (mainJs) {
    record('主脚本体积合理', mainJs.size >= config.build.minMainJsBytes, `${path.basename(mainJs.file)} ${mainJs.size} B`);
    evidence.build.mainJs = { file: path.relative(root, mainJs.file), bytes: mainJs.size };
  } else {
    record('主脚本存在', false, 'dist/assets 下未找到 .js');
    return finish(evidence, config, root, startedAt);
  }

  // --- 2. Environment + browser ----------------------------------------------
  let chrome;
  try {
    chrome = findChrome();
  } catch (error) {
    record('浏览器可用', false, error instanceof EnvironmentError ? error.problems.join('; ') : String(error));
    return finish(evidence, config, root, startedAt);
  }
  const icd = findVulkanIcd(chrome);
  const libs = libraryPaths(chrome);
  evidence.environment = { chrome, vulkanIcd: icd, libraryPath: libs, allowSoftware };
  record('浏览器可用', true, chrome);
  record('Vulkan ICD 可用', !!icd, icd ?? '未找到 ICD；将依赖驱动自带的软件光栅器', false);

  const server = await startStaticServer(distDir);
  let browser = null;
  try {
    browser = await chromium.launch({
      executablePath: chrome,
      args: chromiumLaunchArgs(config),
      env: launchEnvironment(config, { icd, libraryPath: libs }),
    });
  } catch (error) {
    record('浏览器可启动', false, `Chromium 启动失败：${error instanceof Error ? error.message : String(error)}`);
    if (!keepServer) await server.close();
    return finish(evidence, config, root, startedAt);
  }

  let logs = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

    // --- 3. Capability probe -------------------------------------------------
    // WebGPU is only exposed in a secure context, so the probe runs after loading
    // the built package over the local HTTP origin.
    await page.goto(server.url, { waitUntil: 'load' });
    // The primary probe uses exactly the required set from the release config, plus a
    // set of negative controls that must fail on the same device. The optional
    // --missing-format flag injects an impossible requirement so the gate can be
    // exercised against a capability gap: the release must block, not fall back.
    const requiredFormats = missingFormat
      ? [...config.render.requiredTextureFormats, missingFormat]
      : config.render.requiredTextureFormats;
    const capabilities = await probeCapabilities(page, config, {
      requiredFormats,
      negativeControls: {
        format: 'aiml-missing-format-probe',
        // rgba8unorm is a real renderable target but clips at 1.0, so it must fail the
        // "stores values above 1.0" test — proving that check actually discriminates.
        floatTarget: 'rgba8unorm',
        limitName: 'maxTextureDimension2D',
        limitValue: 1e9,
      },
    });
    evidence.capabilities = capabilities;
    const capabilityOk = capabilities.problems.length === 0;
    record('能力探测通过', capabilityOk, capabilityOk ? '全部必需格式与限制满足' : capabilities.problems.join('; '));
    if (!capabilityOk) {
      // Explicitly block release rather than falling through to a soft render.
      return finish(evidence, config, root, startedAt);
    }

    // Negative controls prove the probe can fail: an unavailable texture format, a
    // float target it cannot store, and an impossible limit must all be rejected.
    // Without this, "capabilities passed" could just mean the probe always passes.
    // A missing capability must block, never silently fall back to software.
    const controls = capabilities.negativeControls ?? [];
    const missingChecks = controls.map((c) => ({
      name: c.name,
      ok: c.ok === false,
      detail: c.ok ? '未拒绝，探测可能失效' : c.error ?? '已拒绝',
    }));
    if (missingChecks.length !== 3) {
      missingChecks.push({ name: '负向对照数量', ok: false, detail: `期望 3 项，实际 ${missingChecks.length}` });
    }
    evidence.missingCapabilityControls = missingChecks;
    const negativeOk = missingChecks.every((c) => c.ok);
    record('能力缺失被阻止而不是回退', negativeOk, missingChecks.map((c) => `${c.ok ? 'ok' : 'MISS'}:${c.name}`).join(' | '));
    if (!negativeOk) return finish(evidence, config, root, startedAt);

    const software = capabilities.adapter?.software === true;
    record(
      '适配器路径符合配置',
      config.render.allowSoftwareRasterizer || !software || allowSoftware,
      software
        ? `软件光栅器 (${capabilities.adapter?.architecture || capabilities.adapter?.vendor})${allowSoftware ? '，按 --allow-software 记录为非硬件验收' : ''}`
        : `硬件适配器 (${capabilities.adapter?.vendor}/${capabilities.adapter?.architecture})`,
      !config.render.allowSoftwareRasterizer && !allowSoftware,
    );
    evidence.environment.softwareRasterizer = software;

    // --- 4. Load the built package and let it compile ------------------------
    const ready = await waitForApp(page, 40000);
    record('应用冷启动完成', ready.ok, ready.detail);
    if (!ready.ok) { evidence.logs = tail(logs, 40); return finish(evidence, config, root, startedAt); }

    const compile = await page.evaluate(() => {
      const r = window.__aimlRelease?.renderer();
      return r ? r.shaders : null;
    });
    evidence.compile = compile;
    const compileOk = !!compile && compile.ready && compile.ok;
    record('着色器编译成功', compileOk, describeCompile(compile));
    if (!compileOk) { evidence.logs = tail(logs, 40); return finish(evidence, config, root, startedAt); }

    // --- 5. Kernel drives the sample flow ------------------------------------
    const kernelInfo = await page.evaluate(() => {
      const k = window.__aimlRelease?.kernel();
      return k ? {
        swatch: k.swatch,
        params: k.params,
        revision: k.revision,
        hasSpecimen: !!k.specimen,
        failure: k.failure,
        controls: {
          material: document.querySelectorAll('#material-controls input').length,
          normal: document.querySelectorAll('#normal-controls input').length,
          specimen: document.querySelectorAll('#specimen-controls input').length,
          swatches: document.querySelectorAll('#swatch-list .swatch').length,
        },
      } : null;
    });
    evidence.kernel = kernelInfo;
    record(
      '内核驱动材质控件',
      !!kernelInfo && kernelInfo.controls.material > 0 && kernelInfo.controls.normal > 0 && kernelInfo.controls.specimen > 0,
      kernelInfo ? `材质 ${kernelInfo.controls.material} / 法线 ${kernelInfo.controls.normal} / 样件 ${kernelInfo.controls.specimen} 个控件` : '内核快照缺失',
    );

    // Reject a bad parameter and confirm nothing was half-applied.
    //
    // Two different input surfaces are checked separately:
    //   1. The browser's <input type=range> itself clamps an impossible value like 5
    //      to its max (1). That is legitimate — the widget never emits out-of-range —
    //      and must land exactly on the max with no partial write.
    //   2. The kernel itself must reject a genuinely out-of-range command and leave
    //      the whole state untouched, reporting the failure back to the material UI.
    const reject = await page.evaluate(() => {
      const r = window.__aimlRelease;
      const before = r.kernel();
      // 1. Drive the roughness slider beyond its range through the same surface a user
      //    touches. The browser clamps to max=1; the kernel must accept exactly 1.
      const slider = [...document.querySelectorAll('#material-controls input[type=range]')]
        .find((i) => i.max === '1' && i.step === '0.005');
      let clamp = null;
      if (slider) {
        slider.value = '5';
        slider.dispatchEvent(new Event('input'));
        const afterSlider = r.kernel();
        clamp = {
          roughness: afterSlider.params.roughness,
          max: Number(slider.max),
          revisionGain: afterSlider.revision - before.revision,
        };
      }
      // 2. Push a value the kernel must refuse straight at the kernel, bypassing the
      //    clamped widget. Nothing may change and the failure must name the command.
      const beforeDirect = r.kernel();
      const directAccepted = window.__app.kernel.setSpecimenYaw(99);
      const afterDirect = r.kernel();
      const failedNote = document.querySelector('#specimen-controls .kernel-failure')?.textContent ?? '';
      return {
        clamp,
        beforeDirect: { yaw: beforeDirect.specimen.yaw, rev: beforeDirect.revision, rough: beforeDirect.params.roughness },
        directAccepted,
        afterDirect: { yaw: afterDirect.specimen.yaw, rev: afterDirect.revision, rough: afterDirect.params.roughness },
        failure: afterDirect.failure,
        failedNote,
        rendererAlive: !window.__aimlRelease.renderer()?.isLost,
      };
    });
    // The widget clamp is legitimate only if it landed exactly on the bound; the
    // direct command is legitimate only if it was refused with the state untouched.
    const clampOk = !reject.clamp || Math.abs(reject.clamp.roughness - reject.clamp.max) < 1e-9;
    const directOk = reject.directAccepted === false
      && reject.afterDirect.yaw === reject.beforeDirect.yaw
      && reject.afterDirect.rough === reject.beforeDirect.rough
      && reject.failure?.code === 'specimen-out-of-range';
    const noHalfApplied = clampOk && directOk && reject.failedNote.length > 0;
    record(
      '越界输入被整体拒绝而非半应用',
      config.gates.noHalfAppliedMaterial ? noHalfApplied : true,
      `滑杆钳制=${reject.clamp ? reject.clamp.roughness : 'n/a'}（合法=${clampOk}） 直接越界接受=${reject.directAccepted} 状态未变=${reject.afterDirect.yaw === reject.beforeDirect.yaw} failure=${reject.failure?.code ?? 'none'} 面板提示=${reject.failedNote ? '有' : '无'}`,
      config.gates.noHalfAppliedMaterial,
    );
    evidence.kernel.reject = reject;

    // --- 6. Replay the existing sample preset through the kernel --------------
    // The preset flow is exercised through the UI: click the built-in preset button.
    const presetApplied = await page.evaluate(() => {
      const buttons = [...document.querySelectorAll('#preset-controls .preset-item button')];
      const target = buttons.find((b) => b.textContent?.includes('演播室'));
      if (!target) return { ok: false, reason: '未找到内置预设按钮' };
      target.click();
      return { ok: true, clicked: target.textContent };
    });
    await page.waitForTimeout(400);
    const afterPreset = await page.evaluate(() => {
      const k = window.__aimlRelease.kernel();
      return { swatch: k.swatch, params: k.params, failure: k.failure };
    });
    record(
      '内置样例预设走完原流程',
      presetApplied.ok && afterPreset.swatch === config.fixtures.swatch && !afterPreset.failure,
      presetApplied.ok ? `样块 ${afterPreset.swatch} R=${afterPreset.params.roughness}` : presetApplied.reason,
    );

    // --- 7. Real frames -------------------------------------------------------
    // Playwright's element screenshot goes through the compositor, which returns a
    // blank/transparent canvas under headless SwiftShader even though the swapchain is
    // correct. The canvas's own readback (toDataURL) is the real presented image, so
    // frames are captured that way and the readback is checked to be a genuine PNG.
    const framesDir = resolvePath(config.artifacts.framesDir);
    fs.mkdirSync(framesDir, { recursive: true });
    await settle(page);

    const baselineShot = await captureCanvas(page);
    const baseline = summarizePng(baselineShot);
    evidence.frames.push({ id: 'baseline', file: saveFrame(framesDir, 'baseline', baselineShot), summary: baseline });
    const nonBlank = baseline.mean >= config.gates.minFrameMeanLuma && baseline.max > 32;
    record('真实帧非空白', config.gates.frameMustBeNonBlank ? nonBlank : true, `mean=${baseline.mean.toFixed(2)} max=${baseline.max} ${baseline.width}x${baseline.height}`, config.gates.frameMustBeNonBlank);
    if (!nonBlank && config.gates.frameMustBeNonBlank) return finish(evidence, config, root, startedAt);

    // Reference frame: the fixed material + camera must reproduce a committed frame
    // summary. On first run the reference is seeded from this real frame and the check
    // is a warning, because there is nothing to compare against yet; after that the
    // committed reference is authoritative and a drifted frame blocks.
    const reference = compareReference(baseline, config);
    evidence.reference = reference;
    record(
      '参考镜头帧一致',
      reference.ok,
      reference.detail,
      reference.existed,
    );

    // Roughness change through the kernel must alter real pixels.
    await page.evaluate((roughness) => {
      window.__app.kernel.setParam('roughness', roughness);
    }, config.fixtures.roughnessAlt);
    await settle(page);
    const roughShot = await captureCanvas(page);
    const rough = summarizePng(roughShot);
    const roughDiff = diffPng(baselineShot, roughShot);
    evidence.frames.push({ id: 'roughness', file: saveFrame(framesDir, 'roughness', roughShot), summary: rough, diff: roughDiff });
    record(
      '粗糙度调整改变真实帧',
      config.gates.materialChangeMustAlterFrame ? roughDiff.changedRatio >= config.gates.minChangedPixelRatio : true,
      `roughness ${config.fixtures.roughnessAlt} changed=${(roughDiff.changedRatio * 100).toFixed(2)}% meanΔ=${roughDiff.meanDelta.toFixed(2)}`,
      config.gates.materialChangeMustAlterFrame,
    );

    // Specimen rotation through the kernel must alter real pixels.
    await page.evaluate((yaw) => {
      window.__app.kernel.setSpecimenYaw(yaw);
    }, config.fixtures.rotatedYaw);
    await settle(page);
    const rotatedShot = await captureCanvas(page);
    const rotated = summarizePng(rotatedShot);
    const rotDiff = diffPng(roughShot, rotatedShot);
    evidence.frames.push({ id: 'rotated', file: saveFrame(framesDir, 'rotated', rotatedShot), summary: rotated, diff: rotDiff });
    record(
      '样件旋转改变真实帧',
      config.gates.rotationMustAlterFrame ? rotDiff.changedRatio >= config.gates.minChangedPixelRatio : true,
      `changed=${(rotDiff.changedRatio * 100).toFixed(2)}% meanΔ=${rotDiff.meanDelta.toFixed(2)}`,
      config.gates.rotationMustAlterFrame,
    );

    // --- 8. Benchmark parity ---------------------------------------------------
    const bench = await page.evaluate(async () => {
      const out = [];
      for (const id of ['diffuse', 'flat-normal', 'roughness', 'boundaries']) {
        const r = await window.__app.runBenchScene(id);
        out.push({ id, maxAbsError: r.maxAbsError, samples: r.sampleErrors.map((e) => ({ label: e.label, gpu: e.gpuRgb, cpu: e.cpuRgb, d: e.maxAbsError })) });
      }
      return out;
    });
    evidence.bench = bench;
    for (const b of bench) {
      record(`基准 ${b.id} 与 CPU 参考一致`, b.maxAbsError <= config.gates.benchMaxAbsError, `maxErr=${b.maxAbsError}`, true);
    }

    evidence.logs = tail(logs, 40);
    evidence.compileStable = await page.evaluate(() => {
      const r = window.__aimlRelease.renderer();
      return r ? { ok: r.shaders.ok, ready: r.shaders.ready } : null;
    });
  } catch (error) {
    // A browser crash (OOM, GPU-process death) is itself a blocking result. Record it
    // so the evidence file still says "blocked" instead of vanishing into a stack trace.
    record(
      '门禁执行未中断',
      false,
      `浏览器会话中断：${error instanceof Error ? error.message.split('\n')[0] : String(error)}`,
    );
    evidence.logs = tail(logs, 60);
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (!keepServer) await server.close();
  }

  return finish(evidence, config, root, startedAt);
}

async function waitForApp(page, timeout) {
  try {
    await page.waitForFunction(() => {
      const r = window.__aimlRelease?.renderer();
      return !!r && !r.isLost && r.size.width > 0 && r.shaders?.ready === true;
    }, null, { timeout });
    return { ok: true, detail: '设备、编译状态与画布尺寸就绪' };
  } catch (error) {
    const diag = await page.evaluate(() => ({
      hasRelease: !!window.__aimlRelease,
      hasRenderer: !!window.__aimlRelease?.renderer(),
      lost: window.__aimlRelease?.renderer()?.isLost,
      shaders: window.__aimlRelease?.renderer()?.shaders ?? null,
      overlay: document.getElementById('viewport-overlay')?.textContent?.slice(0, 300),
    })).catch((e) => ({ evalError: String(e) }));
    return { ok: false, detail: `等待就绪超时：${error instanceof Error ? error.message : String(error)} ${JSON.stringify(diag)}` };
  }
}

// Let the render loop present at least two frames after a state change.
async function settle(page) {
  await page.evaluate(() => new Promise((resolve) => {
    let frames = 0;
    const tick = () => { frames++; if (frames >= 3) resolve(); else requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }));
  await page.waitForTimeout(120);
}

// Capture the canvas exactly as presented. `toDataURL` reads the swapchain through
// the WebGPU canvas's own snapshot path; the headless compositor screenshot does not.
async function captureCanvas(page) {
  let dataUrl;
  try {
    dataUrl = await page.evaluate(() => document.getElementById('viewport').toDataURL('image/png'));
  } catch (error) {
    throw new ReleaseBlocked(`画布回读时浏览器不可用：${error instanceof Error ? error.message : String(error)}`);
  }
  if (typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image/png;base64,')) {
    throw new ReleaseBlocked(`画布回读失败：${String(dataUrl).slice(0, 40)}`);
  }
  return Buffer.from(dataUrl.slice('data:image/png;base64,'.length), 'base64');
}

function describeCompile(compile) {
  if (!compile) return '未取得编译状态（渲染器不可用）';
  const modules = compile.modules.map((m) => `${m.label}:${m.ok ? 'ok' : 'FAIL'}`).join(' ');
  const pipes = compile.pipelines.map((p) => `${p.label}:${p.ok ? 'ok' : `FAIL(${p.error ?? ''})`}`).join(' ');
  return `modules[${modules}] pipelines[${pipes}]${compile.unsupported ? ' (getCompilationInfo 不受支持)' : ''}`;
}

// Compares the fixture frame against the committed reference summary. The reference is
// a summary (dimensions, mean, max, opaque ratio) rather than a raw PNG so it is stable
// across minor byte-level differences while still catching a drifted camera/material.
function compareReference(summary, config) {
  const file = resolvePath(config.fixtures.referenceFrame);
  const current = {
    width: summary.width,
    height: summary.height,
    mean: round2(summary.mean),
    max: round2(summary.max),
    opaqueRatio: round4(summary.opaqueRatio ?? 1),
  };
  if (!fs.existsSync(file)) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify({
      schema: 'aiml.reference.frame/v1',
      note: '首轮门禁从真实帧生成；此后该文件的差异会阻断发布。',
      fixture: { swatch: config.fixtures.swatch, roughness: config.fixtures.roughness, camera: config.fixtures.camera },
      frame: current,
    }, null, 2));
    return { ok: true, existed: false, detail: `首次生成参考帧 ${path.relative(repoRoot(), file)}` };
  }
  let ref;
  try {
    ref = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    return { ok: false, existed: true, detail: `参考帧文件无法解析：${error instanceof Error ? error.message : String(error)}` };
  }
  const expected = ref.frame ?? {};
  const meanTol = Math.max(2, expected.mean * 0.12);
  const maxTol = Math.max(6, expected.max * 0.12);
  const sizeOk = expected.width === current.width && expected.height === current.height;
  const meanOk = Math.abs(expected.mean - current.mean) <= meanTol;
  const maxOk = Math.abs(expected.max - current.max) <= maxTol;
  const ok = sizeOk && meanOk && maxOk;
  return {
    ok,
    existed: true,
    current,
    expected,
    detail: ok
      ? `帧统计与参考一致 (mean ${current.mean} vs ${expected.mean}, max ${current.max} vs ${expected.max})`
      : `帧统计偏离参考：尺寸 ${sizeOk ? 'ok' : `${current.width}x${current.height} vs ${expected.width}x${expected.height}`}, mean ${current.mean} vs ${expected.mean} (±${meanTol.toFixed(2)}), max ${current.max} vs ${expected.max} (±${maxTol.toFixed(2)})`,
  };
}

function round2(x) { return Math.round(x * 100) / 100; }
function round4(x) { return Math.round(x * 10000) / 10000; }

function saveFrame(dir, id, buffer) {
  const file = path.join(dir, `${id}.png`);
  fs.writeFileSync(file, buffer);
  return path.relative(repoRoot(), file);
}

function tail(text, lines) {
  if (!text) return '';
  const all = String(text).trimEnd().split('\n');
  return all.slice(-lines).join('\n');
}

function largestEntry(dir) {
  if (!fs.existsSync(dir)) return null;
  const entries = fs.readdirSync(dir).filter((f) => f.endsWith('.js')).map((f) => {
    const file = path.join(dir, f);
    return { file, size: fs.statSync(file).size };
  });
  entries.sort((a, b) => b.size - a.size);
  return entries[0] ?? null;
}

function gitStatus(root) {
  const res = spawnSync('git', ['status', '--porcelain', '--', 'src', 'release', 'index.html', 'package.json', 'scripts', 'tsconfig.json'], {
    cwd: root, encoding: 'utf8',
  });
  const lines = (res.stdout ?? '').split('\n').map((l) => l.trim()).filter(Boolean);
  return { dirtyBuildInputs: lines };
}

function finish(evidence, config, root, startedAt) {
  const failed = checks.filter((c) => !c.pass && c.blocking);
  const warnings = checks.filter((c) => !c.pass && !c.blocking);
  evidence.finishedAt = new Date().toISOString();
  evidence.durationMs = Date.parse(evidence.finishedAt) - Date.parse(startedAt);
  evidence.checks = checks;

  // Hardware-path accounting. A software rasteriser may satisfy the functional gates only
  // when the operator explicitly allowed it, and the verdict must say plainly that the
  // hardware path was NOT verified. It can never be reported as "verified".
  const software = evidence.environment?.softwareRasterizer === true;
  evidence.hardwarePath = software ? 'not-verified' : failed.length === 0 ? 'verified' : 'blocked';
  if (failed.length > 0) {
    evidence.verdict = 'blocked';
  } else if (software) {
    evidence.verdict = config.render.allowSoftwareRasterizer || allowSoftware
      ? 'verified-software-fallback'
      : 'blocked';
  } else {
    evidence.verdict = 'verified';
  }

  evidence.blockingFailures = failed.map((c) => ({ name: c.name, detail: c.detail }));
  evidence.warnings = warnings.map((c) => ({ name: c.name, detail: c.detail }));

  const evidenceFile = resolvePath(config.artifacts.evidenceFile);
  fs.mkdirSync(path.dirname(evidenceFile), { recursive: true });
  fs.writeFileSync(evidenceFile, JSON.stringify(evidence, null, 2));

  console.log(`\n证据已写入 ${path.relative(root, evidenceFile)}`);
  if (evidence.verdict === 'verified') {
    console.log(`发布核验通过：${checks.length - warnings.length - failed.length}/${checks.length} 项；硬件路径已验证`);
    return 0;
  }
  if (evidence.verdict === 'verified-software-fallback') {
    console.log(`功能核验通过（软件光栅器），但硬件路径未验证：${evidence.environment.softwareRasterizer ? 'SwiftShader/软件适配器' : '软件回退'}`);
    console.log('该运行不能作为已验证硬件路径的发布依据；请在具备真实 GPU 的环境重跑门禁。');
    return 0;
  }
  console.log(`发布被阻止：${failed.length} 项阻断失败`);
  for (const f of failed) console.log(`  - ${f.name}: ${f.detail}`);
  return 1;
}

main()
  .then((code) => { process.exitCode = code ?? 1; })
  .catch((error) => {
    console.error('发布门禁异常：', error instanceof ReleaseBlocked ? error.message : error);
    process.exitCode = 2;
  });
