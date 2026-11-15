// Capability probe. Runs inside the page against a real adapter and device before any
// frame is trusted. A missing required texture format or float render target is a
// hard release block: the software fallback path may still draw a picture, but that
// picture must never be reported as a verified hardware path.

export async function probeCapabilities(page, config, override = {}) {
  return page.evaluate(async ({ requiredFormats, requiredFloatTargets, requiredFeature, requiredLimits, negativeControls, backend }) => {
    function decodeHalf(h) {
      const sign = (h >> 15) & 1;
      const exp = (h >> 10) & 0x1f;
      const frac = h & 0x3ff;
      if (exp === 0) return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
      if (exp === 31) return frac ? NaN : (sign ? -Infinity : Infinity);
      return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }

    function fullscreenModule(device, fragExpr) {
      return device.createShaderModule({
        code: `
          @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(p[i], 0.0, 1.0);
          }
          @fragment fn fs() -> @location(0) vec4f { return ${fragExpr}; }`,
      });
    }

    async function renderAndRead(device, format) {
      const texture = device.createTexture({
        size: [4, 4],
        format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const bytesPerRow = 256;
      const buffer = device.createBuffer({ size: bytesPerRow * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      return { texture, buffer, bytesPerRow };
    }

    async function trialFormat(device, format) {
      const depth = /depth/.test(format);
      let target;
      try {
        if (depth) {
          // Depth formats cannot be colour attachments. Prove the format can be bound
          // as a depth attachment, which is where the scene pass actually uses it.
          const texture = device.createTexture({ size: [4, 4], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
          const module = fullscreenModule(device, 'vec4f(1.0)');
          const pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: 'vs' },
            fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
            depthStencil: { format, depthWriteEnabled: true, depthCompare: 'less' },
          });
          const color = device.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: color.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
            depthStencilAttachment: {
              view: texture.createView(),
              depthLoadOp: 'clear',
              depthStoreOp: 'store',
              depthClearValue: 1,
            },
          });
          pass.setPipeline(pipeline);
          pass.draw(3);
          pass.end();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          texture.destroy();
          color.destroy();
          return { ok: true, usage: 'depth-attachment' };
        }

        target = await renderAndRead(device, format);
        const module = fullscreenModule(device, 'vec4f(0.25, 0.5, 0.75, 1.0)');
        const pipeline = device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format }] },
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: target.texture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
        });
        pass.setPipeline(pipeline);
        pass.draw(3);
        pass.end();
        encoder.copyTextureToBuffer({ texture: target.texture }, { buffer: target.buffer, bytesPerRow: target.bytesPerRow }, [4, 4, 1]);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await target.buffer.mapAsync(GPUMapMode.READ);
        const bytes = new Uint8Array(target.buffer.getMappedRange());
        const nonZero = bytes.some((b) => b !== 0);
        target.buffer.unmap();
        if (!nonZero) return { ok: false, error: '渲染结果为全零，格式可能被静默拒绝' };
        return { ok: true, usage: 'color-attachment' };
      } catch (error) {
        return { ok: false, error: error instanceof Error ? error.message : String(error) };
      } finally {
        if (target) { try { target.buffer.destroy(); target.texture.destroy(); } catch { /* best effort */ } }
      }
    }

    async function trialFloat(device, format) {
      let target;
      try {
        target = await renderAndRead(device, format);
        // Values above 1.0 prove the target is genuinely floating point and not an
        // 8-bit UNORM target that clips.
        const module = fullscreenModule(device, 'vec4f(2.5, 1.0, 0.5, 1.0)');
        const pipeline = device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format }] },
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: target.texture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
        });
        pass.setPipeline(pipeline);
        pass.draw(3);
        pass.end();
        encoder.copyTextureToBuffer({ texture: target.texture }, { buffer: target.buffer, bytesPerRow: target.bytesPerRow }, [4, 4, 1]);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await target.buffer.mapAsync(GPUMapMode.READ);
        const raw = new Uint8Array(target.buffer.getMappedRange());
        const half = raw[0] | (raw[1] << 8);
        const decoded = decodeHalf(half);
        target.buffer.unmap();
        if (!(decoded > 2.0)) return { ok: false, error: `写入 2.5 读到 ${decoded}，目标不是可存储的浮点范围` };
        return { ok: true, value: decoded };
      } catch (error) {
        return { ok: false, error: error instanceof Error ? error.message : String(error) };
      } finally {
        if (target) { try { target.buffer.destroy(); target.texture.destroy(); } catch { /* best effort */ } }
      }
    }

    const out = {
      checkedAt: new Date().toISOString(),
      hasGpu: !!navigator.gpu,
      backendRequested: backend,
      adapter: null,
      deviceLimits: null,
      features: [],
      formats: {},
      floatTargets: {},
      problems: [],
    };
    if (!navigator.gpu) {
      out.problems.push('navigator.gpu 不存在：此浏览器未暴露 WebGPU');
      return out;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      out.problems.push('requestAdapter 返回 null：没有可用适配器');
      return out;
    }
    const info = adapter.info ?? {};
    out.adapter = {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      description: info.description ?? '',
      device: info.device ?? '',
      isFallbackAdapter: adapter.isFallbackAdapter ?? false,
      software: /swiftshader|llvmpipe|lavapipe|softpipe|software/i.test(
        `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`,
      ),
    };
    out.features = Array.from(adapter.features);
    if (requiredFeature && !adapter.features.has(requiredFeature)) {
      out.problems.push(`适配器缺少必需特性 ${requiredFeature}`);
    }

    let device;
    try {
      device = await adapter.requestDevice();
    } catch (error) {
      out.problems.push(`requestDevice 失败：${error instanceof Error ? error.message : String(error)}`);
      return out;
    }
    out.deviceLimits = {
      maxTextureDimension2D: device.limits.maxTextureDimension2D,
      maxBufferSize: device.limits.maxBufferSize,
    };
    for (const [limit, min] of Object.entries(requiredLimits)) {
      const have = device.limits[limit];
      if (typeof have !== 'number') {
        out.problems.push(`设备未报告限制 ${limit}`);
      } else if (have < min) {
        out.problems.push(`设备限制 ${limit}=${have} 低于要求的 ${min}`);
      }
    }

    for (const format of requiredFormats) {
      out.formats[format] = await trialFormat(device, format);
      if (!out.formats[format].ok) out.problems.push(`纹理格式 ${format} 不可用：${out.formats[format].error}`);
    }
    for (const format of requiredFloatTargets) {
      out.floatTargets[format] = await trialFloat(device, format);
      if (!out.floatTargets[format].ok) out.problems.push(`浮点渲染目标 ${format} 不可用：${out.floatTargets[format].error}`);
    }

    // Negative controls run on the same device: they must produce failure problems,
    // otherwise "passed" would only mean the probe never fails. One device keeps the
    // probe from exhausting memory on a software rasteriser.
    if (negativeControls) {
      const controls = [];
      const fmt = await trialFormat(device, negativeControls.format);
      controls.push({ name: `纹理格式 ${negativeControls.format}`, ok: fmt.ok, error: fmt.error ?? null });
      const flt = await trialFloat(device, negativeControls.floatTarget);
      controls.push({ name: `浮点目标 ${negativeControls.floatTarget}`, ok: flt.ok, error: flt.error ?? null });
      const limit = negativeControls.limitName;
      const have = device.limits[limit];
      controls.push({
        name: `限制 ${limit}=${negativeControls.limitValue}`,
        ok: typeof have === 'number' && have >= negativeControls.limitValue,
        error: typeof have === 'number' ? `设备报告 ${have}` : `设备未报告 ${limit}`,
      });
      out.negativeControls = controls;
    }

    device.destroy();
    return out;
  }, {
    requiredFormats: override.requiredFormats ?? config.render.requiredTextureFormats,
    requiredFloatTargets: override.requiredFloatTargets ?? config.render.requiredFloatTargets,
    requiredFeature: override.requiredFeature ?? config.render.requiredAdapterFeature,
    requiredLimits: override.requiredLimits ?? config.render.requiredLimits,
    negativeControls: override.negativeControls ?? null,
    backend: config.render.backend,
  });
}
