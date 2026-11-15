// Starts the built material lab from dist/ on a fixed localhost origin.
//
// WebGPU is only exposed in a secure context, and headless/CI hosts treat
// http://127.0.0.1 as secure, so the interface must be opened through this server
// rather than a file:// URL. The same server the release gate uses is reused here so
// what an operator opens is exactly what was verified.
//
//   npm run release:start            # serves dist/ and prints the URL
//   PORT=4173 npm run release:start
//
// Open the printed URL in Chrome/Chromium with WebGPU enabled to get the material
// preview window: swatch list, material roughness, and specimen rotation all drive
// the interior shader through the MaterialKernel.

import path from 'node:path';
import fs from 'node:fs';
import { loadReleaseConfig, resolvePath, repoRoot } from './lib/config.mjs';
import { startStaticServer } from './lib/server.mjs';

const config = loadReleaseConfig();
const distDir = resolvePath(config.build.outputDir);

if (!fs.existsSync(path.join(distDir, config.app.entry))) {
  console.error(`未找到发布产物：${path.relative(repoRoot(), distDir)}/${config.app.entry}`);
  console.error('请先运行 `npm run build`，或执行 `npm run release:check` 生成可核验的发布包。');
  process.exit(1);
}

const port = Number(process.env.PORT ?? 4173);
const server = await startStaticServer(distDir, { port });
const localUrl = server.url;

console.log(`${config.app.name} 已启动`);
console.log(`  入口：${localUrl}`);
console.log('  在 Chrome/Chromium 中打开上面的地址即可看到材质预览窗口。');
console.log('  需要 WebGPU：若浏览器未启用，请用 --enable-unsafe-webgpu 启动，或使用 release 门禁同款启动参数。');
console.log('  操作：左侧选择样块与调整 roughness，右侧旋转相机；「样件姿态」旋转样件本身。');
console.log('  按 Ctrl+C 停止。');

const shutdown = async () => {
  await server.close();
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
