# 内饰材质发布与启动

本目录存放材质台的发布配置（`release.config.json`）和启动/核验脚本说明。发布配置里的每一项都是必填：
缺项或格式错误会直接报错，不会回退到默认值并在证据里伪装成通过。

## 界面怎样启动

材质预览窗口不是靠脚本“跑完即成功”来证明的，必须真的把界面打开、在窗口里操作样件并看到内饰反射变化。
启动分两步：

```bash
npm install          # 首次
npm run build        # 产出 dist/，同时执行 tsc --noEmit
npm run release:start
```

`release:start` 会用内置静态服务器打开 **http://127.0.0.1:4173/**，把打印出的地址在
Chrome/Chromium 里打开，即可得到材质预览窗口：

- 左侧「样块 / 材质参数」：选择皮革、翻毛皮、金属等样块，拖动 `roughness`、`metallic`、`uvScale`；
- 左侧「样件姿态」：旋转当前样件本身（yaw/pitch），场景其余网格保持不动；
- 右侧「相机 / 灯光 / 曝光 / 预设」：调整参考镜头，或点「演播室 · 黑色打孔皮革」重放内置样例；
- 底部状态栏显示帧耗时、显存占用与适配器；设备丢失会显示故障覆盖层而不是旧截图。

> WebGPU 只在安全上下文可用。`http://127.0.0.1` 被视为安全上下文，所以必须通过上面的本地服务器打开，
> 不能直接双击 `dist/index.html`（`file://`）。若浏览器未启用 WebGPU，用
> `chromium --enable-unsafe-webgpu` 启动。无 GPU 的干净环境可用 SwiftShader 软件适配器，
> 但软件回退 **不会** 被记为已验证的硬件路径（见下）。

## 发布核验

```bash
npm run release:check                 # 构建 + 全量门禁
npm run release:check -- --skip-build # 复用现有 dist/
npm run release:check -- --allow-software        # 允许软件光栅器，但证据标记为“非硬件验收”
npm run release:check -- --missing-format=xxx    # 注入缺失格式，必须阻断发布
```

门禁顺序与阻断规则：

1. `npm run build` 可复现构建成功，产物含入口、不含源码目录。
2. 找到 Chromium 与 Vulkan ICD（`AIML_CHROME` / `AIML_SYSROOT` / Playwright 缓存）。
3. 能力探测：必需的纹理格式、浮点渲染目标、特性与限制逐项实测。缺任何一项立即阻断，
   不会退到软件路径假装通过。另有负向对照，证明探测真的会因缺失而失败。
4. 冷启动后读取 `getCompilationInfo()` 与各管线创建结果，着色器必须编译通过。
5. 内核必须驱动材质/法线/样件控件；越界输入整体拒绝、状态不变、面板给出失败提示。
6. 重放内置样例预设，确认走到原有流程。
7. 用固定材质与参考镜头生成真实帧；调整 roughness、旋转样件都必须改变真实像素；
   并与 `artifacts/reference/interior-leather-reference.json` 的参考帧统计比对。
8. 离屏基准与 CPU 参考一致（`≤2`）。

结论写入 `artifacts/evidence/release-evidence.json`：

- `verdict: verified` / `hardwarePath: verified`：真实硬件路径通过；
- `verdict: verified-software-fallback` / `hardwarePath: not-verified`：只在显式 `--allow-software`
  时出现，说明功能通过但**硬件路径未验证**，不能作为已验证硬件路径的发布依据；
- `verdict: blocked`：存在阻断项，不得发布。

## 干净环境准备

- Chromium：`node node_modules/playwright-core/cli.js install chromium`，或设 `AIML_CHROME`；
- 缺少系统库时，把 `.deb` 解包到独立空目录，用 `AIML_SYSROOT` 指向它（脚本会把
  `usr/lib/<arch>` 与 `lib/<arch>` 加入 `LD_LIBRARY_PATH`）；
- Vulkan ICD：默认用 Chromium 自带的 `vk_swiftshader_icd.json`，或设 `AIML_VK_ICD`。

启动参数与环境变量全部来自 `release.config.json` 的 `launch` 段，不靠开发者本机缓存。
