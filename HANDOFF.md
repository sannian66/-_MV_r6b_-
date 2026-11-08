# 交接文档 · 《鬼屠·第三季》主题曲《谁定正邪》代码 MV

> 任何 agent 接手前请先读本文件，再读 docs/PLAN.md 和 docs/NOTES.md。

## 1. 用户需求（原话要点）
- 用代码逐帧做 MV。**不允许生视频**；场景、特效、字体、运镜都不生图。
  - 唯一例外（用户后补）：可以以 `ref/` 里的设定图为参考生成人物姿态立绘，必须锁定五官和发型，服装可以换。
- 希望观众看完记住：「卧槽这后期技术真牛逼」。参考片：B 站 BV16MJY6yEKq（Claude 做的千本樱 MV）。
- 规格：1920×1080，60fps，时长跟歌（241.344 秒），用上传的音乐，不要旁白。
- 署名必须设计进画面，不能做成普通字幕：
  - 出品人：西几王传媒
  - AI画师：沉溺、雪晴、索恩
  - 编剧/导演：苏打
  - 美术设计：Cait
  - 音乐：西几王
  - 标题：鬼屠 · 第三季 主题曲《谁定正邪》
- 用户已拍板的选项：
  - 1A：用角色图做 2.5D
  - 2A：歌词全程上屏，并作为画面元素来设计
  - 3A：署名嵌进前奏，片尾用印章收尾
  - 4A：避开常见坑（赛博霓虹、廉价金光、过度血腥、字幕条）
  - 5B：没有现成视觉资产，标题字和配色由我们设计
  - 6A：Three.js + GLSL，不用 Blender
- 流程规则（必须遵守）：
  1. 先给 2 个方向、分镜表和关键帧；**用户选定方向、确认分镜之前，不写正式代码**。
  2. 每个镜头先渲 3 张静帧自查。
  3. 渲全片前，渲 9 个关键时刻拼成九宫格自评 6 项，每项都要 ≥8 分。
  4. 出 15 秒样片，用户点头后再渲全片。
- 交付：mp4、抽帧总览图、创作说明、NOTES.md。
- 要实时同步本仓库，用户额度用尽前必须推送。

## 2. 当前进度（2026-11-08 23:10 CST）
- [x] 音频分析：129.2 BPM，节拍存于 `audio.json`；Whisper 逐句时间存于 `segs.json`（识别文字有错字，以 `source/lyrics.txt` 为准）
- [x] 抠图脚本 `cut.py`（白底设定图 → 透明 PNG，已处理发丝边）；已出 `assets/guitu_face.png`、`assets/guitu_body.png`
- [x] 两个方向的方案和分镜：`docs/PLAN.md`
- [x] 方向关键帧草图（Canvas 2D 粗做）：`kf/A_key.png`、`kf/B_key.png`，源码 `kf/a.html`、`kf/b.html`
- [ ] **等用户选方向、确认分镜**（推荐 A「墨刃」，#12 借用 B 的铁壁）
- [ ] 生成人物姿态立绘（选定方向后批量出）
- [ ] 正式工程：`src/` 下的 Three.js 场景、`window.draw(t)` 和逐镜头模块
- [ ] 歌词逐句校准到帧
- [ ] 音效合成和混音限幅
- [ ] 九宫格自评 → 15 秒样片 → 全片

## 3. 目录
- `song.mp3`：原曲
- `source/lyrics.txt`：歌词原文
- `ref/`：9 张角色设定图（白底，左侧大头像，右侧三视图）
- `assets/`：抠好的透明 PNG
- `fonts/`：字体，均为 OFL 可商用。马善政、志莽行、刘建毛草、思源宋体 / 黑体（可变字重）
- `audio.json`、`segs.json`：节拍和歌词时间
- `kf/`：关键帧原型。`common.js` 是通用工具（带种子的随机数、fbm、辉光、颗粒、暗角）；`render.mjs` 渲染单帧
- `docs/`：PLAN.md、NOTES.md

## 4. 环境（沙箱是 Debian 13 arm64，2 核 CPU，7 GB 内存）
- Chromium：`sudo apt-get install chromium`（Playwright 自带的 arm64 浏览器下载会返回 403，不能用）
- Node：`npm i` 安装 playwright 和 three。浏览器启动时 `executablePath:'/usr/bin/chromium'`，参数加 `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --allow-file-access-from-files`
- Python：librosa、numpy、scipy、PIL、openai-whisper
- 渲单帧：`node kf/render.mjs kf/a.html 59.4 out.png`
- 实测：简单全屏着色器每帧 0.4–0.7 秒（用 toDataURL 读回比 screenshot 快）

## 5. 关键决定
- 画面只由 t 决定：随机数全部带种子，不用 requestAnimationFrame，帧和帧之间不保存状态，方便断点续渲。
- 辉光的做法：亮部降采样 1/4，模糊后用 screen 叠回，强度封顶，避免把亮部烧成一片白。
- 「谁出现在哪句」是创作安排，不是剧情事实（见 NOTES）。

## 6. 下一步给接手者
1. 读用户最新回复，确认方向和分镜修改意见。
2. 选定方向后，列出需要的人物姿态清单（每个角色 1–3 张），用设定图作参考生成；生成结果放 `assets/poses/`。
3. 搭 `src/index.html` + `src/shots/*.js`：每个镜头一个模块，导出 `{start, end, draw(t, ctx|gl)}`。
4. 写 `render/render.mjs`：逐帧存盘到 `frames/%06d.jpg`，已存在的帧跳过，用于断点续渲；再写 `render/mux.sh` 调 ffmpeg。
5. 每完成一步就 commit + push。

## 7. 2026-11-08 23:20 状态：用户叫停
- 用户看过 A/B 两个关键帧后的反馈：「好丑」，并且做得「不如项目里那个子 agent 做的」。用户要求回到**纯代码**：不用生图，不用人物立绘。
- 项目「谁定正邪 MV」里的另一个 agent 已经做过纯代码关键帧，作为参考底线：
  - 关键帧：`/workspace/work/mv/sfp/p1–p5`
  - 风格：2.39 宽银幕，只用黑白红金，剪影，几何刀光，竖排书法
  - 时间轴：用 demucs 分离人声后做强制对齐，结果在 `/workspace/work/mv/align.json`，比本仓库的 `segs.json` 准，应改用它
- 随后用户说「停止吧」。项目暂停。不要继续，等用户重新发起。
