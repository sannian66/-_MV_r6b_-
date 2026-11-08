# CLAUDE.md — 《谁定正邪》MV（鬼屠 · 第三季主题曲）

接手先读 `HANDOFF.md`（权威交接文档）和 `crew/CONVERSATION.md`（沟通记录）。

## 规则（2026-11-09 04:02 用户决定：回到纯代码）
- **画面纯代码渲染，成片里不得出现任何 AI 生成或外部图片的像素。** 人物也由 `lib/rig.py` 程序化绘制。
- 03:36 的「人物改用 AI 生图」已撤销；**Seedance 重绘方案否决**。
- 角色资产图（`assets/characters/`）和 `out/gen/` 下的 AI 图**只做测量 / 取色 / 描轮廓参考**；描出的轮廓写成代码里的矢量数据（如 `lib/ref/guitu_hero_ref.py`），渲染时不得读入图片。
- 角色一致性：五官、发型以资产图为准；服装、道具、姿势可改。
- Nano Banana skill（`skills/nano-banana-skill/SKILL.md`、`tools/nb.py`）保留，但产物不得进画面；Key 绝不能提交（仓库公开）。
- 视觉语言不变：2.39 黑边；黑白灰 + 血红 + 金；思源宋体字幕；红色「鬼屠」印章。

## 同步
- 改完运行 `./autosync.sh`（导出对话 → 提交 → 推送，跳过 >50MB 文件）。
