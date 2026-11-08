# CLAUDE.md — 《谁定正邪》MV（鬼屠 · 第三季主题曲）

接手先读 `HANDOFF.md`（权威交接文档）和 `crew/CONVERSATION.md`（沟通记录）。

## 生图 / 生视频 / 音频：一律调用 Nano Banana skill
- 规则全文：`skills/nano-banana-skill/SKILL.md`（必须按其中的报价、上传、审核、轮询规则执行）。客户端：`tools/nb.py`。
- **生图默认模型 `RH_image2`（全能图片G2），`size:"4K"`**：这是可用的 GPT Image 2 系列里 4K 最便宜的（2026-11-09 报价 3 积分/张；官渠 `RH_image2_official` 18 积分）。提交前先 `quote` 确认价格。
- Key：环境变量 `NANO_BANANA_KEY` 或根目录 `.env.local`（已 gitignore）。**仓库公开，Key 绝不能提交**；`crew/export_chat.py` 会自动给对话里的 Key 打码。

## 角色一致性
- 角色以用户提供的资产图为准（`assets/characters/`），作为参考图传入 `input`。
- **五官、发型必须一致**；服装、穿搭、道具、姿势可以改。
- 每次生成记一行到 `crew/genlog.md`：时间、镜号、模型、尺寸、积分、task_id、result_urls、落盘路径、提示词。

## 规则变更（2026-11-09 03:36，用户）
- 原「纯代码、禁止 AI 生图」改为：**人物允许并优先用 AI 生图**（代码建模效果不达预期）。场景、字幕、转场、卡点、相机仍沿用代码管线（scenes.py / lib / units），生成的人物图作为图层合成进去。
- 视觉语言不变：2.39 黑边；黑白灰 + 血红 + 金；思源宋体字幕；红色「鬼屠」印章。

## 同步
- 改完运行 `./autosync.sh`（导出对话 → 提交 → 推送，跳过 >50MB 文件）。
