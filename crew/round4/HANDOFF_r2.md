
## C 组（第二棒，09:25–10:00 CST）
- **做了什么**：重下 blade_hot / blade_level（未重生成，0 积分）→ assets/gen/C/ 抠图；41 镜逐镜跑通（修 C01 透视 dtype、C18 Army.draw 返回 (fr,info)，heat_haze / GiantSeal / CharWall 接口实测可用）；单帧 0.3–2.8s。
- 新建 units/c_props.py（只给 C 组用，不改共享库）：墨玉巨印 StoneSeal（透视立方体贴石纹、朱砂阳文底面、金填阴刻、可斩开+碎石）、漆金暗箱 LacquerBox、青铜天平 BronzeBalance（圆柱着色+铜绿+碗形秤盘+斩断）、通用 Cuboid（C23 立体字砖、C10 立体铅活字）。
- 重做：C03 纯代码手影+十字提线板（S062 降到 2 次：C02、C31）；C04/C40/C41 天平；C05/C06 暗箱；C10、C13（风暴之眼漩涡+光速雨线）、C19（甲士轮廓光剪影+切口）、C23、C27、C28（「亮」墨边金芯可读）、C32–C35 巨印、C36（跪伏方阵，呼应 C18 冲锋）。转场去白闪化（C14/C27 smoke_through、C20 ink_bleed、C36 flash_red 1 帧）。
- **积分**：本棒 0；本组累计 24 / 80。
- **联系表**：out/round4/C_contact_v1.jpg（跑通后原样）、out/round4/C_contact_v2.jpg（道具重做后）、out/round4/C_keys_v1.jpg（40 个切点帧 + 23 个歌词关键帧，转场改动前渲；改后 4 处单独抽查过）。脚本 tools/c_contact.py。分镜 crew/round4/C_storyboard.md。
- **自评弱点**：C18/C19/C36 程序化甲士仍可能被评代码味；C41 巨型秤盘口沿偏薄像环；C32 奔跑小人小且糊；C02 提线挂字偏空、手被画顶裁；全程只看了静帧抽样，没出动态预览。
- **下一步**：出 C_contact_v3 / C_keys_v2（含转场改动）；C02 加密构图；C41 秤盘加厚、加盘内侧纹样；出 1–2 段 mp4 预览验证镜内运动与卡点；送评审。
