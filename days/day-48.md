# Day 48 · 开源与交付

> 预计 3–4h ｜ 📓 `notebooks/day-48_open_source_ship.ipynb` ｜ 💻 本地 · `README.md`、`reports/`、`assets/`
> 前置：Day 47

## 今日目标（一句话）

重写 README（含 demo GIF）、清理代码、写贡献指南、上传模型与数据、录 demo 视频；最终交付：**完整开源仓库 + demo 视频 + 技术报告 + 8 周复盘**。

## 一、读（60 min）

材料：
- （交付日）
- 从头读一遍你自己的 README —— 假装你是个陌生人

思考题（先自己想，答案在讲义或代码注释里）：
1. 一个陌生人 clone 你的仓库后，30 分钟内能跑通推理吗？卡在哪一步？
2. 仓库里有没有不该提交的东西？（token / 大文件 / 个人数据）
3. 如果只能展示一样东西给别人看，你选什么？

## 二、写（100 min）

交付（本日以清理 + 打包为主）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `README.md` 重写 | 一段话讲清楚是什么 + 一张架构图 + 30 分钟上手指南 + demo GIF |
| 代码清理 | 删掉调试代码、统一风格、补关键的 docstring |
| `CONTRIBUTING.md` | 怎么跑测试、怎么提 PR（哪怕只有你一个人） |
| 模型上传 | HuggingFace 私人或公开（adapter 只有几十 MB，可以公开） |
| `.gitignore` 复查 | `grep -r` 扫一遍有没有 token / 密钥 |
| demo 视频 | 60–90 秒，展示「看图问货 → 调工具 → 给答案」的完整链路 |
| 8 周复盘 | `progress/weekly-review.md` 收尾 |

## 三、跑（本地（无需 GPU））

```bash
# 看有哪些改动
git status --short | head -30
# 看仓库体积
du -sh . models/ data/ outputs/ reports/ 2>/dev/null
# 看提交历史（应该是有意义的提交信息）
git log --oneline | head -20
# 扫密钥泄漏（**必做**）
grep -rIn "sk-\|AKIA\|shpat_\|Bearer " --include="*.py" --include="*.md" --include="*.json" . 2>/dev/null | grep -v node_modules | head
```

期望输出（节选）：
```
$ du -sh .
 84M    .              ← 不含模型和数据（它们在 .gitignore 里）

$ grep 扫密钥
（无输出 = 干净）✓

$ git log --oneline | head -5
a1b2c3d docs: 补 W8 交付章节
d4e5f6a feat: 加入 Theme App Extension
...

交付物清单：
  [x] 开源仓库（README 可 30 分钟跑通）
  [x] demo 视频 90 秒
  [x] 技术报告（中英双版）
  [x] 8 周复盘
  [x] LoRA adapter 已上传
```

## 四、验收清单

- [ ] **陌生人 clone 后照 README 能在 30 分钟内跑通推理**（最好找个人真试一次）
- [ ] 密钥扫描无输出（`grep` 那一条必须干净）
- [ ] demo 视频已录（60–90 秒）
- [ ] 模型 / 数据已上传，README 里有链接
- [ ] `progress/weekly-review.md` 的 8 周复盘完整；进度表 48 天全部 `[x]`

## 五、容易踩的坑

1. **`.env` / token 提交上去了** —— 交付前必扫一遍。已经推上去的话要立刻轮换密钥，而不是只删文件（git 历史里还在）。
2. **README 的步骤只在你机器上能跑** —— 路径写死了、依赖没写版本、少了环境变量。找个干净环境试一次。
3. 模型文件提交进了 git —— 几 GB 的权重进 git 会让仓库爆炸。用 HF 或 release。
4. 大文件/临时文件没清 —— `data/synthetic/*.jsonl`、`outputs/*.png` 这类要进 .gitignore。
5. 只有一堆「update」的提交历史 —— 交付时看起来很不专业。可以 squash，但要诚实。
6. 忘了卸掉本地绝对路径 —— `sed -i` 扫一遍 `/Users/xxx` 这类硬编码路径。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。

> **全部完成。** 你现在有一个：
> ① 从零搭过 VLM 的手感 ② 一份 3k 条客服图文数据集 + 领域评测集
> ③ 一个训过的 LoRA（SFT）+ 一个偏好对齐版本（DPO）
> ④ 一个多模态客服 Agent ⑤ 一个公网可访问的 Shopify App
> ⑥ 一份有消融、有成本、有失败案例的技术报告
>
> 这已经超过「学完一门课」的量了 —— 这是一份能拿出去讲的项目。
