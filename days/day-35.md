# Day 35 · 端到端联调

> 预计 3–4h ｜ 📓 `notebooks/day-35_agent_demo.ipynb` ｜ ☁️ 云 GPU · `src/agent/demo.py`
> 前置：Day 34

## 今日目标（一句话）

做一个 Gradio 界面：能上传图 + 打字，**能看到 Agent 的思考过程和工具调用**；在 20 条真实场景 query 上跑通，人工判定成功率。

## 一、读（60 min）

材料：
- （动手日，没有新讲义）
- `src/agent/demo.py` 的 `build_agent_demo` / `build_compare_demo`

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么 demo 必须显示中间步骤？不显示会怎样？
2. `--compare` 模式（基座 vs 微调）能说明什么？
3. 什么样的 demo 能录屏给别人看？（提示：有对比、有失败、有工具调用）

## 二、写（100 min）

`src/agent/demo.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `build_agent_demo()` | 上传图 + 输入框 + 对话历史 + **中间步骤折叠面板** |
| `build_compare_demo()` | 左右对照：基座 vs 你的模型，同一问题 |
| 中间步骤展示 | 工具名 / 参数 / 返回 / 耗时 —— 这是 demo 的说服力来源 |
| 错误兜底 | 工具报错时界面要显示友好提示而不是 traceback |
| `--share` | 生成公网链接（方便手机上试） |

## 三、跑（在云 GPU 上）

```bash
# 起界面
python -m src.agent.demo --port 7860
# 起对比界面
python -m src.agent.demo --compare --base Qwen/Qwen2.5-VL-3B-Instruct --adapter outputs/qwen25vl3b-cx-dpo-v0
# 生成公网链接
python -m src.agent.demo --share
```

期望输出（节选）：
```
Running on local URL:  http://127.0.0.1:7860

界面元素：
  [上传图片]  [输入框: 这件有货吗？]  [发送]
  ┌── 思考过程 ─────────────────────┐
  │ step 1  意图=商品咨询            │
  │ step 1  tool=product_qa(图片)    │
  │          → 商品=米白针织衫 SKU-1001│
  │ step 2  tool=check_stock(SKU-1001)│
  │          → {M: 3, L: 5, S: 0}    │
  │ 用时 4.2s                        │
  └──────────────────────────────────┘
  回答：M/L 有现货，S 码预计 3 天补货。
```

## 四、验收清单

- [ ] 界面能上传图 + 打字，且**能看到工具调用过程**
- [ ] 20 条真实场景 query 跑通，人工判定成功率（≥65% 为 M6 目标）
- [ ] 至少能录一段 60 秒的 demo 视频（这是 W8 交付物的素材）
- [ ] 错误路径也好看（工具失败时显示友好提示，不是红字 traceback）

## 五、容易踩的坑

1. **Gradio 里图片没真正传到后端** —— 传的是文件路径而模型要 PIL 对象，静默失败。要显式转换。
2. 长会话上下文爆掉 —— Gradio 的 `chat_history` 越滚越长，必须做截断或摘要。
3. **不显示中间步骤** —— 出问题时你无法 debug，评审也看不出你做了 Agent 而不是套壳。
4. `--share` 链接暴露了未脱敏数据 —— 公网链接不要对着真实订单数据演示。
5. demo 跑在 notebook 里 —— Gradio 要阻塞主线程，notebook 里会很别扭，用终端跑。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
