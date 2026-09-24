# Day 34 · Agent 骨架

> 预计 3–4h ｜ 📓 `notebooks/day-34_agent_loop.ipynb` ｜ ☁️ 云 GPU · `src/agent/agent.py`（主循环）、`src/agent/guardrails.py` / `tracing.py`（今日抽出）
> 前置：Day 31–33

## 今日目标（一句话）

把主循环搭起来：意图理解 → 检索 → 规划 → 工具调用 → 生成回复；并加上三道护栏（最大步数 / 重复动作检测 / 成本上限），连续 10 条不出死循环。

## 一、读（60 min）

材料：
- `docs/10-agent.md` 第 6 节（主循环 + 护栏）
- `src/agent/agent.py` 的 `extract_visual_evidence` —— 解决「第二轮图丢了」的那段

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么「第二轮图片丢了」是 VLM Agent 的经典 bug？怎么根治？
2. 死循环的典型形态是什么？（模型反复调同一个工具、参数一模一样）
3. 成本上限该按什么维度设？轮数还是 token 数？

## 二、写（100 min）

`src/agent/agent.py` + 抽出 `guardrails.py` / `tracing.py`

| 函数 / 文件 | 你要做什么 |
|---|---|
| `extract_visual_evidence()` | 把图片里的关键信息转成**文本证据**存进 state，后续轮次不再依赖原图 |
| `SYSTEM_PROMPT` | 严格 JSON 输出格式；明确「不在能力范围内就调 escalate」 |
| 主循环 `arun()` | 解析模型输出 → 若带 tool_call 则执行并把结果**回灌** → 再问 |
| `guardrails.max_steps` | 步数上限（建议 6） |
| `guardrails` 重复动作检测 | 连续两次相同工具 + 相同参数 → 直接跳出并降级 |
| `guardrails` 成本上限 | 累计 token / 金额超限 → 停止并转人工 |
| `tracing` | 每步落 JSONL：step / 工具 / 参数 / 结果 / 耗时 / token —— 可回放 |

> **今天最值钱的设计是「视觉证据持久化」**：
> 用户第一轮发了图，第二轮只说「那这个多少钱」——如果每轮都只把当轮图喂给模型，第二轮模型就「失明」了。做法：第一轮就把图中的关键信息抽成文本存进 state。

## 三、跑（在云 GPU 上）

```bash
# 自检：三条典型 query 走完整循环
python -m src.agent.agent --selftest
# 单轮真实咨询，看完整 trace
python -m src.agent.agent --query "我的订单 A1 为什么还没到？" --session s_demo_1 --verbose
# 连跑 10 条，验不出死循环
for i in $(seq 1 10); do python -m src.agent.agent --query "查一下订单 A$i" --session s_batch_$i > /dev/null || echo "第 $i 条失败"; done
```

期望输出（节选）：
```
[step 1] 意图：物流查询  → 需要工具：lookup_order
[step 1] tool_call lookup_order({"order_id": "A1"})
         → {"status": "已发货", "carrier": "顺丰", "eta": "2026-09-26"}
[step 2] 信息够了，生成回复
[answer] 您的订单 A1 已于 9 月 23 日发出，承运顺丰，预计 9 月 26 日送达。
[cost] 2 步 · 1 次工具调用 · 1843 tokens · 4.2s

（护栏演示）
[guard] 检测到重复动作 lookup_order(A1) —— 第 2 次完全相同的调用
[guard] 已中断循环，降级为转人工
```

## 四、验收清单

- [ ] `--selftest` 三条典型 query 全部走通（查询类 / 写操作类 / 超范围类）
- [ ] **连续 10 条 query 不出死循环、不超时**（这是硬指标）
- [ ] trace JSONL 落盘，能回放每一步的工具、参数、结果、耗时
- [ ] 能演示「第二轮图片不丢」—— 第一轮传图，第二轮只发文字，模型仍能答对

## 五、容易踩的坑

1. **图片在第二轮「丢了」** —— 只喂当轮图，多轮对话立刻失明。必须做视觉证据持久化。
2. **没有 max_steps** —— 模型陷入循环，一次请求烧掉几十次调用。
3. 重复动作检测只看工具名不看参数 —— 参数不同的同类调用是正常的，别误杀。
4. **工具结果没有回灌给模型** —— 模型不知道执行结果，第二轮又调一遍。
5. SYSTEM_PROMPT 没规定输出格式 —— 每轮解析都得写一堆正则兜底，脆得一碰就碎。
6. trace 只记成功步骤 —— 失败步骤才是最需要 debug 的，一定要记。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
