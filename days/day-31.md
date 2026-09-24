# Day 31 · Agent 范式与工具协议

> 预计 3–4h ｜ 📓 `notebooks/day-31_agent_tools.ipynb` ｜ 💻 本地 · `src/agent/tools.py`、`docs/10-agent.md`
> 前置：Day 30（有可用的推理服务）

## 今日目标（一句话）

定义 7 个客服工具的 schema 并全部 mock 起来，用 20 条测试 query 验证「模型能不能正确选对工具、填对参数」；并说清「让模型输出 JSON」和 「用 function calling」在鲁棒性上的差别。

## 一、读（60 min）

材料：
- `docs/10-agent.md` 第 1–3 节（ReAct / Plan-Execute / function calling schema）
- `src/agent/tools.py` 的 `TOOLS` 注册表与 `tools_schema()`

思考题（先自己想，答案在讲义或代码注释里）：
1. 工具的 **description** 写得含糊，模型会怎样？（这是最常见的失败原因）
2. 为什么 `execute_tool()` 要做白名单校验？不做会怎样？
3. ReAct 和 Plan-Execute 分别适合什么任务？客服场景该用哪个？

## 二、写（100 min）

`src/agent/tools.py`（已给实现，你要扩工具）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `Tool` dataclass | name / description / parameters / fn / 是否需要写权限 |
| `lookup_order` `shipping_status` | 查询类工具（幂等，随便调） |
| `check_stock` `product_qa` | 商品类工具 |
| `check_return_eligibility` `start_return` | 写操作类（**必须幂等**） |
| `escalate_to_human` | 兜底工具 —— 不在能力范围内就转人工，别硬答 |
| `execute_tool()` | 白名单校验 + **丢弃未声明的参数** + 异常不抛出 |
| `tools_schema()` | 导出成模型能吃的 JSON schema |

> 今天最重要的设计原则：**工具要「窄」不要「宽」**。
> 一个 `manage_order(action, order_id, ...)` 万金油工具，模型一定会填错；
> 拆成 `lookup_order` / `start_return` / `cancel_order` 三个，准确率高得多。

## 三、跑（本地（无需 GPU））

```bash
# 自检：幂等性 + 工具幻觉拦截 + schema 导出
python -m src.agent.tools
# 看看导出给模型的 schema 长什么样
python -c "import sys; sys.path.insert(0,'.'); from src.agent.tools import tools_schema; import json; print(json.dumps(tools_schema(), ensure_ascii=False, indent=2))"
```

期望输出（节选）：
```
[1] 工具注册表
     lookup_order / shipping_status / check_stock / product_qa
     check_return_eligibility / start_return / escalate_to_human    共 7 个

[2] 幂等性验证（写操作）
     调 3 次 start_return(order=A1, reason=尺码不合适)
     → 只产生 1 个退货单 ✓   （幂等键生效）

[3] 工具幻觉拦截
     模型要求调用 get_weather(A1)  → 被白名单拦下 ✓
     模型传了未声明的参数 track=true → 已丢弃 ✓

[4] 异常输入
     lookup_order(order_id=None)  → 返回友好提示，不抛异常 ✓
     check_stock(sku="不存在的SKU") → 返回"未找到"，不崩 ✓
```

## 四、验收清单

- [ ] 7 个工具的 schema 能被 `json.dumps` 正常导出（说明结构合法）
- [ ] **幂等性验证通过**：三次 `start_return` 只产生一个退货单
- [ ] 工具幻觉被拦住（模型编造的工具名 → 拒绝执行）
- [ ] 异常输入不崩，返回的是友好话术而不是 traceback
- [ ] 能说清「JSON 输出」vs「function calling」的鲁棒性差异

## 五、容易踩的坑

1. **description 写得含糊** —— 比如 `check_stock` 写「查库存」，模型不知道要传 sku 还是商品名。description 要写清「什么时候用、参数从哪来」。
2. **没有白名单校验** —— 模型编造一个工具名，你的代码就去 `getattr` 了。必须先校验名字在注册表里。
3. **丢弃未声明的参数** —— 模型多传了 `track=true`，直接塞给函数会 TypeError。静默丢弃更稳。
4. 写操作没有幂等键 —— 模型重试一次就产生两个退货单，用户炸了。
5. 工具自己抛异常 —— 异常冒到模型层，模型会开始胡言乱语。工具层必须把异常转成「可读的失败结果 + 建议」。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
