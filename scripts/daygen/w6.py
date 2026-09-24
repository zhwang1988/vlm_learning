"""Week 6 · 多模态 Agent（Day 31–36）

这一周的核心矛盾：**从「一个会看图说话的模型」到「一个能解决问题的客服」**。
关键认知：Agent 的工程质量 > 模型能力。工具设计、幂等、状态、护栏，比换个更大的模型有用得多。
"""
from __future__ import annotations


def D(**kw):
    return kw


DAYS = [
    # ======================================================================
    D(
        week=6, n=31, slug="agent_tools", title="Agent 范式与工具协议",
        where="local",
        files="`src/agent/tools.py`、`docs/10-agent.md`",
        prereq="Day 30（有可用的推理服务）",
        goal="定义 7 个客服工具的 schema 并全部 mock 起来，用 20 条测试 query 验证"
             "「模型能不能正确选对工具、填对参数」；并说清「让模型输出 JSON」和 "
             "「用 function calling」在鲁棒性上的差别。",
        read=[
            "`docs/10-agent.md` 第 1–3 节（ReAct / Plan-Execute / function calling schema）",
            "`src/agent/tools.py` 的 `TOOLS` 注册表与 `tools_schema()`",
        ],
        think=[
            "工具的 **description** 写得含糊，模型会怎样？（这是最常见的失败原因）",
            "为什么 `execute_tool()` 要做白名单校验？不做会怎样？",
            "ReAct 和 Plan-Execute 分别适合什么任务？客服场景该用哪个？",
        ],
        write_title="`src/agent/tools.py`（已给实现，你要扩工具）",
        write_rows=[
            ("`Tool` dataclass", "name / description / parameters / fn / 是否需要写权限"),
            ("`lookup_order` `shipping_status`", "查询类工具（幂等，随便调）"),
            ("`check_stock` `product_qa`", "商品类工具"),
            ("`check_return_eligibility` `start_return`", "写操作类（**必须幂等**）"),
            ("`escalate_to_human`", "兜底工具 —— 不在能力范围内就转人工，别硬答"),
            ("`execute_tool()`", "白名单校验 + **丢弃未声明的参数** + 异常不抛出"),
            ("`tools_schema()`", "导出成模型能吃的 JSON schema"),
        ],
        write_note="> 今天最重要的设计原则：**工具要「窄」不要「宽」**。\n"
                   "> 一个 `manage_order(action, order_id, ...)` 万金油工具，模型一定会填错；\n"
                   "> 拆成 `lookup_order` / `start_return` / `cancel_order` 三个，准确率高得多。",
        run=[
            ("python -m src.agent.tools", "自检：幂等性 + 工具幻觉拦截 + schema 导出"),
            ("python -c \"import sys; sys.path.insert(0,'.'); "
             "from src.agent.tools import tools_schema; import json; "
             "print(json.dumps(tools_schema(), ensure_ascii=False, indent=2))\"",
             "看看导出给模型的 schema 长什么样"),
        ],
        expect="""
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
""",
        accept=[
            "7 个工具的 schema 能被 `json.dumps` 正常导出（说明结构合法）",
            "**幂等性验证通过**：三次 `start_return` 只产生一个退货单",
            "工具幻觉被拦住（模型编造的工具名 → 拒绝执行）",
            "异常输入不崩，返回的是友好话术而不是 traceback",
            "能说清「JSON 输出」vs「function calling」的鲁棒性差异",
        ],
        pits=[
            "**description 写得含糊** —— 比如 `check_stock` 写「查库存」，模型不知道要传 sku 还是商品名。"
            "description 要写清「什么时候用、参数从哪来」。",
            "**没有白名单校验** —— 模型编造一个工具名，你的代码就去 `getattr` 了。"
            "必须先校验名字在注册表里。",
            "**丢弃未声明的参数** —— 模型多传了 `track=true`，直接塞给函数会 TypeError。静默丢弃更稳。",
            "写操作没有幂等键 —— 模型重试一次就产生两个退货单，用户炸了。",
            "工具自己抛异常 —— 异常冒到模型层，模型会开始胡言乱语。工具层必须把异常转成"
            "「可读的失败结果 + 建议」。",
        ],
        nb=[
            ("md", "## 1. 跑工具自检"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.agent.tools"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 逐个工具看「数据长什么样」\n\n写工具之前先看 mock 数据 —— 这决定了参数设计是否合理。"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.agent.tools import MOCK_ORDERS, MOCK_STOCK, RETURN_POLICY

print("订单样本（前 2 条）:")
for oid, o in list(MOCK_ORDERS.items())[:2]:
    print(f"  {oid}: {o}")
print("\\n库存样本（前 3 条）:")
for sku, s in list(MOCK_STOCK.items())[:3]:
    print(f"  {sku}: {s}")
print("\\n退货政策:", str(RETURN_POLICY)[:200])'''),
            ("md", """## 3. 设计你自己的第 8 个工具

客服场景还缺什么？候选：**改地址 / 催发货 / 申请发票 / 补差价**。
写成一个 `Tool` 并加进注册表 —— **description 要写到你妈能看懂**。"""),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.agent.tools import Tool

my_tool = Tool(
    name="urge_shipping",              # ← 改成你的
    description="当用户催促发货、询问为什么还没发货时使用。需要订单号。",
    parameters={"order_id": {"type": "string", "description": "订单号，形如 A1"}},
    fn=lambda order_id: {"ok": True, "msg": f"{order_id} 已加急"},
)
print("工具名:", my_tool.name)
print("描述长度:", len(my_tool.description), "（太短说明你没写清楚）")
print("参数:", list(my_tool.parameters))'''),
        ],
        next_day_hint="`days/day-32.md` —— 多模态 RAG：让用户能「拍图找货」",
    ),

    # ======================================================================
    D(
        week=6, n=32, slug="multimodal_rag", title="多模态 RAG",
        where="cloud",
        files="`src/agent/retriever.py`",
        prereq="Day 31；一批商品图",
        goal="搭商品图文双索引：**CLIP 向量做图搜**（用户上传图 → 找到对应商品）+ "
             "文本向量做知识检索（退货政策、尺码表），并实现混合检索 + VLM 重排。",
        read=[
            "`docs/10-agent.md` 第 4 节（多模态 RAG 的三种架构）",
            "`src/agent/retriever.py` 的 `vlm_rerank()` —— 粗排 + 细排的两段式",
        ],
        think=[
            "为什么「以图搜图」比「先让 VLM 描述再搜文本」更适合商品场景？",
            "CLIP 粗排 100 条 → VLM 细排 top 5，这个两段式为什么比直接 VLM 排 100 条好？",
            "向量归一化漏了会怎样？（内积 ≠ 余弦相似度）",
        ],
        write_title="`src/agent/retriever.py`（已给实现）",
        write_rows=[
            ("`ClipEncoder`", "图片 → 归一化向量（**归一化不能漏**）"),
            ("`build_image_index()`", "把商品图灌进向量索引"),
            ("`build_text_index()`", "把知识文档灌进文本索引"),
            ("`search_by_image()`", "用户传图 → top-k 商品"),
            ("`hybrid_search()`", "图搜 + 文本搜融合（加权或 RRF）"),
            ("`vlm_rerank()`", "CLIP 粗排 top-50 → VLM 细排 top-5（精度主要来自这一步）"),
        ],
        run=[
            ("python -m src.agent.retriever --demo", "自检：造几个商品跑通全链路"),
            ("python -m src.agent.retriever --products data/products.json "
             "--knowledge data/knowledge.json --out data/index",
             "灌正式索引（没有数据文件就用 --demo 的输出）"),
            ("ls -la data/index/", "确认索引落盘"),
        ],
        expect="""
[clip] 载入 CLIP 编码器（约 600 MB）
[build] 商品 100 件 → 图片索引 100 条，维度 512
[build] 知识 42 条 → 文本索引 42 条
[demo] 以图搜图 top-3：
   1. 米白色针织衫 (0.91)
   2. 奶白色针织开衫 (0.87)
   3. 米色圆领毛衣 (0.83)
[rerank] VLM 重排后 top-1 = 米白色针织衫（CLIP 排第 2）← 重排把名次纠正了
""",
        accept=[
            "图搜 top-3 准确率 ≥ 70%（在 100 张商品图上测，要人工核对）",
            "混合检索能同时吃到图片信号和文本信号",
            "能解释「以图搜图」比「先描述再搜文本」好在哪（**信息损失**角度）",
            "知道归一化漏了会有什么后果（相似度排序全乱）",
        ],
        pits=[
            "**只做文本检索** —— 用户传了图你却把图扔了，等于多模态白做。",
            "**向量没归一化** —— 内积相似度会被向量模长主导，长文本/大图占便宜，排序全乱。",
            "没有重排 —— CLIP 的 top-1 经常不准，加一层 VLM 重排能显著提升。",
            "索引没持久化 —— 每次请求都重新编码 100 张图，延迟爆炸。",
            "商品下架了索引没删 —— 用户搜到一个买不到的商品（Day 40 的 webhook 要解决这个）。",
        ],
        nb=[
            ("md", "## 1. 跑自检，看「以图搜图」到底行不行"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.agent.retriever", "--demo"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-2500:] or r.stderr[-2500:])'''),
            ("md", "## 2. 归一化漏了的后果（亲手验证）"),
            ("code", '''import numpy as np

rng = np.random.default_rng(0)
q  = rng.normal(size=8)
a  = rng.normal(size=8) * 0.2      # 同方向但模长很小
b  = rng.normal(size=8) * 5.0      # 另一个方向但模长很大

def cos(x, y):  return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
def dot(x, y):  return float(x @ y)

print("a 与 q 的真实相似度(cos):", round(cos(q, a), 4))
print("b 与 q 的真实相似度(cos):", round(cos(q, b), 4))
print()
print("不做归一化时：a 的内积", round(dot(q, a), 3), " b 的内积", round(dot(q, b), 3))
print("→ 如果不归一化，模长大的向量会霸榜，余弦相似度排序被彻底破坏")'''),
            ("md", "## 3. 设计你的 RAG 评测集\n\n**不测准确率就等于没做**。写 10 条「传图 → 期望商品 ID」。"),
            ("code", '''retrieval_eval = [
    # ("图片路径", "期望的商品ID"),
    # ("data/raw_images/tee_white_01.jpg", "SKU-1001"),
]
print(f"评测集 {len(retrieval_eval)} 条 —— 至少写 10 条，跑一遍算 top-3 命中率")'''),
        ],
        next_day_hint="`days/day-33.md` —— 工具链完善：幂等、事务、失败降级",
    ),

    # ======================================================================
    D(
        week=6, n=33, slug="tools_state_mgmt", title="工具链完善与状态管理",
        where="local",
        files="`src/agent/tools.py`（幂等）、`src/agent/state.py`（今日抽出）",
        prereq="Day 31/32",
        goal="把 mock 工具补成有真实业务逻辑的实现，加上**幂等键、事务边界、失败降级话术**；"
             "并把会话状态从 `agent.py` 里抽成一个独立模块。",
        read=[
            "`docs/10-agent.md` 第 5 节（会话状态、幂等、失败回滚）",
            "`src/agent/agent.py` 里的 `AgentState` —— 今天要把它抽出去",
        ],
        think=[
            "为什么「下单类工具必须幂等」？模型重试的触发路径有哪些？",
            "什么是「半成功状态」？举例：扣了库存但没创建订单。",
            "失败降级话术为什么不能只说「系统繁忙请稍后再试」？",
        ],
        write_title="抽出 `src/agent/state.py`，并给工具补业务语义",
        write_rows=[
            ("`SessionState`", "会话 id / 历史轮次 / 已提取的槽位 / 视觉证据引用"),
            ("`IdempotencyKey`", "由（会话 id + 工具名 + 关键参数）确定性生成 —— **不能用随机数**"),
            ("事务边界", "一个工具内要么全成要么全败，不许留半成品"),
            ("失败降级", "每个工具的失败分支给出**具体**建议，而不是套话"),
            ("`escalate_to_human` 的触发条件", "连续 2 次工具失败 / 用户明确要求 / 超出能力边界"),
            ("`state.py` 的序列化", "状态能存能读（Redis / 数据库 / 内存字典三选一）"),
        ],
        write_note="> 今天的判据很简单：**同一个请求发三次，系统的状态变化必须和发一次完全一样。**",
        run=[
            ("python -m src.agent.tools", "幂等性自检（会自动连调 3 次写工具）"),
            ("python -c \"import sys; sys.path.insert(0,'.'); "
             "from src.agent.state import SessionState; "
             "s=SessionState('s1'); [s.add_turn('user', 'hi') for _ in range(3)]; "
             "print(s.summary())\"",
             "验证抽出来的状态模块能独立工作"),
        ],
        expect="""
[幂等] start_return(order=A1, reason=尺码不合适)
       第 1 次 → 创建退货单 R-001
       第 2 次 → 命中幂等键，返回同一个 R-001（未新建）
       第 3 次 → 同上
       最终退货单数量 = 1  ✓

[降级] check_stock(sku="NOPE")  → {"ok": false, "suggestion": "该 SKU 不存在，"
                                  "建议确认商品链接或让用户提供商品名"}

[状态] SessionState('s1') 3 轮对话，槽位 {"order_id": "A1"}，视觉证据 1 条
""",
        accept=[
            "幂等性自检通过：三次写操作只产生一个业务实体",
            "每个工具的失败分支都有**具体**的 `suggestion`（不是「请稍后再试」）",
            "`state.py` 能独立 import 并使用（不依赖 `agent.py`）",
            "能说出「半成功状态」的一个具体例子和它的防范方式",
        ],
        pits=[
            "**幂等键用随机数 / 时间戳** —— 每次都新建，幂等完全失效。必须是「会话 + 工具 + 参数」的确定性哈希。",
            "**事务边界不清** —— 扣了库存但创建订单失败，库存就少了。要么包事务，要么先做可回滚的步骤。",
            "状态存在进程内存里 —— 服务重启会话就丢，多实例部署更是直接错乱。",
            "失败话术是套话 —— 用户听不懂也帮不上你。要给「下一步能做什么」。",
            "无限重试 —— 工具失败就重试，把下游打挂。要有次数上限 + 退避。",
        ],
        nb=[
            ("md", "## 1. 幂等性：亲手验证"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.agent.tools import execute_tool

results = [execute_tool("start_return", {"order_id": "A1", "reason": "尺码不合适"})
           for _ in range(3)]
for i, r in enumerate(results, 1):
    print(f"第 {i} 次:", r.data if hasattr(r, "data") else r)
unique = {str(r.data if hasattr(r, "data") else r) for r in results}
print("\\n不同结果数:", len(unique), "（应该是 1 —— 三次调用只创建一个退货单）")'''),
            ("md", "## 2. 幂等键的设计（自己写一个）"),
            ("code", '''import hashlib, json

def idem_key(session_id, tool_name, args):
    """确定性幂等键：同样的会话 + 工具 + 参数 → 同一个键。"""
    payload = json.dumps([session_id, tool_name, sorted(args.items())],
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

k1 = idem_key("s1", "start_return", {"order_id": "A1", "reason": "尺码不合适"})
k2 = idem_key("s1", "start_return", {"order_id": "A1", "reason": "尺码不合适"})
k3 = idem_key("s2", "start_return", {"order_id": "A1", "reason": "尺码不合适"})
print("同会话同参数:", k1 == k2, "（必须 True）")
print("异会话同参数:", k1 == k3, "（必须 False —— 不同用户各退一份）")'''),
            ("md", """## 3. 抽出 state.py 的验收

`python -c "from src.agent.state import SessionState"` 能通过，且不触发 torch 导入 ——
说明状态管理和模型推理**解耦**了。这是工程化的关键一步。"""),
            ("code", '''state_check = """
抽出来的模块必须满足：
  [ ] 不 import torch
  [ ] 不 import transformers
  [ ] 能独立单元测试
  [ ] 能序列化成 JSON（为了存 Redis / DB）
"""
print(state_check)'''),
        ],
        next_day_hint="`days/day-34.md` —— Agent 主循环（ReAct + 护栏）",
    ),

    # ======================================================================
    D(
        week=6, n=34, slug="agent_loop", title="Agent 骨架",
        where="cloud",
        files="`src/agent/agent.py`（主循环）、`src/agent/guardrails.py` / `tracing.py`（今日抽出）",
        prereq="Day 31–33",
        goal="把主循环搭起来：意图理解 → 检索 → 规划 → 工具调用 → 生成回复；"
             "并加上三道护栏（最大步数 / 重复动作检测 / 成本上限），连续 10 条不出死循环。",
        read=[
            "`docs/10-agent.md` 第 6 节（主循环 + 护栏）",
            "`src/agent/agent.py` 的 `extract_visual_evidence` —— 解决「第二轮图丢了」的那段",
        ],
        think=[
            "为什么「第二轮图片丢了」是 VLM Agent 的经典 bug？怎么根治？",
            "死循环的典型形态是什么？（模型反复调同一个工具、参数一模一样）",
            "成本上限该按什么维度设？轮数还是 token 数？",
        ],
        write_title="`src/agent/agent.py` + 抽出 `guardrails.py` / `tracing.py`",
        write_rows=[
            ("`extract_visual_evidence()`", "把图片里的关键信息转成**文本证据**存进 state，后续轮次不再依赖原图"),
            ("`SYSTEM_PROMPT`", "严格 JSON 输出格式；明确「不在能力范围内就调 escalate」"),
            ("主循环 `arun()`", "解析模型输出 → 若带 tool_call 则执行并把结果**回灌** → 再问"),
            ("`guardrails.max_steps`", "步数上限（建议 6）"),
            ("`guardrails` 重复动作检测", "连续两次相同工具 + 相同参数 → 直接跳出并降级"),
            ("`guardrails` 成本上限", "累计 token / 金额超限 → 停止并转人工"),
            ("`tracing`", "每步落 JSONL：step / 工具 / 参数 / 结果 / 耗时 / token —— 可回放"),
        ],
        write_note="> **今天最值钱的设计是「视觉证据持久化」**：\n"
                   "> 用户第一轮发了图，第二轮只说「那这个多少钱」——如果每轮都只把当轮图喂给模型，"
                   "第二轮模型就「失明」了。做法：第一轮就把图中的关键信息抽成文本存进 state。",
        run=[
            ("python -m src.agent.agent --selftest", "自检：三条典型 query 走完整循环"),
            ("python -m src.agent.agent --query \"我的订单 A1 为什么还没到？\" "
             "--session s_demo_1 --verbose",
             "单轮真实咨询，看完整 trace"),
            ("for i in $(seq 1 10); do python -m src.agent.agent --query \"查一下订单 A$i\" "
             "--session s_batch_$i > /dev/null || echo \"第 $i 条失败\"; done",
             "连跑 10 条，验不出死循环"),
        ],
        expect="""
[step 1] 意图：物流查询  → 需要工具：lookup_order
[step 1] tool_call lookup_order({"order_id": "A1"})
         → {"status": "已发货", "carrier": "顺丰", "eta": "2026-09-26"}
[step 2] 信息够了，生成回复
[answer] 您的订单 A1 已于 9 月 23 日发出，承运顺丰，预计 9 月 26 日送达。
[cost] 2 步 · 1 次工具调用 · 1843 tokens · 4.2s

（护栏演示）
[guard] 检测到重复动作 lookup_order(A1) —— 第 2 次完全相同的调用
[guard] 已中断循环，降级为转人工
""",
        accept=[
            "`--selftest` 三条典型 query 全部走通（查询类 / 写操作类 / 超范围类）",
            "**连续 10 条 query 不出死循环、不超时**（这是硬指标）",
            "trace JSONL 落盘，能回放每一步的工具、参数、结果、耗时",
            "能演示「第二轮图片不丢」—— 第一轮传图，第二轮只发文字，模型仍能答对",
        ],
        pits=[
            "**图片在第二轮「丢了」** —— 只喂当轮图，多轮对话立刻失明。必须做视觉证据持久化。",
            "**没有 max_steps** —— 模型陷入循环，一次请求烧掉几十次调用。",
            "重复动作检测只看工具名不看参数 —— 参数不同的同类调用是正常的，别误杀。",
            "**工具结果没有回灌给模型** —— 模型不知道执行结果，第二轮又调一遍。",
            "SYSTEM_PROMPT 没规定输出格式 —— 每轮解析都得写一堆正则兜底，脆得一碰就碎。",
            "trace 只记成功步骤 —— 失败步骤才是最需要 debug 的，一定要记。",
        ],
        nb=[
            ("md", "## 1. 自检三条路径"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.agent.agent", "--selftest"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-3000:] or r.stderr[-3000:])'''),
            ("md", "## 2. 视觉证据持久化 —— 看它怎么解决「图丢了」"),
            ("code", '''import sys; sys.path.insert(0, "..")
from PIL import Image, ImageDraw
import inspect
from src.agent.agent import CXAgent, AgentConfig

# 看一眼这个方法的实现思路
src = inspect.getsource(CXAgent.extract_visual_evidence)
print(src[:1400])'''),
            ("md", "## 3. 护栏实验：人为制造死循环\n\n把 max_steps 调到 20，问一个工具解决不了的问题，看它怎么绕不出来、护栏怎么拦。"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.agent.agent import AgentConfig

cfg = AgentConfig()
for field in ("max_steps", "max_tool_calls", "cost_limit", "repeat_threshold"):
    print(f"{field:18s} = {getattr(cfg, field, '（字段名请对照实际实现）')}")

print("\\n把 max_steps 调成 20 再问『帮我预测明天的天气』——")
print("观察：模型会反复尝试不存在的工具，护栏在第 N 步拦截")'''),
        ],
        next_day_hint="`days/day-35.md` —— 端到端联调：做个能录屏的界面",
    ),

    # ======================================================================
    D(
        week=6, n=35, slug="agent_demo", title="端到端联调",
        where="cloud",
        files="`src/agent/demo.py`",
        prereq="Day 34",
        goal="做一个 Gradio 界面：能上传图 + 打字，**能看到 Agent 的思考过程和工具调用**；"
             "在 20 条真实场景 query 上跑通，人工判定成功率。",
        read=[
            "（动手日，没有新讲义）",
            "`src/agent/demo.py` 的 `build_agent_demo` / `build_compare_demo`",
        ],
        think=[
            "为什么 demo 必须显示中间步骤？不显示会怎样？",
            "`--compare` 模式（基座 vs 微调）能说明什么？",
            "什么样的 demo 能录屏给别人看？（提示：有对比、有失败、有工具调用）",
        ],
        write_title="`src/agent/demo.py`（已给实现）",
        write_rows=[
            ("`build_agent_demo()`", "上传图 + 输入框 + 对话历史 + **中间步骤折叠面板**"),
            ("`build_compare_demo()`", "左右对照：基座 vs 你的模型，同一问题"),
            ("中间步骤展示", "工具名 / 参数 / 返回 / 耗时 —— 这是 demo 的说服力来源"),
            ("错误兜底", "工具报错时界面要显示友好提示而不是 traceback"),
            ("`--share`", "生成公网链接（方便手机上试）"),
        ],
        run=[
            ("python -m src.agent.demo --port 7860", "起界面"),
            ("python -m src.agent.demo --compare --base Qwen/Qwen2.5-VL-3B-Instruct "
             "--adapter outputs/qwen25vl3b-cx-dpo-v0",
             "起对比界面"),
            ("python -m src.agent.demo --share", "生成公网链接"),
        ],
        expect="""
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
""",
        accept=[
            "界面能上传图 + 打字，且**能看到工具调用过程**",
            "20 条真实场景 query 跑通，人工判定成功率（≥65% 为 M6 目标）",
            "至少能录一段 60 秒的 demo 视频（这是 W8 交付物的素材）",
            "错误路径也好看（工具失败时显示友好提示，不是红字 traceback）",
        ],
        pits=[
            "**Gradio 里图片没真正传到后端** —— 传的是文件路径而模型要 PIL 对象，静默失败。要显式转换。",
            "长会话上下文爆掉 —— Gradio 的 `chat_history` 越滚越长，必须做截断或摘要。",
            "**不显示中间步骤** —— 出问题时你无法 debug，评审也看不出你做了 Agent 而不是套壳。",
            "`--share` 链接暴露了未脱敏数据 —— 公网链接不要对着真实订单数据演示。",
            "demo 跑在 notebook 里 —— Gradio 要阻塞主线程，notebook 里会很别扭，用终端跑。",
        ],
        nb=[
            ("md", "## 1. 起界面（在终端里跑）"),
            ("code", '''print("""
在终端执行（不要在这个 notebook 里跑，Gradio 会阻塞）：

    python -m src.agent.demo --port 7860

手机上看：

    python -m src.agent.demo --share
""")'''),
            ("md", "## 2. 准备 20 条真实场景 query（今天的主产出）\n\n从你 Day 8 抄来的真实问法里挑 20 条，覆盖 8 个意图。"),
            ("code", '''real_queries = [
    ("这件米白针织衫有货吗？", None),                 # (问题, 图片路径)
    ("订单 A1 到哪了？", None),
    ("我要退货，尺码不合适", None),
    # ... 补齐到 20 条，覆盖 8 个意图
]
from collections import Counter
print(f"共 {len(real_queries)} 条，还差 {max(0, 20 - len(real_queries))} 条")'''),
            ("md", "## 3. 人工判定表"),
            ("code", '''import json
from pathlib import Path

verdicts = {"成功": [], "部分成功": [], "失败": []}
RESULT_FILE = "../reports/agent_demo_20.jsonl"
if Path(RESULT_FILE).exists():
    rows = [json.loads(l) for l in Path(RESULT_FILE).read_text().splitlines() if l.strip()]
    for r in rows:
        verdicts[r.get("verdict", "失败")].append(r.get("query"))
    for k, v in verdicts.items():
        print(f"{k}: {len(v)} 条  ({len(v)/max(len(rows),1):.0%})")
    print("\\n失败集中在:", verdicts["失败"][:3])
else:
    print("先在界面里跑完 20 条，把结果记到 reports/agent_demo_20.jsonl")'''),
        ],
        next_day_hint="`days/day-36.md` —— Agent 评测，W6 收官",
    ),

    # ======================================================================
    D(
        week=6, n=36, slug="agent_eval", title="Agent 评测",
        where="local",
        files="`src/eval/agent_eval.py`",
        prereq="Day 35（20 条真实场景已跑过）",
        goal="建 Agent 评测体系：任务完成率 / 工具选择准确率 / 平均轮数 / P95 延迟 / 单次成本；"
             "出 `reports/agent_eval_v1.md`，并定位失败集中在哪类任务。",
        read=[
            "`src/eval/agent_eval.py` 的 `TaskVerifier` —— 三类判据：调了工具没 / 答对了没 / 状态变了没",
            "`docs/10-agent.md` 第 7 节（Agent 评测与失败模式）",
        ],
        think=[
            "「任务完成率」怎么自动判定？为什么不能只看答案文本？",
            "只看成功率不看 P95 延迟会有什么后果？",
            "单次成本怎么算？（提示：token + 工具调用次数 + 重试次数）",
        ],
        write_title="`src/eval/agent_eval.py`（已给实现，你要扩任务集）",
        write_rows=[
            ("`AgentTask`", "任务定义：query / 图片 / 期望工具 / 期望答案要点 / 期望状态变化"),
            ("`TaskVerifier`", "三条判据：`tool_called` / `answered_contains` / `state_changed`"),
            ("`DEFAULT_TASKS`", "内置任务集 —— **今天要把它扩到 30 条以上**"),
            ("`run_agent_eval()`", "批量跑 + 统计成功率、平均轮数、P95 延迟"),
            ("`build_agent_report()`", "出报告，含失败归类与下一步建议"),
        ],
        run=[
            ("python -m src.eval.agent_eval --self-test", "自检（用假 Agent 验评测逻辑）"),
            ("python -m src.eval.agent_eval --out reports/agent_eval_v1.md",
             "跑真实 Agent 评测"),
            ("head -60 reports/agent_eval_v1.md", "看报告"),
        ],
        expect="""
[verifier] 自检：三条判据在「正例 / 反例」上都判对 ✓
[tasks] 载入 32 条任务（查询 14 / 写操作 8 / 超范围 6 / 多轮 4）
[run] 32/32 完成
  任务完成率    : 68.8%   (22/32)   ← 目标 ≥65% ✓
  工具选择准确率 : 81.3%
  平均轮数       : 2.4
  P95 延迟       : 7.8 s
  单次成本       : ¥0.031
失败归类：
  超范围任务误答 5 条   ← 最大类（该转人工却硬答）
  多轮上下文丢失 3 条
  写操作幂等误判 2 条
[out] reports/agent_eval_v1.md
""",
        accept=[
            "任务集 ≥30 条，覆盖查询 / 写操作 / 超范围 / 多轮四类",
            "`--self-test` 通过（**评测逻辑本身必须先被验证**）",
            "报告含五个指标：成功率 / 工具准确率 / 平均轮数 / P95 / 单次成本",
            "能定位失败集中在哪一类任务，并给出具体改法",
            "`progress/weekly-review.md` 的 W6 段已写；进度表 W6 六天 `[x]`，M6 打卡",
        ],
        pits=[
            "**只看成功率** —— 一个每次都要 8 轮、延迟 30 秒的 Agent，成功率再高也不可用。",
            "**单次成本没算** —— W8 的成本模型就建在这一天的数据上，现在不算后面就是拍脑袋。",
            "失败样本没归类 —— 只知道「失败了 10 条」，不知道「为什么失败」，没法改。",
            "**任务集太小** —— 5 条任务的成功率没有统计意义。至少 30 条。",
            "评测逻辑自己没测 —— 判据写错了，你会得到完全错误的结论。`--self-test` 不能跳。",
        ],
        nb=[
            ("md", "## 1. 先验证评测逻辑本身"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.eval.agent_eval", "--self-test"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 设计你自己的任务（这是今天的核心工作）\n\n每条任务要写清「**怎么算通过**」。"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.eval.agent_eval import DEFAULT_TASKS, AgentTask

print(f"内置任务 {len(DEFAULT_TASKS)} 条，看两条结构：")
for t in DEFAULT_TASKS[:2]:
    print("=" * 60)
    for k, v in vars(t).items():
        print(f"  {k:22s} {v}")

my_tasks = [
    # AgentTask(query="我要把 A1 退了，尺码不对",
    #           expect_tools=["start_return"],
    #           answer_contains=["退货", "已"], state_key="return_created"),
]
print(f"\\n我还需要补 {max(0, 30 - len(DEFAULT_TASKS) - len(my_tasks))} 条")'''),
            ("md", "## 3. 成本核算方法（W8 会用到）"),
            ("code", '''TOKEN_PRICE = {"in": 0.4 / 1e6, "out": 1.2 / 1e6}   # ¥ / token，按你实际用的算

def session_cost(n_steps, in_tokens, out_tokens, tool_calls):
    llm = in_tokens * TOKEN_PRICE["in"] + out_tokens * TOKEN_PRICE["out"]
    tools = tool_calls * 0.0001      # 假设每次工具调用含内部查询成本
    return llm + tools

c = session_cost(n_steps=2, in_tokens=3200, out_tokens=180, tool_calls=2)
print(f"单次会话成本 ≈ ¥{c:.4f}")
print(f"1000 次会话   ≈ ¥{c*1000:.2f}")
print("\\n→ 对照你的定价：¥29/月的店，每月能承受几次会话？")'''),
        ],
        footer="> **M6 达成**：任务成功率 ≥65%，且你能说清失败集中在哪、下一步怎么改。\n"
               "> 达不到也没关系 —— 把「成功率」目标改成「有评测体系 + 定位到失败模式」也算达成。",
        next_day_hint="`days/day-37.md` —— W7 Shopify 周：Admin GraphQL",
    ),
]
