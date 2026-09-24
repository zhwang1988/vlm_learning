"""
agent —— 多模态客服 Agent。

对应讲义：docs/10-agent.md
对应计划：Week 6（Day 31–Day 36）

为什么客服场景必须用 Agent：
  图片理解是基础，但完整解决用户问题必须靠外部信息和副作用操作。
  「我的快递到哪了」纯模型只能瞎猜；「我要退货」纯模型无法真的创建退货单。

文件导航：
  tools.py      Day 31/33  工具 schema + mock 实现 + 幂等 + 结构化错误
  retriever.py  Day 32     多模态 RAG（CLIP 以图搜图 + 文本知识检索 + VLM 精排）
  agent.py      Day 34     Structured ReAct 主循环 + 三个护栏
  demo.py       Day 35     Gradio 演示界面 + 基座对比界面

一键流程：
    # Day 31 验证工具层（含幂等测试）
    python -m src.agent.tools

    # Day 32 验证检索
    python -m src.agent.retriever --demo
    python -m src.agent.retriever --products data/products.json \\
        --knowledge data/knowledge.json --out data/index

    # Day 34 验证护栏
    python -m src.agent.agent --selftest

    # Day 34 跑真实对话（需 API key）
    python -m src.agent.agent --query "我的快递到哪了" --session s_demo_1

    # Day 35 起界面
    python -m src.agent.demo

    # Day 36 评测
    python -m src.eval.agent_eval --self-test

三个必须有的护栏（都在 agent.py 里）：
    1. max_steps        防止无限循环
    2. 重复 action 检测  同一工具同样参数调两次 → 强制跳出
    3. 成本上限         累计 token 超阈值 → 转人工

两个容易被忽略的设计：
    ⭐ 幂等键     有副作用的工具（start_return）必须幂等
    ⭐ 视觉证据   每轮都带上，否则「图丢了」（Agent 拿到工具结果后忘了原图）
"""

# ---------------------------------------------------------------------------
# 惰性导出（PEP 562）
#
# 这里故意不写 `from .xxx import yyy`。因为本包下有些模块要 import torch，
# 有些不要。急切导入会让「只想跑 torch-free 模块」的人在本地直接撞
# ModuleNotFoundError —— 纯粹被连坐。
#
# 改成按需加载后：
#     from src.agent import AgentConfig      # 触发时才 import 对应模块
#     python -m src.agent.<torch-free 模块>   # 本地可跑
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    "SessionState": ".state",
    "Turn": ".state",
    "VisualEvidence": ".state",
    "extract_visual_evidence": ".state",
    "idempotency_key": ".state",
    "Guardrails": ".guardrails",
    "GuardrailConfig": ".guardrails",
    "Verdict": ".guardrails",
    "Trace": ".tracing",
    "Step": ".tracing",
    "AgentConfig": ".agent",
    "AgentState": ".agent",
    "CXAgent": ".agent",
    "ClipEncoder": ".retriever",
    "IndexEntry": ".retriever",
    "MultimodalRetriever": ".retriever",
    "TOOLS": ".tools",
    "TOOL_MAP": ".tools",
    "TextEncoder": ".retriever",
    "Tool": ".tools",
    "ToolResult": ".tools",
    "VectorIndex": ".retriever",
    "execute_tool": ".tools",
    "get_default_agent": ".agent",
    "get_default_retriever": ".retriever",
    "set_default_retriever": ".retriever",
    "tools_schema": ".tools",
    "vlm_rerank": ".retriever",
}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value        # 缓存，下次不再走这里
    return value


def __dir__():
    return sorted(set(__all__) | set(_LAZY))


__all__ = [
    "AgentConfig",
    "AgentState",
    "CXAgent",
    "get_default_agent",
    "ClipEncoder",
    "IndexEntry",
    "MultimodalRetriever",
    "TextEncoder",
    "VectorIndex",
    "get_default_retriever",
    "set_default_retriever",
    "vlm_rerank",
    "TOOL_MAP",
    "TOOLS",
    "Tool",
    "ToolResult",
    "execute_tool",
    "tools_schema",
    "SessionState", "Turn", "VisualEvidence",
    "extract_visual_evidence", "idempotency_key",
    "Guardrails", "GuardrailConfig", "Verdict",
    "Trace", "Step",
]
