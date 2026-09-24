"""
eval —— 评测体系。

对应讲义：docs/08-evaluation.md
对应计划：Week 4（Day 19–24）、Day 36（Agent 评测）

**这是整个项目最重要的一周。** 没有评测的调优是玄学。

文件导航：
  build_domain_eval.py  Day 20  构建客服领域评测集（四层难度 + 评测集卡片）
  metrics.py            Day 21  规则判定（要素/拒答/过度承诺/PII）
  judge.py              Day 21  LLM-as-judge + 位置偏见校准 + 人工一致性校准
  hallucination.py      Day 22  POPE 式幻觉评测 + 四种缓解策略对比
  run_eval.py           Day 21/24 主流程，一条命令出报告
  error_analysis.py     Day 23  失败模式聚类 + bad_cases 输出（供 DPO 用）
  agent_eval.py         Day 36  Agent 评测（看世界状态，不只看回复）

完整流程：
    # Day 20 建评测集
    python -m src.eval.build_domain_eval --source data/processed/clean.jsonl
    # → 人工复核 L4 样本！

    # Day 21 生成 judge 校准表
    python -m src.eval.judge --make-sheet --n 50

    # Day 21 跑基线
    python -m src.eval.run_eval --model Qwen/Qwen2.5-VL-3B-Instruct --tag base

    # Day 24 跑微调版本 + 对比
    python -m src.eval.run_eval --model Qwen/Qwen2.5-VL-3B-Instruct \
        --adapter outputs/qwen25vl3b-cx-lora-v0 --tag sft \
        --baseline reports/eval_base_raw.jsonl

    # Day 23 错误分析
    python -m src.eval.error_analysis --in reports/eval_sft_raw.jsonl

    # Day 22 幻觉评测
    python -m src.eval.hallucination --samples data/processed/clean.jsonl

    # Day 36 Agent 评测
    python -m src.eval.agent_eval --self-test
"""

# ---------------------------------------------------------------------------
# 惰性导出（PEP 562）
#
# 这里故意不写 `from .xxx import yyy`。因为本包下有些模块要 import torch，
# 有些不要。急切导入会让「只想跑 torch-free 模块」的人在本地直接撞
# ModuleNotFoundError —— 纯粹被连坐。
#
# 改成按需加载后：
#     from src.eval import AgentMetrics      # 触发时才 import 对应模块
#     python -m src.eval.<torch-free 模块>   # 本地可跑
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    "AgentMetrics": ".agent_eval",
    "AgentTask": ".agent_eval",
    "DEFAULT_TASKS": ".agent_eval",
    "EvalSample": ".build_domain_eval",
    "Judge": ".judge",
    "MITIGATION_STRATEGIES": ".hallucination",
    "ProbeQuestion": ".hallucination",
    "TaskVerifier": ".agent_eval",
    "analyze": ".error_analysis",
    "build_agent_report": ".agent_eval",
    "build_attribute_probes": ".hallucination",
    "build_eval_set": ".build_domain_eval",
    "build_mitigation_prompt": ".hallucination",
    "build_probes": ".hallucination",
    "build_report": ".run_eval",
    "build_merged_report": ".report",
    "lint_fuzzy": ".report",
    "load_run": ".report",
    "SUITES": ".run_benchmark",
    "load_suite": ".run_benchmark",
    "score_one": ".run_benchmark",
    "stratified_sample": ".quick_eval",
    "generate_pair": ".quick_eval",
    "calibrate": ".judge",
    "check_json_format": ".metrics",
    "check_must_contain": ".metrics",
    "check_must_not_contain": ".metrics",
    "check_over_promise": ".metrics",
    "check_pii_leak": ".metrics",
    "check_refusal": ".metrics",
    "classify": ".error_analysis",
    "compute_hallucination_rate": ".hallucination",
    "evaluate_rules": ".metrics",
    "generate_eval_card": ".build_domain_eval",
    "make_calibration_sheet": ".judge",
    "parse_yes_no": ".hallucination",
    "run_agent_eval": ".agent_eval",
    "run_eval": ".run_eval",
    "run_mitigation_experiment": ".hallucination",
    "semantic_group": ".error_analysis",
    "spearman_correlation": ".judge",
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
    "DEFAULT_TASKS", "AgentMetrics", "AgentTask", "TaskVerifier",
    "build_agent_report", "run_agent_eval",
    "EvalSample", "build_eval_set", "generate_eval_card",
    "analyze", "classify", "semantic_group",
    "MITIGATION_STRATEGIES", "ProbeQuestion", "build_attribute_probes",
    "build_mitigation_prompt", "build_probes", "compute_hallucination_rate",
    "parse_yes_no", "run_mitigation_experiment",
    "Judge", "calibrate", "make_calibration_sheet", "spearman_correlation",
    "check_json_format", "check_must_contain", "check_must_not_contain",
    "check_over_promise", "check_pii_leak", "check_refusal", "evaluate_rules",
    "build_report", "run_eval",
    "build_merged_report", "lint_fuzzy", "load_run",
    "SUITES", "load_suite", "score_one",
    "stratified_sample", "generate_pair",
]
