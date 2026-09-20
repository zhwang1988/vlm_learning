"""
Agent 演示界面（Gradio）。

对应计划 Day 35。产出「一个能录屏 demo 的界面」。

两个模式：
  1. Agent 模式：完整的多模态 Agent（工具 + RAG + 循环）
  2. 对比模式：并排展示 基座模型 vs 微调模型 的回复（Day 46 用）

启动：
    pip install gradio
    python -m src.agent.demo
    python -m src.agent.demo --compare --base Qwen/Qwen2.5-VL-3B-Instruct --adapter outputs/xxx
"""

from __future__ import annotations

import asyncio
import json
import os


def build_agent_demo():
    """Agent 演示界面。"""
    import gradio as gr

    from .agent import CXAgent

    agent = CXAgent()

    def respond(message, image, history, session_id, show_trace):
        imgs = [image] if image is not None else []
        out = agent.run(message, imgs,
                        session_id=session_id or "s_demo_1",
                        history=[{"role": h[0] and "user" or "assistant",
                                  "content": h[1]} for h in (history or [])
                                 if h and h[1]])

        reply = out["reply"]
        if show_trace:
            trace = []
            if out.get("thought"):
                trace.append(f"**推理**: {out['thought']}")
            if out.get("visual_evidence"):
                trace.append(f"**视觉证据**: `{json.dumps(out['visual_evidence'], ensure_ascii=False)}`")
            if out.get("tool_calls"):
                trace.append("**工具调用**:")
                for tc in out["tool_calls"]:
                    mark = "✓" if tc.get("success") else "✗"
                    trace.append(f"  - {mark} `{tc['action']}` "
                                 f"{json.dumps(tc['action_input'], ensure_ascii=False)}")
            if out.get("notes"):
                trace.append(f"**备注**: {out['notes']}")
            trace.append(f"**Token**: {out['tokens_in']} in / {out['tokens_out']} out "
                         f"| **步数**: {out['n_steps']}")
            reply = reply + "\n\n---\n" + "\n".join(trace)

        history = history or []
        history.append((message, reply))
        return history, ""

    with gr.Blocks(title="多模态客服 Agent Demo") as demo:
        gr.Markdown("## 多模态客服 Agent\n"
                    "上传商品图 + 提问，Agent 会自主决定是否调用工具。\n\n"
                    "**试试这些**：\n"
                    "- 「我的快递到哪了」（纯文本，需查订单）\n"
                    "- 上传一张衣服图 + 「这个有 M 码吗」（需查库存）\n"
                    "- 上传一张有瑕疵的图 + 「这个算质量问题吗」（纯视觉推理）\n"
                    "- 「我要退货」（有副作用，Agent 应先确认）\n"
                    "- 「你们这是欺诈！」（应转人工）")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=480, label="对话")
                with gr.Row():
                    msg = gr.Textbox(placeholder="输入问题...", scale=4,
                                     label="", show_label=False)
                    img = gr.Image(type="pil", label="上传图片", scale=1)
                with gr.Row():
                    session = gr.Textbox(value="s_demo_1", label="会话 ID", scale=2)
                    trace_cb = gr.Checkbox(value=True, label="显示推理过程", scale=1)
                    clear = gr.Button("清空", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 可用的测试会话\n"
                            "- `s_demo_1` 有一单已发货（顺丰）\n"
                            "- `s_demo_2` 有一单已签收（在退货期内）\n"
                            "- 其他 ID 无订单，应触发追问\n\n"
                            "### 演示要点\n"
                            "1. 该查工具时查工具\n"
                            "2. 该追问时追问（不硬答）\n"
                            "3. 有副作用的操作先确认\n"
                            "4. 投诉类转人工")

        msg.submit(respond, [msg, img, chatbot, session, trace_cb],
                   [chatbot, msg])
        clear.click(lambda: (None, None), None, [chatbot, img])

    return demo


def build_compare_demo(base_model: str, adapter: str, max_pixels: int = 1280):
    """并排对比：基座 vs 微调。Day 46 的消融实验用。"""
    import gradio as gr

    from ..eval.run_eval import LocalVLM

    print(f"加载基座: {base_model}")
    base_engine = LocalVLM(base_model, None, max_pixels)
    print(f"加载微调: {adapter}")
    tuned_engine = LocalVLM(base_model, adapter, max_pixels)

    system = ("你是一位专业的电商客服助手。请基于用户提供的图片和文字，"
              "给出准确、有帮助、语气自然的回复。")

    def compare(message, image):
        import tempfile
        from pathlib import Path

        paths = []
        if image is not None:
            p = Path(tempfile.gettempdir()) / f"cmp_{abs(hash(str(message))) % 10**8}.jpg"
            image.convert("RGB").save(p, quality=90)
            paths = [str(p)]

        out = {}
        for name, eng in [("基座模型", base_engine), ("微调模型", tuned_engine)]:
            try:
                ans, ms = eng.generate(paths, message, system)
                out[name] = f"{ans}\n\n*({ms:.0f}ms)*"
            except Exception as e:      # noqa: BLE001
                out[name] = f"推理失败: {e}"

        return out.get("基座模型", ""), out.get("微调模型", "")

    with gr.Blocks(title="基座 vs 微调对比") as demo:
        gr.Markdown("## 基座 vs 微调 对比\n"
                    "同一个问题、同一张图，看两个模型的差别。\n\n"
                    "**观察重点**：追问行为、语气自然度、是否编造、格式合规")
        with gr.Row():
            img = gr.Image(type="pil", label="图片")
            msg = gr.Textbox(label="问题",
                             value="这个有 M 码吗？")
        btn = gr.Button("对比", variant="primary")
        with gr.Row():
            a = gr.Textbox(label="基座模型", lines=12)
            b = gr.Textbox(label="微调模型", lines=12)
        btn.click(compare, [msg, img], [a, b])
        msg.submit(compare, [msg, img], [a, b])

    return demo


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Agent / 对比 Demo")
    ap.add_argument("--compare", action="store_true", help="跑基座 vs 微调对比")
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen25vl3b-cx-lora-v0")
    ap.add_argument("--share", action="store_true", help="生成公网链接")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if args.compare:
        d = build_compare_demo(args.base, args.adapter)
    else:
        d = build_agent_demo()

    d.launch(server_port=args.port, share=args.share)
