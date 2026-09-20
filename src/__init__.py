"""
multimodal-lab —— 多模态客服模型全流程代码库

    src/
    ├── minivlm/   从零搭一个 VLM（学习用）           W1  Day 2-7
    ├── data/      数据合成 / 清洗 / 打包              W2  Day 8-13
    ├── train/     LoRA / QLoRA / DPO / 奖励           W3 W5  Day 14-18, 27-29
    ├── eval/      评测 / 归因 / 幻觉探测               W4  Day 20-24
    ├── serve/     vLLM + FastAPI 部署                 W5  Day 25-26
    ├── agent/     多模态客服 Agent                    W6  Day 31-38
    └── shopify/   Shopify App                         W7  Day 39-45

每个模块的 __init__.py 里都写清了「这个模块里文件之间的调用顺序」，
拿不准从哪读起就看那个文件。

--------------------------------------------------------------------------
零、这个仓库的设计原则
--------------------------------------------------------------------------

1. **能跑的代码优先于能读的代码，能读的代码优先于文档。**
   所以关键实现都写在 .py 里而不是 notebook 里，而且都带 `__main__` 自检。
   跑一遍 `make smoke` 比读十页文档有用。

2. **每个「反直觉的点」都在代码注释里解释为什么。**
   比如为什么 connector 一定要训、为什么 label mask 要包含 <|im_end|>、
   为什么 webhook 必须先返回 200 再处理业务。这些光看代码看不出来。

3. **故障模式写进代码，不只写进文档。**
   比如 `merge_visual_embeds` 在 token 数不匹配时会给出可操作的报错，
   而不是一个 `RuntimeError: shape mismatch`。

--------------------------------------------------------------------------
一、按周读代码的顺序
--------------------------------------------------------------------------

W1 架构       src/minivlm/vision.py -> connector.py -> model.py -> processor.py
              （从视觉编码器往 LLM 走，顺着数据流读）

W2 数据       src/data/taxonomy.py -> synth.py -> image_utils.py -> dedup.py
              -> build_sft.py -> report.py
              （从配比设计开始，到打包成训练格式结束）

W3 SFT        src/train/lora_utils.py -> sft_peft.py -> monitor.py
              （先搞清 LoRA 加在哪一层，再看训练循环）

W4 评测       src/eval/build_domain_eval.py -> metrics.py -> judge.py
              -> hallucination.py -> run_eval.py -> error_analysis.py
              （先造评测集，再造指标，最后跑）

W5 对齐+推理   src/train/dpo_loss.py -> rewards.py
              src/serve/quantize.py -> api.py -> vllm_server.sh

W6 Agent      src/agent/tools.py -> retriever.py -> agent.py -> demo.py
              （先把工具做好，再做检索，最后拼循环）

W7 产品       src/shopify/auth.py -> client.py -> indexer.py -> webhooks.py
              -> billing.py -> app.py -> models.py
              （认证 -> 取数 -> 建索引 -> 收事件 -> 计费 -> 拼 app）

--------------------------------------------------------------------------
二、常见困惑
--------------------------------------------------------------------------

Q: 为什么既用 LLaMA-Factory 又自己写训练脚本？
A: LLaMA-Factory 快、稳、功能多，正式跑训练用它。
   但它是黑盒 —— 你不知道 label mask 怎么做的、image_grid_thw 是什么。
   自己写的 sft_peft.py 就是为了把这几件事摊开给你看。
   两者产出的 adapter 都是 PEFT 格式，可以互相接。

Q: MOCK_ 开头的数据是真的吗？
A: 不是。src/agent/tools.py 和 src/shopify/client.py 里的 MOCK_ORDER 之类都是
   造的假数据，目的是让你在没有真实店铺的情况下也能把链路跑通。
   Day 39 之后换成真实 API（代码里已经留好了替换点）。

Q: 为什么有这么多 scripts/？
A: scripts/ 是「操作」，src/ 是「实现」。
   scripts/ 里的东西是你每周都会用到的（体检、估显存、同步、起服务），
   值得给它们短名字和友好输出。

--------------------------------------------------------------------------
三、最短上手路径
--------------------------------------------------------------------------

    make check                 # 环境体检
    make data-plan             # 看数据长什么样
    make train-dry             # 数据格式检查
    make smoke                 # 所有模块自检

详细日程见 PLAN.md。
"""

__version__ = "0.1.0"
