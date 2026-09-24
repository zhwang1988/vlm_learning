# Day 29 · 量化与推理加速

> 预计 3–4h ｜ 📓 `notebooks/day-29_quantize_accelerate.ipynb` ｜ ☁️ 云 GPU · `src/serve/quantize.py`、`src/serve/vllm_server.sh`
> 前置：Day 27（DPO 版本已产出）

## 今日目标（一句话）

对 SFT/DPO 后的模型做 AWQ 量化，量出**显存 / 首 token 延迟 / 吞吐**三项指标的前后对比；并解释为什么量化对 VLM 的**视觉塔**尤其敏感。

## 一、读（60 min）

材料：
- `docs/09-inference.md` 第 1–4 节（AWQ / GPTQ / FP8 / KV cache / continuous batching）
- `src/serve/quantize.py` 的 `VISION_EXCLUDE` —— 这是今天的核心知识点

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么 `VISION_EXCLUDE = ["visual", "vision_tower", "merger"]`？量化了会怎样？
2. calibration 数据如果全是纯文本，量化后的模型看图会出什么问题？
3. AWQ 和 GPTQ 怎么选？（提示：谁对激活值更敏感）

## 二、写（100 min）

`src/serve/quantize.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `VISION_EXCLUDE` | **视觉塔和 merger 排除在量化之外** —— 混合精度量化 |
| `estimate_savings()` | 只算账不真跑，先看值不值得 |
| `build_quant_config()` | 生成 AWQ 配置（含 exclude 列表） |
| `benchmark()` | 三项指标：显存 / 首 token 延迟 / tok/s |
| `src/serve/vllm_server.sh` | 起服务时的坑：`--limit-mm-per-prompt`、别开 128k max-len |

## 三、跑（在云 GPU 上）

```bash
# 先只算账（快）
python -m src.serve.quantize --model outputs/qwen25vl3b-cx-dpo-v0 --estimate
# 真跑 AWQ 量化
python -m src.serve.quantize --model outputs/qwen25vl3b-cx-dpo-v0 --method awq --out outputs/quantized/awq
# 跑基准测试
python -m src.serve.quantize --model outputs/quantized/awq --benchmark --n 20 --concurrency 4
# 起 vLLM 服务
bash src/serve/vllm_server.sh
```

期望输出（节选）：
```
[estimate] 原始显存 ≈ 6.4 GB（bf16）→ 量化后 ≈ 2.1 GB，省 67%
           视觉塔保持 bf16，约 0.6 GB 不参与量化
[quantize] calib 数据 128 条图文样本（**必须含图**）
[benchmark]
                      显存      首 token    吞吐
  bf16              6.4 GB     0.42 s    38 tok/s
  awq (混合精度)     2.1 GB     0.31 s    61 tok/s
  awq (全量化, 反例)  1.6 GB     0.29 s    64 tok/s   ← 但看图能力崩了

⚠️  质量警告：视觉塔被量化后，OCR / 细粒度识别准确率下降 20%+
```

## 四、验收清单

- [ ] 三项指标（显存 / 首 token 延迟 / 吞吐）的前后对比表已产出
- [ ] 能解释为什么视觉塔要保持高精度（激活值分布 + 细粒度信息）
- [ ] calibration 数据里**必须含图** —— 且你能说出为什么
- [ ] vLLM 服务能用 `curl` 打通，知道 `--limit-mm-per-prompt` 是干嘛的

## 五、容易踩的坑

1. **把 vision tower 一起量化了** —— 图片识别能力直接崩，而且你在文本测试里发现不了。必须专门跑一次图文评测。
2. calibration 数据全是纯文本 —— 量化参数完全没见到图像激活分布，视觉通路误差被放大。
3. 忘了 `--limit-mm-per-prompt` —— 有人传 10 张图进来，显存瞬间爆掉。
4. 沿用 128k 的 max-len —— KV cache 把显存吃光。客服场景 4k 足够。
5. 只测吞吐不测质量 —— 量化后的模型必须在 `cx_eval_v1` 上再跑一遍，分数掉超过 3 分就要考虑放弃量化。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
