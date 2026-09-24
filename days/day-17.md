# Day 17 · 规模化训练：DeepSpeed / FSDP

> 预计 3–4h ｜ 📓 `notebooks/day-17_deepspeed_fsdp.ipynb` ｜ 💻 本地 · `configs/deepspeed_zero2.json`、`configs/deepspeed_zero3.json`、`configs/README.md`
> 前置：Day 16

## 今日目标（一句话）

搞清 ZeRO-1/2/3 与 FSDP 的取舍，写出两份能直接用的 DeepSpeed 配置；并回答「为什么 VLM 训练里视觉塔通常冻结」。

## 一、读（60 min）

材料：
- `docs/06-sft-training.md` 第 8 节（ZeRO 三阶段、FSDP、多模态特有的不平衡问题）
- `configs/README.md` —— 两份配置的设计取舍都写在这里

思考题（先自己想，答案在讲义或代码注释里）：
1. ZeRO-2 切了梯度和优化器状态，ZeRO-3 连**权重**也切 —— 代价分别是什么？
2. 为什么单卡训练开 ZeRO-3 是纯亏？（通信开销 > 省下的显存）
3. VLM 训练里视觉塔冻结的三个理由是什么？（答案在 docs/06 第 8 节）

## 二、写（100 min）

两份 DeepSpeed 配置（今天不跑训练，是「写配置」日）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `configs/deepspeed_zero2.json` | 切梯度 + 优化器状态，单机多卡的主力选择 |
| `configs/deepspeed_zero3.json` | 连权重也切 —— 为 7B/32B 全参预留，本项目先不用 |
| `bf16` 段 | `"bf16": {"enabled": true}` 必须写明，否则默认可能走 fp16 |
| `zero_optimization` | 注意 `stage3_gather_16bit_weights_on_model_save` 这个坑 |
| `configs/README.md` | 把你的取舍理由写进去（面试时这一页很值钱） |

## 三、跑（本地（无需 GPU））

```bash
# 先验 JSON 合法（DeepSpeed 的解析很严格，多余逗号都会炸）
python -c "import json; [json.load(open(f)) for f in ['configs/deepspeed_zero2.json','configs/deepspeed_zero3.json']]; print('✓ 两份配置 JSON 合法')"
# 单卡其实不需要 ZeRO，先确认「它能不能起来」就够了
deepspeed --num_gpus=1 src/train/sft_peft.py --config configs/sft_lora_3b.yaml --deepspeed configs/deepspeed_zero2.json
```

期望输出（节选）：
```
✓ 两份配置 JSON 合法

（deepspeed 启动后节选）
[INFO] DeepSpeed Flops Profiler 未启用
[INFO] Using /root/.cache/torch_extensions as PyTorch extensions root
[INFO] ZeRO stage 2, offload=False, contiguous_gradients=True
[INFO] 可训练参数 30.4 M，优化器状态已分片
```

## 四、验收清单

- [ ] 两份配置 JSON 合法，且每项都有注释说明为什么这么写
- [ ] 能说清 ZeRO-2 和 ZeRO-3 的取舍（省显存 vs 通信开销）
- [ ] 能说出 VLM 训练里视觉塔冻结的理由，以及「冻结 ≠ 不占显存」
- [ ] 知道 `stage3_gather_16bit_weights_on_model_save` 是 ZeRO-3 存模型时的必踩坑

## 五、容易踩的坑

1. JSON 里写了注释 —— DeepSpeed 用严格 JSON 解析器，注释和尾逗号都会报错（注释只能写在 README 里）。
2. ZeRO-3 + LoRA 忘了 `stage3_gather_16bit_weights_on_model_save: true` —— 存出来的权重加载不了。
3. CPU offload 看着很香 —— 实际拖慢 2–3 倍，本项目在 24 GB 卡上完全不需要。
4. 单卡开 ZeRO-3 —— 纯亏，通信没有收益。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
