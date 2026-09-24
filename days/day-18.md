# Day 18 · 第一次训练收官

> 预计 3–4h ｜ 📓 `notebooks/day-18_first_training_wrap.ipynb` ｜ ☁️ 云 GPU · `src/eval/quick_eval.py`（本周新增）、LoRA 合并脚本
> 前置：Day 15–17

## 今日目标（一句话）

合并 LoRA → 导出可独立加载的权重，然后做 **20 条 base vs 你的版本的人工并排抽检** —— 第一次能回答「我训的模型到底有没有变好」。

## 一、读（60 min）

材料：
- `docs/06-sft-training.md` 第 9 节（合并与导出）
- `src/eval/quick_eval.py` —— 今天新增的小工具，先读它怎么挑样本

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么抽检样本必须覆盖 L1–L4 四个难度层？（只挑简单的会自我感觉良好）
2. `merge_and_unload()` 之后，模型的参数量和显存占用变成多少？
3. 如果 20 条里变好的只有「语气更客气」，这算不算成功？

## 二、写（100 min）

合并 + 抽检（工具已给，你要做的是**判断**）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `PeftModel.from_pretrained(...)` | 把 adapter 挂到 base 上 |
| `merge_and_unload()` | 合并成一份可直接加载的权重 |
| `torch_dtype=torch.bfloat16` | 别忘了指定精度，否则默认 fp32 显存翻倍 |
| `quick_eval.py` | 按 4 个难度层分层抽样 N 条，两条并排输出 markdown |
| 你的判断 | 逐条标注「变好 / 变差 / 没变」，并写下理由 |

## 三、跑（在云 GPU 上）

```bash
# 生成 20 条并排对比
python -m src.eval.quick_eval --base Qwen/Qwen2.5-VL-3B-Instruct --adapter outputs/qwen25vl3b-cx-lora-v0 --n 20 --out reports/quick_eval_w3.md
# 打开看结果
head -60 reports/quick_eval_w3.md
```

期望输出（节选）：
```
[load] base = Qwen/Qwen2.5-VL-3B-Instruct
[load] adapter = outputs/qwen25vl3b-cx-lora-v0
[merge] 合并完成，参数量 3.75 B，dtype bfloat16
[sample] 分层抽样 20 条：L1 6 / L2 7 / L3 5 / L4 2
[run] 20/20 完成，耗时 143s，峰值显存 8.9 GB
[out] reports/quick_eval_w3.md

  | # | 难度 | 问题 | base | 你的版本 | 你的判断 |
  |---|---|---|---|---|---|
  | 1 | L1 | 这件有货吗？ | 有的哦~ | 您好，M/L 码现货，S 码 3 天内补 | ? |
  ...
```

## 四、验收清单

- [ ] 合并后的权重能用**一个独立的 `from_pretrained` 直接加载**（不挂 adapter）
- [ ] 20 条并排对比已生成，且**你逐条写了判断**
- [ ] 能指出至少 **3 个变好的 case** 和 **2 个变差的 case**，并给出原因假设
- [ ] `progress/weekly-review.md` 的 W3 段已写；进度表 W3 六天 `[x]`，M3 打卡

## 五、容易踩的坑

1. 忘了 `merge_and_unload()` —— 只保存 adapter 就以为「训练完了」，换个加载方式就报错。
2. 不指定 dtype —— 默认 fp32，3B 模型显存直接翻倍到 12 GB 以上。
3. 抽检全是容易的样本 —— 结论会过于乐观。**必须分层抽**。
4. 只看「变好的」不看「变差的」—— 变差的 case 才是下一轮 DPO 的素材（Day 26 要用）。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。

> **M3 达成条件**：有一份合并后的权重 + 20 条带人工判断的对比 + W3 周复盘。
> 没达成也别急 —— Day 28 的止损线允许你把 W3 的目标降为「跑通流程」。
