# Day 16 · 训练日志与踩坑

> 预计 3–4h ｜ 📓 `notebooks/day-16_training_log_debug.ipynb` ｜ ☁️ 云 GPU · `src/train/monitor.py`
> 前置：Day 15（至少跑过 ~200 步）

## 今日目标（一句话）

把 trainer log 变成三联图（loss / lr / grad_norm），**并主动制造一个 bug 再修好** —— 后者才是今天真正的产出。

## 一、读（60 min）

材料：
- `docs/06-sft-training.md` 第 5–7 节（loss 不降的 5 类原因 / 梯度爆炸 / OOM / template 错配）
- `src/train/monitor.py` 的 `diagnose()` —— 每类病因的判据都写在里面

思考题（先自己想，答案在讲义或代码注释里）：
1. loss 不降有 5 类原因，怎么用三条曲线把它们区分开？
2. `grad_norm` 突然出现尖刺（比如从 0.9 跳到 50）说明什么？该做什么？
3. `eval_loss` 开始抬头、`train_loss` 还在降 —— 这是 bug 还是正常现象？

## 二、写（100 min）

`src/train/monitor.py`

| 函数 / 文件 | 你要做什么 |
|---|---|
| `parse_trainer_log` | 从文本日志里正则抽取 loss / lr / grad_norm 三列 |
| `diagnose` | 五类病因判据：loss 平、loss 炸、grad_norm 尖刺、过拟合、lr 调度异常 |
| `plot` | 三联图输出 PNG |
| 你的 bug 实验 | **故意改坏一处**（推荐：把 chat template 换成错的），再诊断出来 |

## 三、跑（在云 GPU 上）

```bash
# 默认读取该目录下的 trainer log 并出图
python -m src/train/monitor outputs/qwen25vl3b-cx-lora-v0
# 确认三联图已生成
ls -la outputs/qwen25vl3b-cx-lora-v0/*.png
```

期望输出（节选）：
```
[parse] 从 trainer_state.json / train.log 解析到 380 个 step
[diagnose]
  loss      : 2.84 → 1.12   趋势正常（缓降，无平台期）
  lr        : warmup 20 步 → 峰值 3e-5 → cosine 衰减   正常
  grad_norm : 均值 0.98，最大 3.21，无尖刺             正常
  结论：这次训练没有明显异常。建议再跑 1 个 epoch 看 eval_loss 拐点。
[plot] 已保存 outputs/qwen25vl3b-cx-lora-v0/train_curves.png
```

## 四、验收清单

- [ ] 三联图已生成，并写了一段**解读**（不是贴图了事）
- [ ] **主动制造并修复了至少 1 个 bug**，能说清「症状 → 判据 → 根因 → 修法」
- [ ] 能指出你这次训练的过拟合拐点大概在第几步（或说明为什么还没到）
- [ ] 知道 `eval_loss` 抬头不一定是 bug —— 要先看 `train_loss` 是否还在降

## 五、容易踩的坑

1. 只盯 loss 不看 lr 调度 —— lr 调错时 loss 的表现和「数据有问题」一模一样。
2. `grad_norm` 长期 > 1 却不处理 —— 加大 warmup 或降 lr，必要时开梯度裁剪。
3. 把 `eval_loss` 上升当成 bug —— 可能只是正常过拟合，先看 train/eval 的差值。
4. 日志没落盘就重启了机器 —— 训练一定重定向到文件（`nohup ... > log 2>&1 &`）。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
