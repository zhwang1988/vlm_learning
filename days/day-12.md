# Day 12 · 数据集 v0 交付（M2 验收日）

> 预计 3–4h ｜ 📓 `notebooks/day-12_dataset_card.ipynb` ｜ 💻 `src/data/report.py`
> 今天交付的不是代码，是**一份别人能拿去训模型的数据集**。

## 今日目标（一句话）

生成 DATASET_CARD.md（规模/分布/来源/已知缺陷），推送 HuggingFace，
并完成 W2 周复盘 —— 里程碑 M2 达成。

## 一、读（30 min）

- 复盘 W2 六天打卡
- 找一份公开数据集卡片做参照（如 LLaVA-Instruct 的 model card），看它披露了什么

## 二、写（120 min）

### 1. 数据集卡片 `src/data/report.py` → `generate_card()`

卡片必含字段：
| 字段 | 来源 |
|---|---|
| 总量 / train / eval | Day 11 产出 |
| 意图 × 图像类型分布表 | Day 8 矩阵 vs 实际分布的偏差 |
| 清洗漏斗 | Day 10 的 CleaningReport 数字 |
| 合成成本 | Day 9 打卡里记的钱数 |
| **已知缺陷**（≥3 条，诚实写） | 例如：合成语气偏单一 / 长尾意图样本少 / 全部来自 2 个平台品类 |
| 推荐超参 | batch / lr / epoch 建议（给 W3 的自己看） |

**已知缺陷必须诚实** —— 这不是丢人的部分，是审稿人/面试官最看重的部分。

### 2. 推送 HuggingFace（私有仓库即可）

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-name/cx-sft-v0", private=True, exist_ok=True)
api.upload_folder(folder_path="data/processed/", repo_id="your-name/cx-sft-v0",
                  repo_type="dataset")
```

### 3. W2 周复盘（`progress/weekly-review.md`）

1. W2 最值钱的一个决策是什么？（我的答案备选：按图去重）
2. 数据集最大的风险是什么？W3 训练前要不要补数据？
3. 如果只给你 1k 预算改进这条数据流水线，你花在哪？

## 三、跑（本地）

```bash
python -m src.data.report          # 生成 DATASET_CARD.md
jupyter lab   # notebooks/day-12_dataset_card.ipynb（数据集体检总表）
```

## 四、验收清单（= M2 验收）

- [ ] DATASET_CARD.md 六个字段齐全，已知缺陷 ≥ 3 条且诚实
- [ ] HuggingFace 私有仓库上传成功，卡片同步
- [ ] `data/processed/` 下 train/eval/card 三件套齐全
- [ ] 周复盘写完；进度表 W2 六天 `[x]`；M2 状态改 `[x]`
- [ ] 你能一句话说清：**这个数据集适合训什么、不适合训什么**

## 五、容易踩的坑

- 卡片里的分布数字**从真实 jsonl 统计**，别手抄 Day 8 的计划值 —— 计划 ≠ 实际
- HF 上传前确认 `check_leakage` 再跑一遍（上传后才发现泄漏就尴尬了）
- 别把 `.env`、API key、真实用户截图打进数据集 —— 上传前 `grep` 一遍敏感词

## 六、打卡 + 里程碑

打卡三行 + 「M2 我认为 达成/未达成」。达成的话，明天 Day 13 见 —— 开始花钱训模型了。
