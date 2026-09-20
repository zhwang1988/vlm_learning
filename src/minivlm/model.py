"""
Mini-VLM：把 SigLIP + Connector + Qwen2.5 拼成一个能跑的 VLM。

对应讲义 docs/01-architecture.md，对应计划 Day 5。

这个文件的价值不在「好用」，而在「看清结构」。
真正的 Qwen2.5-VL 有几千行，但骨架和这里一模一样：

    图像 → 视觉编码器 → 连接器 → 替换 input embedding 里的占位符 → LLM

你在这里踩的每一个坑（shape 对不上、label mask 错、chat template 错配），
在训 Qwen2.5-VL 的时候都会以同样的形式再出现一次。
先在便宜的模型上踩完，到真的训练时就顺了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .connector import MLPConnector, build_connector
from .vision import SiglipVisionWrapper, estimate_visual_tokens


@dataclass
class MiniVLMConfig:
    vision_model: str = "google/siglip-base-patch16-224"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    connector_type: str = "mlp"          # linear | mlp | perceiver
    connector_kwargs: dict = None
    freeze_vision: bool = True
    freeze_llm: bool = False
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.connector_kwargs is None:
            self.connector_kwargs = {}


class MiniVLM(nn.Module):
    """最小可运行的 VLM。

    前向流程：
      1. 图像过视觉塔        -> (B, N_patch, D_v)
      2. 连接器投影          -> (B, N_vis, D_llm)
      3. 文本过 embedding     -> (B, L, D_llm)
      4. 把文本 embedding 里 image_token 的位置替换成视觉特征
      5. 送进 LLM
    """

    def __init__(self, config: MiniVLMConfig):
        super().__init__()
        self.config = config
        self.dtype = config.dtype

        # --- 1. 视觉塔 ---
        self.vision = SiglipVisionWrapper(
            config.vision_model,
            frozen=config.freeze_vision,
            dtype=config.dtype,
        )

        # --- 2. 语言模型 ---
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            dtype=config.dtype,
            attn_implementation="sdpa",
        )
        self.d_llm = self.llm.config.hidden_size

        # --- 3. 连接器 ---
        self.connector = build_connector(
            config.connector_type,
            d_vision=self.vision.hidden_size,
            d_llm=self.d_llm,
            **config.connector_kwargs,
        )

        if config.freeze_llm:
            self.llm.requires_grad_(False)

        # 一个特殊的占位 token：文本里写 <image>，我们把它替换成 N 个视觉 token
        # 真实模型（Qwen）用一组连续的 <|image_pad|> token 实现，
        # 这里用单个 token 简化，但替换逻辑完全一致。
        self.image_token = "<image>"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_token]}
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        # 打印结构概览，方便你确认拼装正确
        self._print_summary()

    # ------------------------------------------------------------------ #

    def _print_summary(self):
        n_vis = sum(p.numel() for p in self.vision.parameters())
        n_conn = sum(p.numel() for p in self.connector.parameters())
        n_llm = sum(p.numel() for p in self.llm.parameters())
        total = n_vis + n_conn + n_llm
        print("=" * 66)
        print("MiniVLM 结构概览")
        print("=" * 66)
        print(f"  视觉塔   {self.config.vision_model}")
        print(f"           {n_vis:>14,} 参数  ({n_vis / total:>5.1%})  "
              f"{'[冻结]' if self.config.freeze_vision else '[可训]'}")
        print(f"  连接器   {self.config.connector_type}")
        print(f"           {n_conn:>14,} 参数  ({n_conn / total:>5.1%})  [可训]")
        print(f"  语言主干 {self.config.llm_model}")
        print(f"           {n_llm:>14,} 参数  ({n_llm / total:>5.1%})  "
              f"{'[冻结]' if self.config.freeze_llm else '[可训]'}")
        print(f"  {'总计':<8} {total:>14,} 参数")
        print("=" * 66)
        print("注意：连接器参数占比极小（<1%），但它决定了视觉信息能不能被 LLM 读懂。")
        print("      这就是为什么后训练时它必须可训。")
        print()

    # ------------------------------------------------------------------ #

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """图像 -> 视觉 token 序列。"""
        feats = self.vision(pixel_values)          # (B, N_patch, D_v)
        return self.connector(feats)               # (B, N_vis, D_llm)

    def merge_visual_embeds(
        self,
        input_ids: torch.Tensor,          # (B, L) 含若干 image_token_id
        inputs_embeds: torch.Tensor,      # (B, L, D_llm)
        visual_embeds: torch.Tensor,      # (B, N_vis, D_llm)
    ) -> torch.Tensor:
        """把 image_token 位置的文本嵌入替换成视觉特征。

        **整个 VLM 最关键的一步。**

        实现注意：
          - 我们用「一个 <image> token 展开成 N 个视觉 token」的方式，
            所以不是 1:1 替换，而是要先把序列「撑开」。
          - 真实 Qwen2.5-VL 的做法是：processor 在文本里预先生成 N 个
            <|image_pad|> token，模型只需按位替换（1:1）。
            那个方案更简单，但要求你提前算准 N —— 这就是
            docs/04-qwen25vl.md 里 `compute_visual_tokens` 的作用。

        这里为了让代码可读，用 1:1 的单 token 版本：
          输入里放 1 个 <image>，我们把它替换成「N 个视觉 token 中的第 1 个」，
          然后靠 attention mask 让 LLM 看到全部 —— 这是简化处理，
          严格来说应该展开序列。下面提供了展开版本 `merge_visual_embeds_expand`。
        """
        image_mask = (input_ids == self.image_token_id)     # (B, L)
        n_img_tokens = image_mask.sum().item()
        n_visual = visual_embeds.reshape(-1, visual_embeds.size(-1)).size(0)

        if n_img_tokens * visual_embeds.size(1) != n_visual * 1:
            # 每个 <image> 需要 N_vis 个视觉 token
            expected = n_img_tokens * visual_embeds.size(1)
            if expected != n_visual:
                raise ValueError(
                    f"视觉 token 数与占位符不匹配！\n"
                    f"  文本里 <image> 出现 {n_img_tokens} 次\n"
                    f"  视觉特征共 {n_visual} 个（每张图 {visual_embeds.size(1)} 个）\n"
                    f"  期望 {expected} 个视觉特征\n"
                    f"这是多模态训练最经典的报错。检查：\n"
                    f"  1) images 列表顺序是否与文本中 <image> 顺序一致\n"
                    f"  2) 是否重复 resize 了图片导致特征数变化"
                )

        # 1:1 替换（简化版）：只替换第一个视觉 token
        # 完整版见下面的 expand 函数
        B, L, D = inputs_embeds.shape
        # 把视觉特征按出现顺序排好
        flat_visual = visual_embeds.reshape(-1, D)          # (n_img*N_vis, D)
        first_visual = flat_visual[::visual_embeds.size(1)]  # (n_img, D)

        out = inputs_embeds.clone()
        cursor = 0
        for b in range(B):
            positions = image_mask[b].nonzero(as_tuple=True)[0]
            for pos in positions:
                out[b, pos] = first_visual[cursor].to(out.dtype)
                cursor += 1
        return out

    def merge_visual_embeds_expand(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        visual_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """严格版本：把 <image> 位置展开成 N_vis 个位置。

        返回 (新的 inputs_embeds, 新的 attention_mask)。

        这才是正确做法。上面那个 1:1 版本只是为了让你先理解「替换」这个动作，
        真正训练必须用这个展开版本，否则一批图就吃掉大量信息。
        """
        B, L, D = inputs_embeds.shape
        N_vis = visual_embeds.size(1)
        outputs, masks = [], []

        for b in range(B):
            ids = input_ids[b]
            emb = inputs_embeds[b]
            vis = visual_embeds[b]                    # (N_vis, D)
            img_pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]

            chunks, cur = [], 0
            for k, pos in enumerate(img_pos):
                chunks.append(emb[cur:pos])           # 占位符之前的文本
                chunks.append(vis.to(emb.dtype))      # 换成 N_vis 个视觉 token
                cur = pos + 1
            chunks.append(emb[cur:])                  # 最后一段文本

            new_emb = torch.cat(chunks, dim=0)        # (L + n_img*(N_vis-1), D)
            outputs.append(new_emb)

            if attention_mask is not None:
                am = attention_mask[b]
                m_chunks, cur = [], 0
                for pos in img_pos:
                    m_chunks.append(am[cur:pos])
                    m_chunks.append(torch.ones(N_vis, dtype=am.dtype, device=am.device))
                    cur = pos + 1
                m_chunks.append(am[cur:])
                masks.append(torch.cat(m_chunks, dim=0))

        new_embeds = torch.stack(outputs, dim=0)
        new_mask = torch.stack(masks, dim=0) if masks else None
        return new_embeds, new_mask

    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        expand: bool = True,
    ):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)   # (B, L, D)

        if pixel_values is not None:
            visual_embeds = self.encode_images(pixel_values)         # (B, N_vis, D)
            if expand:
                inputs_embeds, attention_mask = self.merge_visual_embeds_expand(
                    input_ids, inputs_embeds, visual_embeds, attention_mask
                )
                if labels is not None:
                    # 展开后 labels 也要跟着变长，视觉 token 位置不参与 loss
                    B, L, _ = inputs_embeds.shape
                    new_labels = torch.full(
                        (B, L), -100, dtype=labels.dtype, device=labels.device
                    )
                    # 简化处理：assistant 部分的 label 依然有效
                    # 严格实现需要按同样的展开逻辑重排（见 src/train/sft_peft.py）
                    new_labels[:, :labels.size(1)] = labels
                    labels = new_labels
            else:
                inputs_embeds = self.merge_visual_embeds(
                    input_ids, inputs_embeds, visual_embeds
                )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


if __name__ == "__main__":
    print("MiniVLM 结构自检（只构建模型，不下载大权重时请注释掉）\n")

    cfg = MiniVLMConfig(
        vision_model="google/siglip-base-patch16-224",
        llm_model="Qwen/Qwen2.5-0.5B-Instruct",
        connector_type="mlp",
        freeze_vision=True,
        freeze_llm=False,
        dtype=torch.float32,     # CPU 上自检用 fp32
    )
    model = MiniVLM(cfg)

    print("\n" + "=" * 66)
    print("前向 + 替换逻辑自检")
    print("=" * 66)

    tok = model.tokenizer
    text = f"user\n{model.image_token}这是什么材质？\nassistant\n"

    # 模拟一张纯色图
    pixel_values = torch.randn(1, 3, 224, 224)
    enc = tok(text, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        vis = model.encode_images(pixel_values)
        print(f"视觉特征:      {tuple(vis.shape)}   (B, N_vis, D_llm)")
        print(f"序列长度:      {enc['input_ids'].shape[1]}")
        print(f"<image> 出现:  {(enc['input_ids'] == model.image_token_id).sum().item()} 次")

        emb = model.llm.get_input_embeddings()(enc["input_ids"])
        new_emb, new_mask = model.merge_visual_embeds_expand(
            enc["input_ids"], emb, vis, enc["attention_mask"]
        )
        print(f"展开后长度:    {new_emb.shape[1]}"
              f"  (原 {emb.shape[1]} + N_vis-1 = {emb.shape[1] + vis.shape[1] - 1})")
        print(f"展开后 mask:   {tuple(new_mask.shape)}")

        out = model(input_ids=enc["input_ids"],
                    pixel_values=pixel_values,
                    attention_mask=enc["attention_mask"])
        print(f"前向输出 logits: {tuple(out.logits.shape)}")
        print("\n✓ 前向跑通，视觉 token 已进入 LLM 的 attention")
