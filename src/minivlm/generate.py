"""
Mini-VLM / Qwen2.5-VL 的推理入口。

对应计划 Day 5（Mini-VLM 出句子）和 Day 1（第一次跑通 Qwen2.5-VL）。

两个入口：
    qwen_infer()   —— 直接调官方 Qwen2.5-VL（Day 1 用）
    minivlm_infer()—— 调你手搭的 Mini-VLM（Day 5 用）

对比两者的输出质量，你会真正体会到「连接器训练过」和「没训练过」的差距：
Mini-VLM 的答案会明显更差甚至胡说，因为它没做过 Stage 0 对齐。
这个对比比任何讲义都有说服力。
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch


# =============================================================================
# 1. 官方 Qwen2.5-VL 推理
# =============================================================================


def load_qwen(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", dtype=torch.bfloat16):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def qwen_infer(
    model,
    processor,
    image,
    question: str,
    system: Optional[str] = None,
    max_new_tokens: int = 256,
    images_first: bool = True,
) -> str:
    """单图问答。

    参数 images_first 控制「图在前」还是「文在前」。
    讲义 docs/01-architecture.md 提到过：顺序会影响效果，值得亲手试一次。
    """
    content = []
    if images_first:
        content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": question})
    else:
        content.append({"type": "text", "text": question})
        content.append({"type": "image", "image": image})

    messages = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({"role": "user", "content": content})

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    n_vis = (inputs["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum().item()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # 评测时用贪心，保证可复现
            temperature=None, top_p=None, top_k=None,
        )

    gen = out[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return answer, n_vis


# =============================================================================
# 2. Mini-VLM 推理
# =============================================================================


def minivlm_infer(model, image, question: str, max_new_tokens: int = 128) -> str:
    """用手搭的 Mini-VLM 生成。

    注意：因为连接器是随机初始化的，输出质量会很差（可能胡言乱语）。
    这正好说明了 Stage 0 对齐训练的必要性 —— 视觉信息和语言空间
    在没有对齐之前，LLM 根本读不懂。

    Day 5 的验收标准只是「能跑出不报错的一句话」，不要求质量。
    """
    from PIL import Image
    import torchvision.transforms as T

    # 预处理（要和视觉塔的预训练一致）
    tfm = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    pixel_values = tfm(image.convert("RGB")).unsqueeze(0).to(model.device)

    prompt = f"<|im_start|>system\n你是一个乐于助人的助手。<|im_end|>\n"
    prompt += f"<|im_start|>user\n{model.image_token}{question}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"

    ids = model.tokenizer(prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        visual = model.encode_images(pixel_values)
        emb = model.llm.get_input_embeddings()(ids["input_ids"])
        emb, mask = model.merge_visual_embeds_expand(
            ids["input_ids"], emb, visual, ids["attention_mask"]
        )
        out = model.llm.generate(
            inputs_embeds=emb,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
        )
    return model.tokenizer.decode(out[0], skip_special_tokens=True)


# =============================================================================
# 3. CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="VLM 推理入口")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", required=True, help="本地图片路径或 URL")
    parser.add_argument("--question", default="请详细描述这张商品图，包括颜色、材质、款式特征。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--compare-order", action="store_true",
                        help="对比「图在前」和「文在前」的效果差异")
    args = parser.parse_args()

    from PIL import Image
    import requests
    from io import BytesIO

    if args.image.startswith("http"):
        img = Image.open(BytesIO(requests.get(args.image, timeout=30).content))
    else:
        img = Image.open(args.image)
    img = img.convert("RGB")
    print(f"图片: {args.image}  尺寸: {img.size}\n")

    model, processor = load_qwen(args.model)

    if args.compare_order:
        print("=" * 70)
        print("顺序对比实验（Day 1 的自检任务之一）")
        print("=" * 70)
        for first in (True, False):
            label = "图在前" if first else "文在前"
            ans, n_vis = qwen_infer(model, processor, img, args.question,
                                    images_first=first,
                                    max_new_tokens=args.max_new_tokens)
            print(f"\n【{label}】visual tokens = {n_vis}")
            print(ans[:500])
    else:
        ans, n_vis = qwen_infer(model, processor, img, args.question,
                                max_new_tokens=args.max_new_tokens)
        print(f"visual tokens: {n_vis}")
        print("-" * 70)
        print(ans)


if __name__ == "__main__":
    main()
