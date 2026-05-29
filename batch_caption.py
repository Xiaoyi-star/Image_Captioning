from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          MarianMTModel, MarianTokenizer)
from PIL import Image
import torch, os

# 1. 加 载 所 有 模 型
print("加 载 模 型 中， 请 稍 候...")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")
blip.eval()

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
print("模 型 加 载 完 成！ \n")


def generate_caption(image):
    """生 成 英 文 描 述"""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)


def translate(text):
    """英 文 翻 译 为 中 文"""
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = translator.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# 2. 批 量 处 理 images/ 目 录 下 的 所 有 图 片
img_dir = "images"
results = []
exts = {".jpg", ".jpeg", ".png", ".bmp"}

img_files = [f for f in os.listdir(img_dir)
if os.path.splitext(f)[1].lower() in exts]

print(f"共 找 到 {len(img_files)} 张 图 片， 开 始 处 理...\n")

for fname in img_files:
    img_path = os.path.join(img_dir, fname)
    image = Image.open(img_path).convert("RGB")

    caption_en = generate_caption(image)
    caption_zh = translate(caption_en)

    print(f"[{fname}]")
    print(f"英 文： {caption_en}")
    print(f"中 文： {caption_zh}\n")
    results.append((fname, caption_en, caption_zh))

#  3. 保 存 结 果 到 文 本 文 件
with open("results.txt", "w", encoding="utf-8") as f:
    for fname, en, zh in results:
        f.write(f"文 件： {fname}\n")
        f.write(f"英 文： {en}\n")
        f.write(f"中 文： {zh}\n")
        f.write("-" * 40 + "\n")

print(f"全 部 完 成！ 结 果 已 保 存 至 results.txt")