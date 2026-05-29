from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


# 1. 加 载 模 型 （首 次 运 行 自 动 下 载， 约 900MB； 之 后 从 缓 存 读 取）
print("正 在 加 载BLIP模 型， 请 稍 候...")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model.eval()
print("模 型 加 载 完 成！ ")

# 2. 读 取 图 片
img_path = "images/street.jpg"  # 替 换 为 你 的 图 片 路 径
image = Image.open(img_path).convert("RGB")

# 3. 无 条 件 生 成： 让 模 型 自 由 描 述 图 片 内 容
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
caption_en = processor.decode(output[0], skip_special_tokens=True)
print(f"英 文 描 述： {caption_en}")

# 4. 条 件 生 成： 给 定 提 示 词， 让 模 型 补 全 描 述
prompt = "a photo of"
inputs_cond = processor(image, prompt, return_tensors="pt")
with torch.no_grad():
    output_cond = model.generate(**inputs_cond, max_new_tokens=50)
caption_cond = processor.decode(output_cond[0], skip_special_tokens=True)
print(f"条 件 描 述： {caption_cond}")

# 5. 翻 译 为 中 文
# 使 用 Helsinki -NLP 的 英 译 中 模 型 （首 次 运 行 自 动 下 载， 约 300MB）
from transformers import MarianMTModel, MarianTokenizer

print("正 在 加 载 翻 译 模 型...")
trans_model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(trans_model_name)
trans_model = MarianMTModel.from_pretrained(trans_model_name)

inputs_zh = tokenizer(caption_en, return_tensors="pt", padding=True)
translated = trans_model.generate(**inputs_zh)
caption_zh = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"中 文 翻 译： {caption_zh}")

# 6. 可 视 化 并 保 存 结 果
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(image)
ax.axis("off")
ax.set_title(
    f"英 文： {caption_en}\n中 文： {caption_zh}",
    fontsize=11, pad=10
)
plt.tight_layout()
plt.savefig("caption_result.jpg", dpi=150, bbox_inches="tight")
plt.show()
print("结 果 已 保 存 至 caption_result.jpg")