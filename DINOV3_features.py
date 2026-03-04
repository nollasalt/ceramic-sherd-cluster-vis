"""提取陶片图像的 DINOv3 特征并导出 CSV。"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchvision import transforms

from model import DINOv3_S_Encoder   # 你的模型定义


# =========================
# 配置
# =========================
IMAGE_FOLDER = os.path.abspath("all_cutouts")
OUTPUT_CSV_PATH = "all_features_dinov3.csv"

WEIGHT_PATH = "dinov3_epoch_100.pth"   # ⭐ 你训练好的模型
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TARGET_SIZE = 224
MARGIN_RATIO = 0.08


# =========================
# 图像预处理（与你训练时一致）
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# =========================
# 前景裁剪 + padding
# =========================
def crop_and_pad_foreground(image_rgba, target_size=224, margin_ratio=0.08):
    """裁剪前景区域并按目标尺寸进行等比缩放与居中填充。

    Args:
        image_rgba: RGBA 格式输入图像。
        target_size: 目标正方形尺寸。
        margin_ratio: 前景边界外扩比例。

    Returns:
        PIL.Image | None: 处理后的 RGB 图像；若无有效前景则返回 None。
    """
    rgba = np.array(image_rgba)
    alpha = rgba[:, :, 3] / 255.0
    mask = alpha > 0

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    H, W = alpha.shape
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    pad = int(margin_ratio * max(x_max - x_min, y_max - y_min))
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(W - 1, x_max + pad)
    y_max = min(H - 1, y_max + pad)

    rgb = rgba[y_min:y_max+1, x_min:x_max+1, :3].astype(np.float32)
    alpha_crop = alpha[y_min:y_max+1, x_min:x_max+1]

    rgb *= alpha_crop[..., None]

    pil = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))

    w, h = pil.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = pil.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized, (left, top))

    return canvas


# =========================
# 主流程
# =========================
def extract_features():
    """批量提取图像特征并保存到 `OUTPUT_CSV_PATH`。

    流程包括：模型加载、图像预处理、前向推理、特征汇总与 CSV 导出。
    """
    print(f"✅ 使用设备: {DEVICE}")

    # --- 加载模型 ---
    model = DINOv3_S_Encoder(
        weight_path=WEIGHT_PATH,
        proj_dim=128,
        train_backbone=False   # ⭐ 推理阶段必须 False
    )
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"), strict=True)
    model = model.to(DEVICE)
    model.eval()

    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    all_feats = []
    all_names = []

    for file in tqdm(image_files, desc="提取 DINOv3 特征"):
        img_path = os.path.join(IMAGE_FOLDER, file)

        try:
            image = Image.open(img_path).convert("RGBA")
        except UnidentifiedImageError:
            print(f"⚠️ 无法识别 {file}，跳过")
            continue

        processed = crop_and_pad_foreground(image, TARGET_SIZE, MARGIN_RATIO)
        if processed is None:
            print(f"⚠️ {file} 无前景，跳过")
            continue

        x = transform(processed).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, z = model(x)     # ⭐ 用 projector 输出
            feat = z.squeeze(0).cpu().numpy()   # [128]

        all_feats.append(feat)
        all_names.append(file)

    if not all_feats:
        print("❌ 没有成功提取任何特征")
        return

    feats = np.stack(all_feats, axis=0)

    df = pd.DataFrame(feats)
    df.insert(0, "filename", all_names)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n✅ 已提取 {len(all_names)} 张图像的 DINOv3 特征")
    print(f"📄 特征维度: {feats.shape[1]}")
    print(f"💾 保存到: {OUTPUT_CSV_PATH}")


# =========================
# 启动
# =========================
if __name__ == "__main__":
    extract_features()
