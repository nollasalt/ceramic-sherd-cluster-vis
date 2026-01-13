import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from model import DINOv3_S_Encoder   # 你的模型定义


# =========================
# 配置 - 仅处理正面图像
# =========================
IMAGE_FOLDER = (ROOT_DIR / "all_cutouts").resolve()  # 回到上级目录找图像
OUTPUT_CSV_PATH = (BASE_DIR / "exterior_features_dinov3.csv").resolve()

WEIGHT_PATH = (ROOT_DIR / "dinov3_epoch_100.pth").resolve()   # 预训练模型
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TARGET_SIZE = 224
MARGIN_RATIO = 0.08


# =========================
# 图像预处理（与训练时一致）
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
# 主流程 - 仅处理exterior图像
# =========================
def extract_features_exterior_only():
    print(f"z使用设备: {DEVICE}")
    print(" 仅处理陶片正面(exterior)图像")

    # --- 加载模型 ---
    model = DINOv3_S_Encoder(
        weight_path=WEIGHT_PATH,
        proj_dim=128,
        train_backbone=False   # ⭐ 推理阶段必须 False
    )
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"), strict=True)
    model = model.to(DEVICE)
    model.eval()

    # 获取所有图像文件并筛选出exterior图像
    all_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    
    #  仅保留exterior图像
    exterior_files = [
        f for f in all_files 
        if "exterior" in f.lower()
    ]
    
    print(f" 总图像文件数: {len(all_files)}")
    print(f" Exterior图像文件数: {len(exterior_files)}")
    print(f" 将处理: {len(exterior_files)} 张正面图像")

    all_feats = []
    all_names = []
    skipped_count = 0

    for file in tqdm(exterior_files, desc="提取 DINOv3 特征 (仅正面)"):
        img_path = IMAGE_FOLDER / file

        try:
            image = Image.open(img_path).convert("RGBA")
        except UnidentifiedImageError:
            print(f" 无法识别 {file}，跳过")
            skipped_count += 1
            continue

        processed = crop_and_pad_foreground(image, TARGET_SIZE, MARGIN_RATIO)
        if processed is None:
            print(f" {file} 无前景，跳过")
            skipped_count += 1
            continue

        x = transform(processed).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, z = model(x)     # ⭐ 用 projector 输出
            feat = z.squeeze(0).cpu().numpy()   # [128]

        all_feats.append(feat)
        all_names.append(file)

    if not all_feats:
        print(" 没有成功提取任何特征")
        return

    feats = np.stack(all_feats, axis=0)

    df = pd.DataFrame(feats)
    df.insert(0, "filename", all_names)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n 已提取 {len(all_names)} 张正面图像的 DINOv3 特征")
    print(f" 特征维度: {feats.shape[1]}")
    print(f" 跳过文件数: {skipped_count}")
    print(f" 保存到: {OUTPUT_CSV_PATH}")


# =========================
# 启动
# =========================
if __name__ == "__main__":
    extract_features_exterior_only()