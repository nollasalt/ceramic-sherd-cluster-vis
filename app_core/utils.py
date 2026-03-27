"""应用级通用工具：颜色生成与部位符号映射。"""

import plotly.express as px

# 离散形状序列，用于在散点图中区分陶片部位
PART_SYMBOL_SEQUENCE = [
    'circle', 'square', 'diamond', 'cross', 'x',
    'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right',
    'pentagon', 'hexagon', 'star', 'hexagram', 'star-square', 'star-diamond'
]

_color_cache = {}

def generate_distinct_colors(n_colors: int):
    """生成并缓存 `n_colors` 个可区分的离散颜色。"""
    if n_colors in _color_cache:
        return _color_cache[n_colors]

    base_colors = (
        px.colors.qualitative.Plotly +      # 10 colors
        px.colors.qualitative.D3 +          # 10 colors
        px.colors.qualitative.G10 +         # 10 colors
        px.colors.qualitative.T10 +         # 10 colors
        px.colors.qualitative.Alphabet      # 26 colors
    )

    colors = [base_colors[i % len(base_colors)] for i in range(n_colors)]
    _color_cache[n_colors] = colors
    return colors

# 预生成50种不同的颜色
CLUSTER_COLORS = generate_distinct_colors(50)


def get_part_symbol_settings(dataframe):
    """根据 `part_C/part` 字段生成散点图 symbol 映射。"""
    symbol_col = None
    if 'part_C' in dataframe.columns and dataframe['part_C'].notna().any():
        symbol_col = 'part_C'
    elif 'part' in dataframe.columns and dataframe['part'].notna().any():
        symbol_col = 'part'

    if symbol_col is None:
        return None, {}

    parts = [p for p in dataframe[symbol_col].dropna().unique()]
    parts = sorted(parts, key=lambda x: str(x))
    symbol_map = {p: PART_SYMBOL_SEQUENCE[i % len(PART_SYMBOL_SEQUENCE)] for i, p in enumerate(parts)}

    return symbol_col, symbol_map


# ── 图像视觉特征提取 ──────────────────────────────────────────────────────────

import numpy as np
from pathlib import Path
from PIL import Image
import colorsys

# GLCM功能可选（需要scikit-image）
try:
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def extract_visual_features(image_path):
    """从单张图像提取视觉特征。"""
    try:
        img = Image.open(image_path)

        # 处理透明背景：只计算非透明像素
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img_rgba = img.convert('RGBA')
            img_array = np.array(img_rgba)
            alpha = img_array[:, :, 3]

            # 只保留alpha > 128的像素（非透明部分）
            mask = alpha > 128
            if not mask.any():
                return None

            rgb_array = img_array[:, :, :3]
            mean_rgb = rgb_array[mask].mean(axis=0)

            # 灰度和纹理计算
            gray_array = np.array(img.convert('L'))
            gray_masked = gray_array[mask]
            brightness = float(gray_masked.mean())
            contrast = float(gray_masked.std())

            # 在2D图像上计算梯度，然后只取非透明部分
            gy, gx = np.gradient(gray_array.astype(float))
            edge_magnitude = np.sqrt(gx**2 + gy**2)
            texture_complexity = float(edge_magnitude[mask].mean())

        else:
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            mean_rgb = img_array.mean(axis=(0, 1))

            gray_array = np.array(img.convert('L'))
            brightness = float(gray_array.mean())
            contrast = float(gray_array.std())

            gy, gx = np.gradient(gray_array.astype(float))
            edge_magnitude = np.sqrt(gx**2 + gy**2)
            texture_complexity = float(edge_magnitude.mean())

        r, g, b = mean_rgb / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # GLCM纹理特征（仅在scikit-image可用时计算）
        if HAS_SKIMAGE:
            gray_32 = (gray_array // 8).astype(np.uint8)
            if gray_32.max() > 0:
                glcm = graycomatrix(gray_32, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                   levels=32, symmetric=True, normed=True)
                glcm_contrast = float(graycoprops(glcm, 'contrast').mean())
                glcm_homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
                glcm_energy = float(graycoprops(glcm, 'energy').mean())
                glcm_correlation = float(graycoprops(glcm, 'correlation').mean())
                glcm_entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
            else:
                glcm_contrast = glcm_homogeneity = glcm_energy = glcm_correlation = glcm_entropy = 0.0
        else:
            glcm_contrast = glcm_homogeneity = glcm_energy = glcm_correlation = glcm_entropy = 0.0

        return {
            'mean_rgb': tuple(mean_rgb),
            'mean_hsv': (h * 360, s * 100, v * 100),
            'brightness': brightness,
            'contrast': contrast,
            'texture_complexity': texture_complexity,
            'glcm_contrast': glcm_contrast,
            'glcm_homogeneity': glcm_homogeneity,
            'glcm_energy': glcm_energy,
            'glcm_correlation': glcm_correlation,
            'glcm_entropy': float(glcm_entropy),
        }
    except Exception as e:
        print(f"[ERROR] 提取图像特征失败 {image_path}: {e}")
        return None


def extract_cluster_visual_profile(sample_ids, image_col, search_dirs, max_samples=10):
    """为一组样本提取视觉特征统计。"""
    if len(sample_ids) > max_samples:
        sample_ids = list(np.random.choice(sample_ids, max_samples, replace=False))

    features_list = []
    found_count = 0
    for sid in sample_ids:
        img_name = image_col.get(sid)
        if not img_name:
            continue

        img_path = None
        for search_dir in search_dirs:
            candidate = Path(search_dir) / img_name
            if candidate.exists():
                img_path = candidate
                found_count += 1
                break

        if img_path:
            feat = extract_visual_features(img_path)
            if feat:
                features_list.append(feat)

    print(f"[DEBUG] 尝试读取 {len(sample_ids)} 张图像，找到 {found_count} 张，成功提取 {len(features_list)} 张")

    if not features_list:
        return None

    rgb_vals = np.array([f['mean_rgb'] for f in features_list])
    hsv_vals = np.array([f['mean_hsv'] for f in features_list])
    brightness_vals = np.array([f['brightness'] for f in features_list])
    contrast_vals = np.array([f['contrast'] for f in features_list])
    texture_vals = np.array([f['texture_complexity'] for f in features_list])
    glcm_contrast_vals = np.array([f['glcm_contrast'] for f in features_list])
    glcm_homogeneity_vals = np.array([f['glcm_homogeneity'] for f in features_list])
    glcm_energy_vals = np.array([f['glcm_energy'] for f in features_list])
    glcm_correlation_vals = np.array([f['glcm_correlation'] for f in features_list])
    glcm_entropy_vals = np.array([f['glcm_entropy'] for f in features_list])

    profile = {
        'mean_rgb': tuple(rgb_vals.mean(axis=0)),
        'std_rgb': tuple(rgb_vals.std(axis=0)),
        'mean_hsv': tuple(hsv_vals.mean(axis=0)),
        'std_hsv': tuple(hsv_vals.std(axis=0)),
        'mean_brightness': float(brightness_vals.mean()),
        'std_brightness': float(brightness_vals.std()),
        'mean_contrast': float(contrast_vals.mean()),
        'std_contrast': float(contrast_vals.std()),
        'mean_texture': float(texture_vals.mean()),
        'std_texture': float(texture_vals.std()),
        'mean_glcm_contrast': float(glcm_contrast_vals.mean()),
        'std_glcm_contrast': float(glcm_contrast_vals.std()),
        'mean_glcm_homogeneity': float(glcm_homogeneity_vals.mean()),
        'std_glcm_homogeneity': float(glcm_homogeneity_vals.std()),
        'mean_glcm_energy': float(glcm_energy_vals.mean()),
        'std_glcm_energy': float(glcm_energy_vals.std()),
        'mean_glcm_correlation': float(glcm_correlation_vals.mean()),
        'std_glcm_correlation': float(glcm_correlation_vals.std()),
        'mean_glcm_entropy': float(glcm_entropy_vals.mean()),
        'std_glcm_entropy': float(glcm_entropy_vals.std()),
        'n_samples': len(features_list),
    }

    # 推断装饰技法类型
    if HAS_SKIMAGE:
        profile['decoration_type'] = infer_decoration_type(profile)
    else:
        profile['decoration_type'] = '需要安装scikit-image'

    return profile


def infer_decoration_type(profile):
    """根据GLCM特征推断陶片装饰技法类型。"""
    contrast = profile['mean_glcm_contrast']
    homogeneity = profile['mean_glcm_homogeneity']
    energy = profile['mean_glcm_energy']
    correlation = profile['mean_glcm_correlation']
    entropy = profile['mean_glcm_entropy']

    # 素面：低对比度、高同质性、低熵
    if contrast < 5 and homogeneity > 0.8 and entropy < 3:
        return '素面（光滑表面）'

    # 绳纹：中对比度、高相关性（方向性强）
    if 5 <= contrast <= 15 and correlation > 0.7:
        return '绳纹（平行线条）'

    # 刻划纹：高对比度、低同质性、高熵
    if contrast > 15 and homogeneity < 0.6 and entropy > 4:
        return '刻划纹（深刻不规则）'

    # 篮纹：中高对比度、中等相关性、中高熵
    if 10 <= contrast <= 20 and 0.4 <= correlation <= 0.7 and 3 <= entropy <= 5:
        return '篮纹（交叉编织）'

    # 其他情况
    if contrast < 8:
        return '素面或浅纹'
    elif contrast > 20:
        return '深刻纹饰'
    else:
        return '混合纹饰'


def infer_decoration_from_features(feat):
    """从单个样本的特征推断装饰技法。"""
    if not HAS_SKIMAGE or feat['glcm_contrast'] == 0:
        return '未知'

    contrast = feat['glcm_contrast']
    homogeneity = feat['glcm_homogeneity']
    correlation = feat['glcm_correlation']
    entropy = feat['glcm_entropy']

    if contrast < 5 and homogeneity > 0.8 and entropy < 3:
        return '素面'
    if 5 <= contrast <= 15 and correlation > 0.7:
        return '绳纹'
    if contrast > 15 and homogeneity < 0.6 and entropy > 4:
        return '刻划纹'
    if 10 <= contrast <= 20 and 0.4 <= correlation <= 0.7 and 3 <= entropy <= 5:
        return '篮纹'
    if contrast < 8:
        return '素面或浅纹'
    elif contrast > 20:
        return '深刻纹饰'
    else:
        return '混合纹饰'


def analyze_cluster_feature_distribution(sample_ids, image_col, search_dirs, max_samples=50):
    """分析簇内特征分布，返回每片的特征和统计。"""
    if len(sample_ids) > max_samples:
        sample_ids = list(np.random.choice(sample_ids, max_samples, replace=False))

    features_list = []
    for sid in sample_ids:
        img_name = image_col.get(sid)
        if not img_name:
            continue

        img_path = None
        for search_dir in search_dirs:
            candidate = Path(search_dir) / img_name
            if candidate.exists():
                img_path = candidate
                break

        if img_path:
            feat = extract_visual_features(img_path)
            if feat:
                # 添加颜色分类和装饰技法
                v = feat['mean_hsv'][2]
                if v < 30:
                    feat['color_category'] = '深色'
                elif v < 60:
                    feat['color_category'] = '中等'
                else:
                    feat['color_category'] = '浅色'

                feat['decoration_category'] = infer_decoration_from_features(feat)
                features_list.append(feat)

    if not features_list:
        return None

    # 统计分布
    from collections import Counter
    color_dist = Counter([f['color_category'] for f in features_list])
    decoration_dist = Counter([f['decoration_category'] for f in features_list])

    return {
        'features': features_list,
        'color_distribution': dict(color_dist),
        'decoration_distribution': dict(decoration_dist),
        'n_samples': len(features_list),
    }

