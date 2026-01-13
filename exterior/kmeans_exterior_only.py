import os
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ========= 配置 =========
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

CSV_PATH = (BASE_DIR / "exterior_features_dinov3.csv").resolve()  # 仅正面特征文件
IMAGE_ROOT = (ROOT_DIR / "all_cutouts").resolve()  # 图像文件夹
OUTPUT_FOLDER = (BASE_DIR / "exterior_kmeans_results").resolve()  # 聚类目录

# 在运行前删除旧的聚类目录
if OUTPUT_FOLDER.exists():
    try:
        shutil.rmtree(OUTPUT_FOLDER)
        print(f"已删除旧的聚类目录: {OUTPUT_FOLDER}")
    except Exception as e:
        print(f"删除旧目录时出错: {e}")
        # 继续执行，不终止程序

# 创建新的聚类目录
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========= 1. 读取仅正面特征 =========
df = pd.read_csv(CSV_PATH)

if "filename" not in df.columns:
    raise ValueError("CSV 中必须包含 'filename' 列")

print(f"读取 {len(df)} 条正面图像特征记录")

# 验证所有文件都是exterior
exterior_count = sum(1 for f in df['filename'] if 'exterior' in f.lower())
print(f"其中 exterior 文件数: {exterior_count}")

if exterior_count != len(df):
    print(f" 警告: 发现 {len(df) - exterior_count} 个非exterior文件")

# ========= 2. 提取陶片ID =========
def get_piece_id(filename):
    """提取陶片ID，去掉_exterior后缀"""
    name = os.path.splitext(filename)[0]
    name = name.replace("_exterior", "")
    return name.lower()

df["piece_id"] = df["filename"].apply(get_piece_id)

print(f"唯一陶片数量: {df['piece_id'].nunique()}")

# ========= 3. 特征列 =========
feature_cols = [c for c in df.columns if c not in ["filename", "piece_id"]]
print(f"特征维度: {len(feature_cols)}")

# ========= 4. 准备聚类数据（直接使用正面特征，无需融合） =========
features = df[feature_cols].values
piece_ids = df["piece_id"].values
filenames = df["filename"].values

print(f"聚类特征矩阵形状: {features.shape}")

# ========= 4.5. 标准化特征 =========
print("对特征进行标准化...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print(f"标准化后特征统计: 均值={features_scaled.mean():.6f}, 标准差={features_scaled.std():.6f}")

# ========= 5. 手动设置聚类数 =========
N_CLUSTERS = 40   # 可以根据需要调整

print(f"使用设置的聚类数: {N_CLUSTERS}")

best_k = min(N_CLUSTERS, len(features))

# ========= 6. KMeans 聚类 =========
print(f"\n开始进行 K-Means 聚类 (k={best_k})...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(features_scaled)  # 使用标准化后的特征

# 保存聚类中心
cluster_centers = kmeans.cluster_centers_
print(f"聚类中心形状: {cluster_centers.shape}")

# ========= 7. 计算聚类质量指标 =========
if len(set(labels)) > 1:  # 确保不是所有样本都在一个聚类中
    silhouette_avg = silhouette_score(features_scaled, labels)  # 使用标准化后的特征
    print(f"轮廓系数: {silhouette_avg:.4f}")
else:
    silhouette_avg = 0
    print("所有样本都在一个聚类中，无法计算轮廓系数")

# ========= 8. 聚类统计信息 =========
unique_labels, counts = np.unique(labels, return_counts=True)
print(f"\n聚类分布:")
for label, count in zip(unique_labels, counts):
    print(f"  Cluster {label}: {count} 张图像")

# ========= 9. 选择每个聚类的典型样本 =========
print("\n正在选择每个聚类的典型样本...")
representative_samples = {}

for cluster_id in range(best_k):
    # 获取该聚类的所有样本
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        continue
    
    # 计算每个样本到中心的距离
    cluster_features = features_scaled[cluster_indices]  # 使用标准化后的特征
    center = cluster_centers[cluster_id]
    distances = np.linalg.norm(cluster_features - center, axis=1)
    
    # 找到距离最近的样本
    closest_idx = cluster_indices[np.argmin(distances)]
    representative_samples[cluster_id] = {
        'piece_id': piece_ids[closest_idx],
        'filename': filenames[closest_idx],
        'distance': float(distances[np.argmin(distances)]),
        'index': int(closest_idx)
    }

print(f"已选择 {len(representative_samples)} 个聚类的典型样本")

# ========= 10. 创建输出目录 =========
for cluster_id in range(best_k):
    os.makedirs(os.path.join(OUTPUT_FOLDER, f"cluster_{cluster_id}"), exist_ok=True)

# ========= 11. 按聚类复制图像 =========
print("\n正在复制图像到对应cluster...")

for i, (filename, label) in enumerate(tqdm(zip(filenames, labels), total=len(filenames), desc="复制图像")):
    src = IMAGE_ROOT / filename
    dst = OUTPUT_FOLDER / f"cluster_{label}" / filename
    
    try:
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f" 源文件不存在: {src}")
    except Exception as e:
        print(f" 无法复制 {filename}: {e}")

# ========= 12. 保存聚类元数据 =========
cluster_metadata = {
    'algorithm': 'kmeans',
    'description': 'Clustering using exterior images only',
    'n_clusters': best_k,
    'n_samples': len(features),
    'feature_dim': len(feature_cols),
    'silhouette_score': silhouette_avg,
    'cluster_centers': cluster_centers.tolist(),
    'cluster_distribution': {str(label): int(count) for label, count in zip(unique_labels, counts)},
    'representative_samples': representative_samples,
    'piece_ids': piece_ids.tolist(),
    'filenames': filenames.tolist(),
    'labels': labels.tolist()
}

meta_path = OUTPUT_FOLDER / "cluster_metadata.json"
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(cluster_metadata, f, ensure_ascii=False, indent=2)

print(f"\n 聚类元数据已保存到: {meta_path}")
print(f" 完成！{len(filenames)} 张正面图像已按 {best_k} 个聚类分组")
print(f" 聚类结果保存在: {OUTPUT_FOLDER}")

# ========= 13. 显示每个聚类的典型样本 =========
print(f"\n 各聚类典型样本:")
for cluster_id in sorted(representative_samples.keys()):
    sample = representative_samples[cluster_id]
    print(f"  Cluster {cluster_id}: {sample['filename']} (距离: {sample['distance']:.4f})")

print(f"\n 聚类质量评估:")
print(f"  - 聚类数: {best_k}")
print(f"  - 轮廓系数: {silhouette_avg:.4f} (越高越好, 范围[-1,1])")
print(f"  - 最大聚类: {max(counts)} 张图像")
print(f"  - 最小聚类: {min(counts)} 张图像")
print(f"  - 平均每聚类: {len(features)/best_k:.1f} 张图像")