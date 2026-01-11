import os
import shutil
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# ========= 配置 =========
CSV_PATH = "all_features_dinov3.csv"  # DINOv1 特征文件
IMAGE_ROOT = os.path.abspath("all_cutouts")  # 图像文件夹
OUTPUT_FOLDER = "all_kmeans_new"  # 聚类目录
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========= 1. 读取特征 =========
df = pd.read_csv(CSV_PATH).iloc[:5000].copy()

if "filename" not in df.columns:
    raise ValueError("❌ CSV 中必须包含 'filename' 列")

print(f"✅ 读取 {len(df)} 条特征记录")

# ========= 2. 主编号：只去 _exterior / _interior =========
def get_piece_id(filename):

    name = os.path.splitext(filename)[0]
    name = name.replace("_exterior", "").replace("_interior", "")
    return name.lower()

df["main_id"] = df["filename"].apply(get_piece_id)

dropped_main_ids = []

# ========= 3. 每个 main_id 只保留前两张（认为是正反） =========
def select_two_images(group):
    """
    不管正反，只要 main_id 一致 → 一个陶片
    只保留前两张用于融合
    """
    mid = group["main_id"].iloc[0]

    if len(group) < 2:
        dropped_main_ids.append(mid)

        return pd.DataFrame([])  # 少于两张丢弃

    return group.iloc[:2]  # 只保留前两张

selected_df = (
    df.groupby("main_id", group_keys=False)
      .apply(select_two_images)
      .reset_index(drop=True)
)

for mid in dropped_main_ids:
    print(mid)

print(f"✅ 过滤后剩余陶片数: {len(selected_df['main_id'].unique())}")

# ========= 4. 特征列 =========
feature_cols = [c for c in df.columns if c not in ["filename", "main_id"]]

# ========= 5. 拼接特征（两张图简单相加/平均） =========
def fuse_features(group):
    """
    两张图像 → 特征拼接（concat）
    注意：group 正好 2 张图片
    """
    vec1 = group.iloc[0][feature_cols].values
    vec2 = group.iloc[1][feature_cols].values

    fused = np.concatenate([vec1, vec2], axis=0)

    return pd.Series(fused, name=group["main_id"].iloc[0])
#

merged_features = selected_df.groupby("main_id").apply(fuse_features)

features = np.stack(merged_features.values)
piece_ids = merged_features.index.to_numpy()

print(f"✅ 每件陶片融合后的特征维度: {features.shape}")

# ========= 6. 手动设置聚类数 =========
N_CLUSTERS = 200   # 👈 你想要的聚类数量
N_CLUSTERS = 20

print(f"🎯 使用手动设置的聚类数: {N_CLUSTERS}")

best_k = min(N_CLUSTERS, len(piece_ids))

# ========= 7. KMeans 聚类 =========
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)

# 保存聚类中心
cluster_centers = kmeans.cluster_centers_
print(f"✅ 聚类中心形状: {cluster_centers.shape}")

# ========= 8. 选择每个聚类的典型样本（距离中心最近的样本） =========
print("\n🔍 正在选择每个聚类的典型样本 ...")
representative_samples = {}

for cluster_id in range(best_k):
    # 获取该聚类的所有样本
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        continue
    
    # 计算每个样本到中心的距离
    cluster_features = features[cluster_indices]
    center = cluster_centers[cluster_id]
    distances = np.linalg.norm(cluster_features - center, axis=1)
    
    # 找到距离最近的样本
    closest_idx = cluster_indices[np.argmin(distances)]
    representative_samples[cluster_id] = {
        'piece_id': piece_ids[closest_idx],
        'distance': float(distances[np.argmin(distances)]),
        'index': int(closest_idx)
    }

print(f"✅ 已选择 {len(representative_samples)} 个聚类的典型样本")

# ========= 9. 创建输出目录 =========
for cluster_id in range(best_k):
    os.makedirs(os.path.join(OUTPUT_FOLDER, f"cluster_{cluster_id}"), exist_ok=True)

# ========= 10. 按聚类复制图像 =========
print("\n📁 正在复制图像到对应 cluster ...")

for piece_id, label in tqdm(zip(piece_ids, labels), total=len(piece_ids)):
    files = selected_df[selected_df["main_id"] == piece_id]["filename"].values

    for f in files:
        src = os.path.join(IMAGE_ROOT, f)
        dst = os.path.join(OUTPUT_FOLDER, f"cluster_{label}", f)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"⚠️ 无法复制 {f}: {e}")

# ========= 11. 保存聚类元数据 =========
import json
from pathlib import Path

cluster_metadata = {
    'n_clusters': best_k,
    'features_shape': features.shape,
    'cluster_centers': cluster_centers.tolist(),
    'representative_samples': representative_samples,
    'piece_ids': piece_ids.tolist(),
    'labels': labels.tolist()
}

meta_path = os.path.join(OUTPUT_FOLDER, "cluster_metadata.json")
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(cluster_metadata, f, ensure_ascii=False, indent=2)

print(f"✅ 聚类元数据已保存到: {meta_path}")
print("\n🎉 完成！所有同一陶片的两张图已放在同一类文件夹中。")
