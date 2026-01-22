import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import igraph as ig
import leidenalg

# =========================================================
# 配置（偏“粗聚类”的安全参数）
# =========================================================
CSV_PATH = "all_features_dinov3.csv"
IMAGE_ROOT = os.path.abspath("all_cutouts")
OUTPUT_FOLDER = "all_leiden_init"

TOPK = 30                  # Mutual kNN
# SECOND_ORDER_WEIGHT = 0.15  # 二阶扩散（保证传递）
# RESOLUTION = 1.8           # ⭐ Leiden 分辨率（越小越粗，0.4~0.8 推荐）

SECOND_ORDER_WEIGHT = 0.05  # 二阶扩散（保证传递）
RESOLUTION = 2.5           # ⭐ Leiden 分辨率（越小越粗，0.4~0.8 推荐）

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# 1. 读取特征
# =========================================================
df = pd.read_csv(CSV_PATH)
if "filename" not in df.columns:
    raise ValueError("CSV 中必须包含 'filename' 列")

print(f"✅ 读取 {len(df)} 条特征记录")

# =========================================================
# 2. 主编号（合并 exterior / interior）
# =========================================================
def get_piece_id(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace("_exterior", "").replace("_interior", "")
    return name.lower()

df["main_id"] = df["filename"].apply(get_piece_id)

# =========================================================
# 3. 每个陶片只保留两张（正反）
# =========================================================
def select_two_images(group):
    if len(group) < 2:
        return pd.DataFrame([])
    return group.iloc[:2]

selected_df = (
    df.groupby("main_id", group_keys=False)
      .apply(select_two_images)
      .reset_index(drop=True)
)

num_pieces = selected_df["main_id"].nunique()
print(f"✅ 过滤后陶片数: {num_pieces}")

# =========================================================
# 4. 特征列
# =========================================================
feature_cols = [c for c in df.columns if c not in ["filename", "main_id"]]

# =========================================================
# 5. 正反面特征融合（拼接，保留判别性）
# =========================================================
def fuse_features(group):
    v1 = group.iloc[0][feature_cols].values
    v2 = group.iloc[1][feature_cols].values
    fused = np.concatenate([v1,v2])
    return pd.Series(fused, name=group["main_id"].iloc[0])

merged_features = (
    selected_df.groupby("main_id", group_keys=False)
    .apply(fuse_features)
)

features = np.stack(merged_features.values).astype(np.float32)
features = StandardScaler().fit_transform(features)

piece_ids = merged_features.index.to_numpy()
n = features.shape[0]

print(f"✅ 融合后特征矩阵: {features.shape}")

# =========================================================
# 6. Mutual kNN 图（连续权重）
# =========================================================
print("🔗 构建 Mutual kNN 相似度图...")

sim = cosine_similarity(features)
np.fill_diagonal(sim, 0)

neighbors = np.argsort(sim, axis=1)[:, -TOPK:]

rows, cols, data = [], [], []

for i in tqdm(range(n), desc="Building mutual kNN graph"):
    for j in neighbors[i]:
        if i in neighbors[j]:
            w = sim[i, j] ** 2   # 抑制弱边
            rows.append(i)
            cols.append(j)
            data.append(w)

A = sp.csr_matrix(
    (data, (rows, cols)),
    shape=(n, n),
    dtype=np.float32
)

print(f"✅ 一阶图：节点 {n}，边数 {A.nnz}")

# =========================================================
# 7. 二阶扩散（增强传递性，但不过度）
# =========================================================
print("🌊 二阶扩散增强...")

A2 = A @ A
A2 = A2.multiply(SECOND_ORDER_WEIGHT)

# 轻量剪枝，防止爆炸
A2.data[A2.data < 0.01] = 0
A2.eliminate_zeros()

A_final = A + A2
A_final.eliminate_zeros()

print(f"✅ 最终图：边数 {A_final.nnz}")

# =========================================================
# 8. Leiden 社区发现（初始粗聚类）
# =========================================================
print("🧠 执行 Leiden 初始聚类...")

edges = list(zip(A_final.nonzero()[0], A_final.nonzero()[1]))
weights = A_final.data.tolist()

g = ig.Graph(n=n, edges=edges, directed=False)
g.es["weight"] = weights

partition = leidenalg.find_partition(
    g,
    leidenalg.RBConfigurationVertexPartition,
    weights="weight",
    resolution_parameter=RESOLUTION
)

clusters = partition
print(f"🎯 Leiden 得到 {len(clusters)} 个初始聚类")

# =========================================================
# 9. piece_id → cluster_id
# =========================================================
piece_to_cluster = {}
for cid, cluster in enumerate(clusters):
    for node_idx in cluster:
        piece_to_cluster[piece_ids[node_idx]] = cid

# =========================================================
# 10. 创建输出目录
# =========================================================
for cid in range(len(clusters)):
    os.makedirs(os.path.join(OUTPUT_FOLDER, f"cluster_{cid}"), exist_ok=True)

# =========================================================
# 11. 按聚类复制图像
# =========================================================
print("📁 正在复制图像到对应 cluster 文件夹...")

for piece_id, cluster_id in tqdm(piece_to_cluster.items()):
    files = selected_df[selected_df["main_id"] == piece_id]["filename"].values
    for f in files:
        src = os.path.join(IMAGE_ROOT, f)
        dst = os.path.join(OUTPUT_FOLDER, f"cluster_{cluster_id}", f)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"⚠️ 无法复制 {f}: {e}")

print("🎉 Leiden 初始聚类完成！")
