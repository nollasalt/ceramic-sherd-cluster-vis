"""构建聚类结果总表并合并陶片元信息。

从 cluster_metadata.json 读取 piece_to_cluster 映射，
不依赖目录结构，无需复制图片文件。
"""

import json
import os
import pandas as pd

FEATURE_CSV = "all_features_dinov3.csv"
CLUSTER_METADATA = "all_kmeans_new/cluster_metadata.json"
INFO_CSV = "jd_sherds_info.csv"
OUTPUT_CSV = "sherd_cluster_table_clustered_only.csv"
IMAGE_ROOT = "all_cutouts"

# ========================
# 1. 读取聚类元数据
# ========================
with open(CLUSTER_METADATA, "r", encoding="utf-8") as f:
    metadata = json.load(f)

piece_to_cluster = metadata.get("piece_to_cluster", {})
print(f"从 {CLUSTER_METADATA} 读取到 {len(piece_to_cluster)} 个 piece 的聚类结果")

# ========================
# 2. 读取特征 CSV
# ========================
df = pd.read_csv(FEATURE_CSV)
df = df.rename(columns={"filename": "image_name"})

# ========================
# 3. 解析 sample_id / side / main_id
# ========================
def parse_filename(name):
    """从图像文件名解析 sample_id、side 和 main_id。"""
    stem = os.path.splitext(os.path.basename(name))[0]
    parts = stem.split("_")
    sample_id = parts[0]
    side = parts[1] if len(parts) > 1 else "unknown"
    main_id = stem.replace("_exterior", "").replace("_interior", "").lower()
    return sample_id, side, main_id

parsed = df["image_name"].apply(lambda x: pd.Series(parse_filename(x),
                                                      index=["sample_id", "side", "main_id"]))
df = pd.concat([df, parsed], axis=1)

# ========================
# 4. 映射 cluster_id
# ========================
df["cluster_id"] = df["main_id"].map(piece_to_cluster)
df = df.dropna(subset=["cluster_id"]).copy()
df["cluster_id"] = df["cluster_id"].astype(int)
print(f"映射后有效行数: {len(df)}")

# ========================
# 5. image_path 指向原始目录
# ========================
df["image_path"] = df["image_name"].apply(
    lambda name: os.path.join(IMAGE_ROOT, name)
)
df = df.drop(columns=["main_id"])

# ========================
# 6. 合并 jd_sherds_info.csv（如果存在）
# ========================
if os.path.exists(INFO_CSV):
    print(f"Loading info from: {INFO_CSV}")
    try:
        info_df = pd.read_csv(INFO_CSV)
        df["image_id_key"] = (df["image_name"]
                              .str.replace(".png", "", regex=False)
                              .str.replace(".jpg", "", regex=False)
                              .str.replace(".jpeg", "", regex=False))
        df = df.merge(info_df, left_on="image_id_key", right_on="image_id",
                      how="left", suffixes=("", "_info"))
        df = df.drop(columns=["image_id_key"], errors="ignore")
        matched = df["sherd_id"].notna().sum()
        print(f"成功合并 {INFO_CSV}，匹配了 {matched}/{len(df)} 条记录 ({matched/len(df)*100:.1f}%)")
    except Exception as e:
        print(f"加载或合并 {INFO_CSV} 失败: {e}，将继续使用原始数据")
else:
    print(f"未找到 {INFO_CSV}，跳过信息合并")

# ========================
# 7. 保存
# ========================
df.to_csv(OUTPUT_CSV, index=False)
print(f"已保存合并后的表格到: {OUTPUT_CSV}")
print(f"表格包含 {len(df)} 行，{len(df.columns)} 列")
