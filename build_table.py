import os
import pandas as pd

FEATURE_CSV = "all_features_dinov3.csv"
CLUSTER_DIR = "all_kmeans_new"
INFO_CSV = "jd_sherds_info.csv"
OUTPUT_CSV = "sherd_cluster_table_clustered_only.csv"

# ========================
# 1. 收集已完成聚类的文件名
# ========================
clustered_files = {}  # filename -> cluster_id

for cluster_name in os.listdir(CLUSTER_DIR):
    if not cluster_name.startswith("cluster_"):
        continue

    cluster_id = int(cluster_name.split("_")[1])
    cluster_path = os.path.join(CLUSTER_DIR, cluster_name)

    for fname in os.listdir(cluster_path):
        if fname.endswith(".png"):
            clustered_files[fname] = cluster_id

print(f"Found {len(clustered_files)} clustered images")

# ========================
# 2. 读取特征 CSV
# ========================
df = pd.read_csv(FEATURE_CSV)
df = df.rename(columns={"filename": "image_name"})

# ========================
# 3. 仅保留已聚类图片
# ========================
df = df[df["image_name"].isin(clustered_files.keys())].copy()
print(f"Rows after filtering: {len(df)}")

# ========================
# 4. 解析 sample_id / side
# ========================
def parse_filename(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    parts = stem.split("_")
    sample_id = parts[0]
    side = parts[1] if len(parts) > 1 else "unknown"
    return sample_id, side

df[["sample_id", "side"]] = df["image_name"].apply(
    lambda x: pd.Series(parse_filename(x))
)

# ========================
# 5. 添加 cluster_id 和 image_path
# ========================
df["cluster_id"] = df["image_name"].map(clustered_files)
df["image_path"] = df.apply(
    lambda r: os.path.join(
        CLUSTER_DIR, f"cluster_{r.cluster_id}", r.image_name
    ),
    axis=1
)

# ========================
# 6. 合并 jd_sherds_info.csv（如果存在）
# ========================
if os.path.exists(INFO_CSV):
    print(f"Loading info from: {INFO_CSV}")
    try:
        info_df = pd.read_csv(INFO_CSV)
        # 将 image_name 去掉扩展名用于匹配
        df['image_id_key'] = df['image_name'].str.replace('.png', '', regex=False).str.replace('.jpg', '', regex=False).str.replace('.jpeg', '', regex=False)
        # 合并信息
        df = df.merge(info_df, left_on='image_id_key', right_on='image_id', how='left', suffixes=('', '_info'))
        # 删除临时列
        df = df.drop(columns=['image_id_key'], errors='ignore')
        
        matched_count = df['sherd_id'].notna().sum()
        total_count = len(df)
        print(f"✅ 成功合并 {INFO_CSV}，匹配了 {matched_count}/{total_count} 条记录 ({matched_count/total_count*100:.1f}%)")
    except Exception as e:
        print(f"⚠️ 加载或合并 {INFO_CSV} 失败: {e}，将继续使用原始数据")
else:
    print(f"⚠️ 未找到 {INFO_CSV}，跳过信息合并")

# ========================
# 7. 保存
# ========================
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 已保存合并后的表格到: {OUTPUT_CSV}")
print(f"📊 表格包含 {len(df)} 行，{len(df.columns)} 列")
