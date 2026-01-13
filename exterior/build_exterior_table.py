import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# ========= 配置 - 仅处理正面图像 =========
FEATURE_CSV = BASE_DIR / "exterior_features_dinov3.csv"
CLUSTER_DIR = BASE_DIR / "exterior_kmeans_results"
INFO_CSV = ROOT_DIR / "jd_sherds_info.csv"  # 回到上级目录找信息文件
OUTPUT_CSV = BASE_DIR / "exterior_cluster_table.csv"

print("构建仅正面图像的聚类表格")

# ========================
# 1. 收集已完成聚类的正面图像文件名
# ========================
clustered_files = {}  # filename -> cluster_id

if not CLUSTER_DIR.exists():
    print(f" 聚类目录不存在: {CLUSTER_DIR}")
    print("请先运行 kmeans_exterior_only.py 进行聚类")
    exit(1)

for cluster_name in os.listdir(CLUSTER_DIR):
    if not cluster_name.startswith("cluster_"):
        continue
    
    # 确保只处理目录而不是文件
    cluster_path = os.path.join(CLUSTER_DIR, cluster_name)
    if not os.path.isdir(cluster_path):
        continue

    # 处理噪声点目录
    if cluster_name == "cluster_noise":
        cluster_id = -1  # 为噪声点分配特殊ID
    else:
        cluster_id = int(cluster_name.split("_")[1])

    for fname in os.listdir(cluster_path):
        if fname.endswith((".png", ".jpg", ".jpeg")):
            clustered_files[fname] = cluster_id

print(f" 找到 {len(clustered_files)} 张已聚类的正面图像")

# 验证都是exterior图像
exterior_count = sum(1 for fname in clustered_files.keys() if 'exterior' in fname.lower())
print(f" 其中 exterior 图像: {exterior_count} 张")

if exterior_count != len(clustered_files):
    print(f" 警告: 发现 {len(clustered_files) - exterior_count} 个非exterior文件")

# ========================
# 2. 读取正面特征 CSV
# ========================
if not FEATURE_CSV.exists():
    print(f" 特征文件不存在: {FEATURE_CSV}")
    print("请先运行 DINOV3_features_exterior.py 提取特征")
    exit(1)

df = pd.read_csv(FEATURE_CSV)
df = df.rename(columns={"filename": "image_name"})

print(f" 读取特征文件: {len(df)} 条记录")

# ========================
# 3. 仅保留已聚类的正面图片
# ========================
df = df[df["image_name"].isin(clustered_files.keys())].copy()
print(f" 筛选后剩余: {len(df)} 条记录")

# ========================
# 4. 解析 sample_id（陶片ID）
# ========================
def parse_exterior_filename(name):
    """解析正面图像文件名"""
    stem = os.path.splitext(os.path.basename(name))[0]
    # 去掉_exterior后缀得到陶片ID
    sample_id = stem.replace("_exterior", "")
    side = "exterior"  # 都是正面
    return sample_id, side

df[["sample_id", "side"]] = df["image_name"].apply(
    lambda x: pd.Series(parse_exterior_filename(x))
)

# ========================
# 5. 添加 cluster_id 和 image_path
# ========================
df["cluster_id"] = df["image_name"].map(clustered_files)
df["image_path"] = df.apply(
    lambda r: str(CLUSTER_DIR / f"cluster_{r.cluster_id}" / r.image_name),
    axis=1
)

# 验证cluster_id分配
print(f" 聚类ID分布:")
cluster_counts = df['cluster_id'].value_counts().sort_index()
for cluster_id, count in cluster_counts.head(10).items():
    print(f"  Cluster {cluster_id}: {count} 张图像")
if len(cluster_counts) > 10:
    print(f"  ... 总共 {len(cluster_counts)} 个聚类")

# ========================
# 6. 合并 jd_sherds_info.csv（如果存在）
# ========================
if INFO_CSV.exists():
    print(f" 加载陶片信息: {INFO_CSV}")
    try:
        info_df = pd.read_csv(INFO_CSV)
        
        # 只保留exterior相关的信息
        info_exterior = info_df[info_df['image_side'] == 'exterior'].copy()
        print(f" 信息文件中的exterior记录: {len(info_exterior)} 条")
        
        # 将 image_name 去掉扩展名和_exterior后缀用于匹配
        df['image_id_key'] = df['image_name'].str.replace('.png', '', regex=False).str.replace('.jpg', '', regex=False).str.replace('.jpeg', '', regex=False).str.replace('_exterior', '', regex=False)
        
        # 合并信息（基于陶片ID匹配）
        df = df.merge(info_exterior, left_on='image_id_key', right_on='sherd_id', how='left', suffixes=('', '_info'))
        
        # 删除临时列
        df = df.drop(columns=['image_id_key'], errors='ignore')
        
        matched_count = df['sherd_id'].notna().sum()
        total_count = len(df)
        print(f" 成功合并陶片信息，匹配了 {matched_count}/{total_count} 条记录 ({matched_count/total_count*100:.1f}%)")
        
        # 显示合并后的列
        info_cols = [col for col in df.columns if col not in ['image_name', 'sample_id', 'side', 'cluster_id', 'image_path'] and not col.isdigit()]
        print(f" 新增信息列: {info_cols}")
        
    except Exception as e:
        print(f" 加载或合并 {INFO_CSV} 失败: {e}，将继续使用原始数据")
else:
    print(f" 未找到 {INFO_CSV}，跳过信息合并")

# ========================
# 7. 保存
# ========================
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n 已保存正面图像聚类表格到: {OUTPUT_CSV}")
print(f" 表格包含 {len(df)} 行，{len(df.columns)} 列")

# ========================
# 8. 数据统计摘要
# ========================
print(f"\n 数据摘要:")
print(f"  - 总图像数: {len(df)}")
print(f"  - 聚类数: {df['cluster_id'].nunique()}")
print(f"  - 唯一陶片数: {df['sample_id'].nunique()}")
print(f"  - 特征维度: {len([col for col in df.columns if col.isdigit()])}")

# 显示最大和最小的聚类
if len(cluster_counts) > 0:
    max_cluster = cluster_counts.idxmax()
    min_cluster = cluster_counts.idxmin()
    print(f"  - 最大聚类 {max_cluster}: {cluster_counts[max_cluster]} 张图像")
    print(f"  - 最小聚类 {min_cluster}: {cluster_counts[min_cluster]} 张图像")
    print(f"  - 平均每聚类: {len(df)/df['cluster_id'].nunique():.1f} 张图像")

print(f"\n 正面图像聚类表格构建完成！")