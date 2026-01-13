"""
测试重新聚类功能
"""
from pathlib import Path
from data_processing import perform_kmeans_clustering
import pandas as pd

# 配置路径
FEATURES_CSV = Path(__file__).parent / 'all_features_dinov3.csv'
TABLE_CSV = Path(__file__).parent / 'sherd_cluster_table_clustered_only.csv'

print("=" * 60)
print("测试 K-means 动态聚类功能")
print("=" * 60)

# 测试不同的聚类数
for n_clusters in [5, 10, 15]:
    print(f"\n测试 n_clusters={n_clusters}")
    print("-" * 60)
    
    try:
        result = perform_kmeans_clustering(
            features_csv_path=FEATURES_CSV,
            n_clusters=n_clusters,
            max_samples=1000  # 使用较小的样本数进行快速测试
        )
        
        print(f"✓ 聚类成功!")
        print(f"  - 实际聚类数: {result['n_clusters']}")
        print(f"  - 轮廓系数: {result['silhouette_score']:.4f}")
        print(f"  - 样本总数: {len(result['labels'])}")
        print(f"  - 聚类中心形状: {result['cluster_centers'].shape}")
        
        # 统计每个聚类的样本数
        counts = pd.Series(result['labels']).value_counts().sort_index()
        print(f"  - 聚类分布: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        
    except Exception as e:
        print(f"✗ 聚类失败: {e}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
