"""
数据处理模块
包含特征融合、降维、可视化图表生成等功能
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import plotly.express as px

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ============================================
# 配置常量
# ============================================
DEFAULT_CSV = 'sherd_cluster_table_clustered_only.csv'
DEFAULT_IMAGE_ROOT = Path('all_cutouts')
DEFAULT_CLUSTER_METADATA_PATH = Path('all_kmeans_new/cluster_metadata.json')


# ============================================
# 元数据与基础工具
# ============================================
def load_cluster_metadata(path: Path = DEFAULT_CLUSTER_METADATA_PATH):
    """加载聚类元数据，不存在则返回 None"""
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as exc:  # 读文件失败时打印并返回 None
        print(f"读取聚类元数据失败: {exc}")
    return None


# def detect_columns(df: pd.DataFrame):
#     """简单检测聚类列与图片列"""

#     # print(f"检测到聚类列: {cluster_col}, 图片列: {image_col}")
#     cluster_col = "cluster_id"
#     image_col = "image_name"
#     return cluster_col, image_col


# ============================================
# 聚类功能
# ============================================
def _prepare_clustering_data(features_csv_path, cluster_mode='merged'):
    """加载特征并按聚类模式准备特征矩阵"""
    # 读取特征
    df = pd.read_csv(features_csv_path).copy()

    if "filename" not in df.columns:
        raise ValueError("CSV 中必须包含 'filename' 列")

    print(f"读取 {len(df)} 条特征记录")
    print(f"聚类模式: {cluster_mode}")

    # 获取主编号
    def get_piece_id(filename):
        name = Path(filename).stem
        name = name.replace("_exterior", "").replace("_interior", "")
        return name.lower()

    df["main_id"] = df["filename"].apply(get_piece_id)

    # 特征列
    feature_cols = [c for c in df.columns if c not in ["filename", "main_id"]]

    if cluster_mode == 'merged':
        # 融合模式：每个 main_id 需要正反面两张图像
        selected_list = []
        for main_id, group in df.groupby("main_id"):
            if len(group) >= 2:
                result = group.iloc[:2].copy()
                result['main_id'] = main_id
                selected_list.append(result)

        if len(selected_list) == 0:
            raise ValueError("过滤后没有符合条件的陶片（需要至少有正反面两张图片）")

        selected_df = pd.concat(selected_list, ignore_index=True)
        unique_pieces = selected_df['main_id'].unique()
        print(f"过滤后剩余陶片数: {len(unique_pieces)}")

        # 分别提取 exterior 和 interior 特征
        exterior_features_list = []
        interior_features_list = []
        piece_ids_list = []

        for main_id, group in selected_df.groupby("main_id"):
            if len(group) < 2:
                continue
            # 按文件名排序，确保顺序一致（exterior 字母序在 interior 之前）
            group_sorted = group.sort_values('filename')
            vec1 = group_sorted.iloc[0][feature_cols].values  # exterior
            vec2 = group_sorted.iloc[1][feature_cols].values  # interior
            exterior_features_list.append(vec1)
            interior_features_list.append(vec2)
            piece_ids_list.append(main_id)

        if len(exterior_features_list) == 0:
            raise ValueError("没有成功融合任何特征，请检查数据格式")

        exterior_features = np.stack(exterior_features_list)
        interior_features = np.stack(interior_features_list)
        piece_ids = np.array(piece_ids_list)

        # 分别对 exterior 和 interior 特征进行标准化，然后拼接
        print("分别对 exterior 和 interior 特征进行标准化...")
        scaler_ext = StandardScaler()
        scaler_int = StandardScaler()
        exterior_scaled = scaler_ext.fit_transform(exterior_features)
        interior_scaled = scaler_int.fit_transform(interior_features)

        # 拼接标准化后的特征
        features = np.concatenate([exterior_scaled, interior_scaled], axis=1)
        print(f"融合后特征维度: {features.shape}")
        features_scaled = features  # 已标准化

    else:
        # 单面模式：仅使用 exterior 或 interior 图像
        if cluster_mode == 'exterior':
            filter_keyword = 'exterior'
        elif cluster_mode == 'interior':
            filter_keyword = 'interior'
        else:
            raise ValueError(f"未知的聚类模式: {cluster_mode}")

        # 筛选对应类型的图像
        df_filtered = df[df['filename'].str.lower().str.contains(filter_keyword)].copy()
        print(f"筛选 {filter_keyword} 图像: {len(df_filtered)} 张")

        if len(df_filtered) == 0:
            raise ValueError(f"没有找到任何 {filter_keyword} 图像")

        # 每个 main_id 只保留一张
        selected_list = []
        for main_id, group in df_filtered.groupby("main_id"):
            result = group.iloc[:1].copy()
            result['main_id'] = main_id
            selected_list.append(result)

        selected_df = pd.concat(selected_list, ignore_index=True)
        unique_pieces = selected_df['main_id'].unique()
        print(f"过滤后剩余陶片数: {len(unique_pieces)}")

        # 直接使用单张图像特征
        features = selected_df[feature_cols].values
        piece_ids = selected_df['main_id'].values

        print("对特征进行标准化...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

    print(f"特征维度: {features.shape}")

    return {
        'features': features,
        'features_scaled': features_scaled,
        'piece_ids': piece_ids,
        'selected_df': selected_df
    }


def perform_kmeans_clustering(features_csv_path, n_clusters=20, cluster_mode='merged'):
    """执行 K-Means 聚类"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    prep = _prepare_clustering_data(features_csv_path, cluster_mode)
    features = prep['features']
    features_scaled = prep['features_scaled']
    piece_ids = prep['piece_ids']
    selected_df = prep['selected_df']

    if len(piece_ids) < n_clusters:
        print(f"警告: 陶片数量 ({len(piece_ids)}) 小于聚类数 ({n_clusters})，将自动调整聚类数")

    best_k = min(n_clusters, len(piece_ids))
    print(f"开始 K-Means 聚类 (k={best_k})...")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    cluster_centers = kmeans.cluster_centers_

    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, labels)
        print(f"轮廓系数: {silhouette_avg:.4f}")
    else:
        silhouette_avg = 0

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n聚类分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} 个样本")

    return {
        'labels': labels,
        'cluster_centers': cluster_centers,
        'piece_ids': piece_ids,
        'features': features,
        'features_scaled': features_scaled,
        'silhouette_score': silhouette_avg,
        'selected_df': selected_df,
        'n_clusters': best_k,
        'algorithm': 'kmeans'
    }


def perform_agglomerative_clustering(features_csv_path, n_clusters=20, cluster_mode='merged', linkage='ward'):
    """执行层次聚类（凝聚式）"""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    prep = _prepare_clustering_data(features_csv_path, cluster_mode)
    features = prep['features']
    features_scaled = prep['features_scaled']
    piece_ids = prep['piece_ids']
    selected_df = prep['selected_df']

    if len(piece_ids) < n_clusters:
        print(f"警告: 陶片数量 ({len(piece_ids)}) 小于聚类数 ({n_clusters})，将自动调整聚类数")

    best_k = min(n_clusters, len(piece_ids))
    print(f"开始 Agglomerative 聚类 (k={best_k}, linkage={linkage})...")

    model = AgglomerativeClustering(n_clusters=best_k, linkage=linkage)
    labels = model.fit_predict(features_scaled)

    # 计算每个簇的中心（均值），用于与 K-Means 输出保持一致结构
    cluster_centers = []
    for c in range(best_k):
        mask = labels == c
        if np.any(mask):
            center = features_scaled[mask].mean(axis=0)
        else:
            center = np.zeros(features_scaled.shape[1])
        cluster_centers.append(center)
    cluster_centers = np.stack(cluster_centers)

    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, labels)
        print(f"轮廓系数: {silhouette_avg:.4f}")
    else:
        silhouette_avg = 0

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n聚类分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} 个样本")

    return {
        'labels': labels,
        'cluster_centers': cluster_centers,
        'piece_ids': piece_ids,
        'features': features,
        'features_scaled': features_scaled,
        'silhouette_score': silhouette_avg,
        'selected_df': selected_df,
        'n_clusters': best_k,
        'algorithm': f'agglomerative-{linkage}'
    }


def perform_spectral_clustering(features_csv_path, n_clusters=20, cluster_mode='merged', assign_labels='kmeans', gamma=None):
    """执行谱聚类；gamma 未指定时默认 1.0"""
    from sklearn.metrics import silhouette_score

    prep = _prepare_clustering_data(features_csv_path, cluster_mode)
    features = prep['features']
    features_scaled = prep['features_scaled']
    piece_ids = prep['piece_ids']
    selected_df = prep['selected_df']

    if len(piece_ids) < n_clusters:
        print(f"警告: 陶片数量 ({len(piece_ids)}) 小于聚类数 ({n_clusters})，将自动调整聚类数")

    best_k = min(n_clusters, len(piece_ids))
    if best_k < 2:
        raise ValueError("谱聚类需要至少 2 个样本")

    if gamma is None:
        gamma = 1.0  # sklearn 需非负浮点数

    print(f"开始 Spectral 聚类 (k={best_k}, assign_labels={assign_labels}, gamma={gamma})...")

    model = SpectralClustering(
        n_clusters=best_k,
        affinity='rbf',
        assign_labels=assign_labels,
        random_state=42,
        gamma=gamma
    )
    labels = model.fit_predict(features_scaled)

    # 使用簇内均值作为聚类中心的近似
    cluster_centers = []
    for c in range(best_k):
        mask = labels == c
        if np.any(mask):
            center = features_scaled[mask].mean(axis=0)
        else:
            center = np.zeros(features_scaled.shape[1])
        cluster_centers.append(center)
    cluster_centers = np.stack(cluster_centers)

    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, labels)
        print(f"轮廓系数: {silhouette_avg:.4f}")
    else:
        silhouette_avg = 0

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n聚类分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} 个样本")

    return {
        'labels': labels,
        'cluster_centers': cluster_centers,
        'piece_ids': piece_ids,
        'features': features,
        'features_scaled': features_scaled,
        'silhouette_score': silhouette_avg,
        'selected_df': selected_df,
        'n_clusters': best_k,
        'algorithm': f'spectral-{assign_labels}'
    }


# ============================================
# 降维处理
# ============================================
def generate_reduction_key(algorithm, n_components, **params):
    """生成降维结果的唯一标识符
    
    Args:
        algorithm: 降维算法名称 ('pca', 'tsne', 'umap')
        n_components: 目标维度
        **params: 算法特定参数
        
    Returns:
        str: 唯一标识符
    """
    if algorithm == 'pca':
        return f'{algorithm}_{n_components}'
    elif algorithm == 'tsne':
        perplexity = params.get('perplexity', 30.0)
        return f'{algorithm}_{n_components}_perp{int(perplexity)}'
    elif algorithm == 'umap':
        n_neighbors = params.get('n_neighbors', 15)
        min_dist = params.get('min_dist', 0.1)
        return f'{algorithm}_{n_components}_nn{n_neighbors}_md{int(min_dist*100)}'
    else:
        return f'{algorithm}_{n_components}'


def ensure_dimensionality_reduction(df, feature_cols, algorithm='umap', n_components=2, 
                                     perplexity=30.0, n_neighbors=15, min_dist=0.1):
    """确保已对给定算法和参数执行降维
    
    如果已经计算过，直接返回缓存结果；否则执行降维并缓存。
    
    Args:
        df: pandas DataFrame
        feature_cols: 特征列名列表
        algorithm: 降维算法 ('pca', 'tsne', 'umap')
        n_components: 目标维度
        perplexity: t-SNE 困惑度参数
        n_neighbors: UMAP 邻居数参数
        min_dist: UMAP 最小距离参数
        
    Returns:
        tuple: (df, reduction_key)
            - df: 包含降维结果列的 DataFrame
            - reduction_key: 降维结果的唯一标识符
    """
    # 生成当前参数组合的唯一键
    reduction_key = generate_reduction_key(algorithm, n_components, 
                                           perplexity=perplexity, 
                                           n_neighbors=n_neighbors, 
                                           min_dist=min_dist)
    cols = [f'{reduction_key}_{i}' for i in range(n_components)]
    
    # 检查是否已经有了该参数组合的结果
    if all(c in df.columns for c in cols):
        return df, reduction_key
    
    X = df[feature_cols].values
    
    # 标准化特征（对降维很重要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algorithm == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca')
    elif algorithm == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif algorithm == 'umap':
        if not UMAP_AVAILABLE:
            print("警告: UMAP 未安装，回退到 t-SNE")
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca')
            algorithm = 'tsne'
            reduction_key = generate_reduction_key(algorithm, n_components, perplexity=perplexity)
            cols = [f'{reduction_key}_{i}' for i in range(n_components)]
        else:
            reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    else:
        # 回退到 t-SNE
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca')
        algorithm = 'tsne'
    
    Xt = reducer.fit_transform(X_scaled)
    for i in range(n_components):
        df[f'{reduction_key}_{i}'] = Xt[:, i]
    return df, reduction_key


# ============================================
# 可视化图表生成
# ============================================
def create_cluster_pattern_heatmap(cluster_centers, feature_names=None):
    """创建聚类中心特征热力图
    
    Args:
        cluster_centers: 聚类中心矩阵 (n_clusters, n_features)
        feature_names: 特征名称列表，默认为 Feature 0, Feature 1, ...
        
    Returns:
        plotly Figure 对象
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(cluster_centers.shape[1])]
    
    # 对特征进行降维以便可视化
    if cluster_centers.shape[1] > 100:
        cluster_centers_vis = cluster_centers[:, :100]
        feature_names = feature_names[:100]
    else:
        cluster_centers_vis = cluster_centers
    
    fig = px.imshow(
        cluster_centers_vis,
        x=feature_names,
        y=[f'Cluster {i}' for i in range(cluster_centers_vis.shape[0])],
        color_continuous_scale='Viridis',
        title='聚类中心特征热力图',
        labels={'x': '特征', 'y': '聚类', 'color': '特征值'}
    )
    
    fig.update_layout(
        xaxis={'tickangle': -45, 'title_text': '特征索引'},
        yaxis={'title_text': '聚类ID'},
        coloraxis_colorbar={'title': '特征值'}
    )
    
    return fig


# ============================================
# 图像处理
# ============================================
def img_to_base64(path, max_size=400):
    """将图像转换为 base64 编码的 data URI
    
    Args:
        path: 图像文件路径
        max_size: 缩略图最大尺寸
        
    Returns:
        str: base64 编码的 data URI，失败时返回 None
    """
    try:
        im = Image.open(path).convert('RGBA')
        
        # 将接近白色的像素设为透明
        try:
            arr = np.array(im)
            if arr.shape[2] == 4:
                r, g, b, a = np.split(arr, 4, axis=2)
                mask = (np.squeeze(r) > 245) & (np.squeeze(g) > 245) & (np.squeeze(b) > 245)
                arr[mask, 3] = 0
                im = Image.fromarray(arr)
        except Exception:
            pass

        # 调整大小以加快传输
        im.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        data = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{data}'
    except Exception:
        return None


def img_to_base64_full(path, max_size=1200):
    """将图像转换为 base64 编码的 data URI (原图质量)
    
    Args:
        path: 图像文件路径
        max_size: 原图最大尺寸（更大以保持细节）
        
    Returns:
        str: base64 编码的 data URI，失败时返回 None
    """
    try:
        im = Image.open(path).convert('RGBA')
        
        # 将接近白色的像素设为透明
        try:
            arr = np.array(im)
            if arr.shape[2] == 4:
                r, g, b, a = np.split(arr, 4, axis=2)
                mask = (np.squeeze(r) > 245) & (np.squeeze(g) > 245) & (np.squeeze(b) > 245)
                arr[mask, 3] = 0
                im = Image.fromarray(arr)
        except Exception:
            pass

        # 保持较高分辨率的原图
        if max_size > 0:
            im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buf = io.BytesIO()
        im.save(buf, format='PNG', optimize=True)
        data = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{data}'
    except Exception:
        return None


# ============================================
# Leiden 聚类（Mutual kNN + Leiden 社区发现）
# ============================================
def perform_leiden_clustering(
    features_csv_path,
    cluster_mode='merged',
    topk=30,
    second_order_weight=0.05,
    resolution=2.5,
    min_diffusion_weight=0.01
):
    """使用 mutual kNN 图和 Leiden 社区发现进行聚类。

    Returns 结构与其他聚类函数保持一致，便于可视化和表格生成。
    """

    try:
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_similarity
        import scipy.sparse as sp
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # 依赖缺失时给出明确提示
        raise ImportError(
            "Leiden 聚类需要安装 python-igraph 和 leidenalg, 以及 scipy"
        ) from exc

    prep = _prepare_clustering_data(features_csv_path, cluster_mode)
    features = prep['features']
    features_scaled = prep['features_scaled']
    piece_ids = prep['piece_ids']
    selected_df = prep['selected_df']

    n = features_scaled.shape[0]
    if n < 2:
        raise ValueError('样本数量不足，无法执行 Leiden 聚类')

    print(f"开始 Leiden 聚类: n={n}, topk={topk}, resolution={resolution}, second_order_weight={second_order_weight}")

    # 构建 mutual kNN 相似度图
    sim = cosine_similarity(features_scaled)
    np.fill_diagonal(sim, 0)

    neighbors = np.argsort(sim, axis=1)[:, -topk:]
    neighbor_sets = [set(row) for row in neighbors]

    rows, cols, data = [], [], []
    for i in range(n):
        for j in neighbors[i]:
            if i in neighbor_sets[j]:
                w = sim[i, j] ** 2  # 抑制弱边
                rows.append(i)
                cols.append(j)
                data.append(w)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    print(f"一阶图：节点 {n}，边数 {A.nnz}")

    # 二阶扩散增强传递性
    A2 = A @ A
    A2 = A2.multiply(second_order_weight)
    if min_diffusion_weight is not None and min_diffusion_weight > 0:
        A2.data[A2.data < min_diffusion_weight] = 0
        A2.eliminate_zeros()

    A_final = A + A2
    A_final.eliminate_zeros()
    print(f"最终图：边数 {A_final.nnz}")

    if A_final.nnz == 0:
        raise ValueError('图为空，无法进行 Leiden 聚类')

    edges = list(zip(A_final.nonzero()[0], A_final.nonzero()[1]))
    weights = A_final.data.tolist()

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es['weight'] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution
    )

    labels = np.empty(n, dtype=int)
    for cid, cluster in enumerate(partition):
        labels[list(cluster)] = cid

    n_clusters = len(partition)
    print(f"Leiden 得到 {n_clusters} 个聚类")

    # 计算聚类中心（使用标准化特征的均值）
    cluster_centers = []
    for c in range(n_clusters):
        mask = labels == c
        if np.any(mask):
            center = features_scaled[mask].mean(axis=0)
        else:
            center = np.zeros(features_scaled.shape[1], dtype=np.float32)
        cluster_centers.append(center)
    cluster_centers = np.stack(cluster_centers)

    silhouette_avg = 0.0
    if n_clusters > 1:
        silhouette_avg = float(silhouette_score(features_scaled, labels))
        print(f"轮廓系数: {silhouette_avg:.4f}")

    return {
        'labels': labels,
        'cluster_centers': cluster_centers,
        'piece_ids': piece_ids,
        'features': features,
        'features_scaled': features_scaled,
        'silhouette_score': silhouette_avg,
        'selected_df': selected_df,
        'n_clusters': n_clusters,
        'algorithm': 'leiden'
    }
