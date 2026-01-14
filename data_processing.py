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
# 元数据加载
# ============================================
def load_cluster_metadata(meta_path=None):
    """加载聚类元数据
    
    Args:
        meta_path: 元数据文件路径，默认为 all_kmeans_new/cluster_metadata.json
        
    Returns:
        dict: 聚类元数据，如果文件不存在则返回 None
    """
    if meta_path is None:
        meta_path = DEFAULT_CLUSTER_METADATA_PATH
    meta_path = Path(meta_path)
    
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ============================================
# 列检测与样本ID处理
# ============================================
def detect_columns(df):
    """自动检测 DataFrame 中的聚类列和图像列
    
    Args:
        df: pandas DataFrame
        
    Returns:
        tuple: (cluster_col, image_col) 列名
    """
    cluster_col = None
    image_col = None
    for c in df.columns:
        if 'cluster' in c.lower() and df[c].nunique() > 1:
            cluster_col = c
        if 'image' in c.lower() and ('path' in c.lower() or 'name' in c.lower() or 'file' in c.lower()):
            image_col = c
    return cluster_col, image_col


def ensure_sample_ids(df, image_col):
    """确保 DataFrame 中存在 sample_id 列，如果不存在则从文件名派生
    
    Args:
        df: pandas DataFrame
        image_col: 图像列名
        
    Returns:
        DataFrame: 包含 sample_id 列的 DataFrame
    """
    if 'sample_id' in df.columns:
        return df

    def _derive_sample_id(val):
        name = Path(str(val)).stem
        return name.replace('_exterior', '').replace('_interior', '').split('_')[0]

    df = df.copy()
    df['sample_id'] = df[image_col].apply(_derive_sample_id)
    return df


# ============================================
# 按模式构建样本数据
# ============================================
def build_samples_for_mode(df, feature_cols, cluster_col, image_col, cluster_mode='merged'):
    """根据聚类模式构建用于可视化的样本数据
    
    Args:
        df: pandas DataFrame，包含特征和元数据
        feature_cols: 原始特征列名列表（128维）
        cluster_col: 聚类列名
        image_col: 图像列名
        cluster_mode: 聚类模式 ('merged', 'exterior', 'interior')
        
    Returns:
        tuple: (result_df, result_feature_cols, dropped_samples)
    """
    if cluster_mode == 'merged':
        # 融合模式：使用 build_fused_samples
        return build_fused_samples(df, feature_cols, cluster_col, image_col)
    
    # 单面模式：筛选对应的图像
    if cluster_mode == 'exterior':
        filter_keyword = 'exterior'
    elif cluster_mode == 'interior':
        filter_keyword = 'interior'
    else:
        raise ValueError(f"未知的聚类模式: {cluster_mode}")
    
    img_name_col = 'image_name' if 'image_name' in df.columns else image_col
    
    # 筛选对应类型的图像
    df_filtered = df[df[img_name_col].str.lower().str.contains(filter_keyword)].copy()
    
    if len(df_filtered) == 0:
        raise ValueError(f"没有找到任何 {filter_keyword} 图像")
    
    # 确保有 sample_id
    if 'sample_id' not in df_filtered.columns:
        df_filtered = ensure_sample_ids(df_filtered, image_col)
    
    # 每个 sample_id 只保留一张图像
    rows = []
    dropped = []
    
    for sample_id, group in df_filtered.groupby('sample_id'):
        row = group.iloc[0].to_dict()
        row['sample_id'] = sample_id
        row['image_name'] = str(group.iloc[0][img_name_col])
        # 为单面模式添加paired_images列，只包含该单面的图像
        row['paired_images'] = str(group.iloc[0][img_name_col])
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    # 返回原始特征列（128维），不是融合后的
    return result_df, feature_cols, dropped


# ============================================
# 特征融合
# ============================================
def build_fused_samples(df, feature_cols, cluster_col, image_col):
    """融合正反面特征，使每个样本只表示一次（用于可视化）
    
    与 kmeans_DINO.py 中的融合逻辑保持一致，将两张图像的特征向量拼接。
    
    Args:
        df: pandas DataFrame，包含特征和元数据
        feature_cols: 特征列名列表
        cluster_col: 聚类列名
        image_col: 图像列名
        
    Returns:
        tuple: (fused_df, fused_feature_cols, dropped_samples)
            - fused_df: 融合后的 DataFrame
            - fused_feature_cols: 融合后的特征列名
            - dropped_samples: 因缺少正反面而被跳过的样本ID列表
    """
    img_name_col = 'image_name' if 'image_name' in df.columns else image_col
    fused_feature_cols = [f'fused_{i}' for i in range(len(feature_cols) * 2)]
    
    # 先收集所有的 exterior 和 interior 特征，用于计算标准化参数
    exterior_vecs = []
    interior_vecs = []
    sample_pairs = []  # (sample_id, exterior_row, interior_row)
    dropped = []

    for sample_id, group in df.groupby('sample_id'):
        # 保持确定性顺序；与 kmeans 流程匹配，按文件名排序
        group = group.copy().sort_values(img_name_col)
        if len(group) < 2:
            dropped.append(sample_id)
            continue

        first_two = group.head(2)
        vec1 = first_two.iloc[0][feature_cols].values  # exterior
        vec2 = first_two.iloc[1][feature_cols].values  # interior
        exterior_vecs.append(vec1)
        interior_vecs.append(vec2)
        sample_pairs.append((sample_id, first_two))

    if len(exterior_vecs) == 0:
        return pd.DataFrame(), fused_feature_cols, dropped
    
    # 分别标准化 exterior 和 interior 特征
    exterior_arr = np.stack(exterior_vecs)
    interior_arr = np.stack(interior_vecs)
    
    scaler_ext = StandardScaler()
    scaler_int = StandardScaler()
    exterior_scaled = scaler_ext.fit_transform(exterior_arr)
    interior_scaled = scaler_int.fit_transform(interior_arr)
    
    # 拼接标准化后的特征并构建 DataFrame
    rows = []
    for i, (sample_id, first_two) in enumerate(sample_pairs):
        fused_vec = np.concatenate([exterior_scaled[i], interior_scaled[i]], axis=0)

        # 从第一行获取元数据；同一对内的聚类应该相同
        meta = first_two.iloc[0].drop(feature_cols).to_dict()
        meta['sample_id'] = sample_id
        meta[cluster_col] = first_two[cluster_col].mode().iloc[0]
        meta['image_name'] = str(first_two[img_name_col].iloc[0])
        meta['paired_images'] = ';'.join(first_two[img_name_col].astype(str).tolist())

        for idx, val in enumerate(fused_vec):
            meta[fused_feature_cols[idx]] = val

        rows.append(meta)

    fused_df = pd.DataFrame(rows)
    return fused_df, fused_feature_cols, dropped


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


def create_cluster_similarity_matrix(cluster_centers):
    """创建聚类相似度矩阵（余弦相似度）
    
    Args:
        cluster_centers: 聚类中心矩阵 (n_clusters, n_features)
        
    Returns:
        plotly Figure 对象
    """
    n_clusters = cluster_centers.shape[0]
    similarity_matrix = np.zeros((n_clusters, n_clusters))
    
    # 计算聚类间的余弦相似度
    for i in range(n_clusters):
        for j in range(n_clusters):
            a = cluster_centers[i]
            b = cluster_centers[j]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                similarity = np.dot(a, b) / (norm_a * norm_b)
            else:
                similarity = 0
            similarity_matrix[i, j] = similarity
    
    fig = px.imshow(
        similarity_matrix,
        x=[f'Cluster {i}' for i in range(n_clusters)],
        y=[f'Cluster {i}' for i in range(n_clusters)],
        color_continuous_scale='RdBu_r',
        title='聚类相似度矩阵',
        labels={'x': '聚类', 'y': '聚类', 'color': '余弦相似度'},
        zmin=-1, zmax=1
    )
    
    fig.update_layout(
        xaxis={'tickangle': -45},
        coloraxis_colorbar={'title': '相似度'}
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
# 数据加载与预处理管道
# ============================================
def load_and_prepare_data(csv_path=None, image_root=None):
    """加载并预处理数据，执行特征融合
    
    Args:
        csv_path: CSV 文件路径
        image_root: 图像根目录
        
    Returns:
        dict: 包含以下键的字典
            - df: 融合后的 DataFrame
            - feature_cols: 融合后的特征列名
            - cluster_col: 聚类列名
            - image_col: 图像列名
            - raw_feature_cols: 原始特征列名
            - dropped_samples: 被跳过的样本列表
    """
    if csv_path is None:
        csv_path = DEFAULT_CSV
    if image_root is None:
        image_root = DEFAULT_IMAGE_ROOT
    
    df = pd.read_csv(csv_path)
    cluster_col, image_col = detect_columns(df)
    
    if cluster_col is None or image_col is None:
        raise RuntimeError('无法识别聚类列或图片列，请检查 CSV')
    
    df = ensure_sample_ids(df, image_col)
    
    # 检查是否包含 jd_sherds_info 的字段
    has_info = 'sherd_id' in df.columns or 'unit_C' in df.columns
    if has_info:
        matched_count = df['sherd_id'].notna().sum() if 'sherd_id' in df.columns else 0
        print(f"已加载包含 jd_sherds_info 的表格，匹配了 {matched_count}/{len(df)} 条记录")
    else:
        print(f"表格中未包含 jd_sherds_info 字段，请运行 build_table.py 进行合并")
    
    # 检测特征列（数值列，排除元数据）
    exclude = {cluster_col, image_col, 'image_name', 'sample_id', 'side', 'image_id', 'sherd_id', 
               'unit', 'part', 'type', 'image_side', 'image_id_original', 'unit_C', 'part_C', 'type_C'}
    raw_feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    
    if len(raw_feature_cols) == 0:
        raise RuntimeError('未找到数值特征列')
    
    # 融合正反面特征
    fused_df, feature_cols, dropped_samples = build_fused_samples(df, raw_feature_cols, cluster_col, image_col)
    
    if len(fused_df) == 0:
        raise RuntimeError('融合正反面后没有可用的数据，请检查输入')
    
    if dropped_samples:
        print(f"有 {len(dropped_samples)} 个样本因缺少正反面被跳过: {dropped_samples[:10]} ...")
    
    return {
        'df': fused_df,
        'feature_cols': feature_cols,
        'cluster_col': cluster_col,
        'image_col': image_col,
        'raw_feature_cols': raw_feature_cols,
        'dropped_samples': dropped_samples,
        'image_root': Path(image_root)
    }


# ============================================
# K-Means 聚类功能
# ============================================
def perform_kmeans_clustering(features_csv_path, n_clusters=20, cluster_mode='merged'):
    """执行 K-Means 聚类
    
    Args:
        features_csv_path: DINOv3 特征 CSV 文件路径
        n_clusters: 聚类数量
        cluster_mode: 聚类模式
            - 'merged': 融合正反面特征（默认）
            - 'exterior': 仅使用外部(exterior)图像特征
            - 'interior': 仅使用内部(interior)图像特征
        
    Returns:
        dict: 包含聚类结果的字典
            - labels: 聚类标签
            - cluster_centers: 聚类中心
            - piece_ids: 样本ID
            - features: 融合后的特征
            - silhouette_score: 轮廓系数
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
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
            raise ValueError(f"过滤后没有符合条件的陶片（需要至少有正反面两张图片）")
        
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
        print(f"分别对 exterior 和 interior 特征进行标准化...")
        scaler_ext = StandardScaler()
        scaler_int = StandardScaler()
        exterior_scaled = scaler_ext.fit_transform(exterior_features)
        interior_scaled = scaler_int.fit_transform(interior_features)
        
        # 拼接标准化后的特征
        features = np.concatenate([exterior_scaled, interior_scaled], axis=1)
        print(f"融合后特征维度: {features.shape}")
        
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
    
    print(f"特征维度: {features.shape}")
    
    if len(piece_ids) < n_clusters:
        print(f"警告: 陶片数量 ({len(piece_ids)}) 小于聚类数 ({n_clusters})，将自动调整聚类数")
    
    # 标准化特征（融合模式已在拼接前分别标准化，无需再次标准化）
    if cluster_mode == 'merged':
        print("融合模式: 已在拼接前分别对 exterior/interior 特征进行标准化")
        features_scaled = features
    else:
        print("对特征进行标准化...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    
    # K-Means 聚类
    best_k = min(n_clusters, len(piece_ids))
    print(f"开始 K-Means 聚类 (k={best_k})...")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    cluster_centers = kmeans.cluster_centers_
    
    # 计算轮廓系数
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, labels)
        print(f"轮廓系数: {silhouette_avg:.4f}")
    else:
        silhouette_avg = 0
    
    # 聚类统计
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n聚类分布:")
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
        'n_clusters': best_k
    }
