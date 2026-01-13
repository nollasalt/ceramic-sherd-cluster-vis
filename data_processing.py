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
    rows = []
    dropped = []

    for sample_id, group in df.groupby('sample_id'):
        # 保持确定性顺序；与 kmeans 流程匹配，按文件名排序
        group = group.copy().sort_values(img_name_col)
        if len(group) < 2:
            dropped.append(sample_id)
            continue

        first_two = group.head(2)
        vec1 = first_two.iloc[0][feature_cols].values
        vec2 = first_two.iloc[1][feature_cols].values
        fused_vec = np.concatenate([vec1, vec2], axis=0)

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
