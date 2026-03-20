"""
Main entry for the Dash clustering app.
负责应用初始化、数据缓存、布局构建与回调注册。
"""

import os
from pathlib import Path

from dash import Dash
import pandas as pd
import plotly.express as px
from flask import jsonify, request

from app_core.callbacks import (
    register_analytics_callbacks,
    register_cluster_panel_callbacks,
    register_compare_callbacks,
    register_recluster_callbacks,
    register_scatter_callbacks,
)
from app_core.layout import build_layout
from app_core.data_cache import set_data_cache
from app_core.utils import CLUSTER_COLORS, PART_SYMBOL_SEQUENCE, get_part_symbol_settings
from data_processing import (
    #detect_columns,
    ensure_dimensionality_reduction,
    #ensure_sample_ids,
    load_cluster_metadata,
    img_to_base64_full,
)
from performance_utils import optimize_dataframe


APP_CONFIG = {
    # 应用标题与默认端口
    'title': '陶片聚类交互可视化',
    'port': 9357,
}

BASE_DIR = Path(__file__).parent
DATA_CSV = BASE_DIR / 'sherd_cluster_table_clustered_only.csv'
FEATURES_CSV = BASE_DIR / 'all_features_dinov3.csv'
IMAGE_ROOT = BASE_DIR / 'all_cutouts'
UMAP_CACHE = BASE_DIR / 'umap_cache.npz'  # 缓存 UMAP 坐标，避免每次启动重算
DEFAULT_CLUSTER_MODE = 'merged'  # 默认聚类模式，正反面融合
IMAGE_SEARCH_DIRS = list(dict.fromkeys([
    IMAGE_ROOT,
    BASE_DIR / 'all_cutouts',
    BASE_DIR / 'all_kmeans_new',
]))
FULL_IMAGE_CACHE = {}


def load_dataset(csv_path: Path, cluster_mode: str):
    """加载聚类表并提取可用于可视化/分析的数值特征列。"""
    df_raw = pd.read_csv(csv_path)
    cluster_col = "cluster_id"
    image_col = "image_name"

    df_raw = df_raw.dropna(subset=[cluster_col, image_col]).reset_index(drop=True)
    #df_raw = ensure_sample_ids(df_raw, image_col)

    raw_feature_cols = [c for c in df_raw.columns if c not in {cluster_col, image_col}]
    df = df_raw.copy()
    feature_cols = list(raw_feature_cols)
    # Only keep numeric feature columns to avoid encoding errors
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    df = optimize_dataframe(df)
    return df, feature_cols, raw_feature_cols, cluster_col, image_col


def find_image_path(image_path: str) -> Path | None:
    """
    Locate an image by name across known roots, with a tiny in-process cache.
    """

    if not image_path:
        return None

    target = Path(image_path)
    target_name = target.name

    cached = FULL_IMAGE_CACHE.get(target_name)
    if cached:
        cached_path = Path(cached)
        if cached_path.exists():
            return cached_path
        FULL_IMAGE_CACHE.pop(target_name, None)

    candidates = [target]
    if not target.is_absolute():
        candidates.append(Path(target_name))

    for base in IMAGE_SEARCH_DIRS:
        base = Path(base)
        if not base.exists():
            continue
        for cand in candidates:
            cand_path = base / cand
            if cand_path.exists():
                FULL_IMAGE_CACHE[target_name] = str(cand_path)
                return cand_path
        try:
            match = next(base.rglob(target_name))
            if match.exists():
                FULL_IMAGE_CACHE[target_name] = str(match)
                return match
        except StopIteration:
            pass

    return None


def build_initial_figure(df: pd.DataFrame, feature_cols, cluster_col, hover_cols, custom_data):
    """构建首页默认二维降维散点图（UMAP），结果缓存到磁盘。"""
    import time
    import numpy as np

    # ── 尝试读取磁盘缓存 ──────────────────────────────────────────────────────
    reduction_key = 'umap_2_nn15_md10'
    umap_cols = [f'{reduction_key}_0', f'{reduction_key}_1']
    cache_valid = False

    if UMAP_CACHE.exists():
        try:
            cache = np.load(UMAP_CACHE, allow_pickle=True)
            cached_ids = cache['sample_id'].astype(str)
            cur_ids = df['sample_id'].astype(str).reset_index(drop=True).values
            if len(cached_ids) == len(cur_ids) and (cached_ids == cur_ids).all():
                df = df.reset_index(drop=True)
                df[umap_cols[0]] = cache['x']
                df[umap_cols[1]] = cache['y']
                cache_valid = True
                print("✓ UMAP 缓存命中，跳过重新计算")
        except Exception as e:
            print(f"UMAP 缓存读取失败，将重新计算: {e}")

    if not cache_valid:
        t0 = time.time()
        print("计算 UMAP（首次或缓存失效）...")
        df, reduction_key = ensure_dimensionality_reduction(
            df.copy(),
            feature_cols,
            algorithm='umap',
            n_components=2,
            perplexity=None,
            n_neighbors=15,
            min_dist=0.1,
        )
        print(f"UMAP 完成，耗时 {time.time() - t0:.1f}s")
        # 保存缓存
        try:
            np.savez_compressed(
                UMAP_CACHE,
                sample_id=df['sample_id'].astype(str).values,
                x=df[umap_cols[0]].values,
                y=df[umap_cols[1]].values,
            )
            print(f"✓ UMAP 坐标已缓存到 {UMAP_CACHE.name}")
        except Exception as e:
            print(f"UMAP 缓存写入失败: {e}")

    part_symbol_col, part_symbol_map = get_part_symbol_settings(df)
    symbol_kwargs = {}
    if part_symbol_col:
        symbol_kwargs = {
            'symbol': part_symbol_col,
            'symbol_map': part_symbol_map,
            'symbol_sequence': PART_SYMBOL_SEQUENCE,
        }

    fig = px.scatter(
        df,
        x=f'{reduction_key}_0',
        y=f'{reduction_key}_1',
        color=df[cluster_col].astype(str),
        hover_data=hover_cols,
        custom_data=custom_data,
        color_discrete_sequence=CLUSTER_COLORS,
        title='降维散点图 (UMAP)',
        render_mode='webgl',
        **symbol_kwargs,
    )
    fig.update_traces(marker={'size': 8})
    fig.update_layout(uirevision='tsne-plot')

    # 把带 UMAP 列的 df 写回 data_cache，避免首次筛选时重算
    from app_core.data_cache import get_data_cache
    dc = get_data_cache()
    if dc:
        dc['df'] = df
        set_data_cache(dc)

    return fig


def create_app():
    """创建并配置 Dash 应用实例。"""
    df, feature_cols, raw_feature_cols, cluster_col, image_col = load_dataset(DATA_CSV, DEFAULT_CLUSTER_MODE)

    # 将数据集与元数据放入服务端缓存，避免前端携带大体量 JSON
    set_data_cache({
        'df': df,
        'feature_cols': feature_cols,
        'raw_feature_cols': raw_feature_cols,
        'cluster_col': cluster_col,
        'image_col': image_col,
        'cluster_mode': DEFAULT_CLUSTER_MODE,
    })

    hover_cols = [cluster_col]
    for col in ['sample_id', 'unit_C', 'part_C', 'type_C']:
        if col in df.columns:
            hover_cols.append(col)
    hover_cols = list(dict.fromkeys(hover_cols))

    custom_data = ['sample_id']
    for col in ['image_name', 'paired_images']:
        if col in df.columns:
            custom_data.append(col)

    fig = build_initial_figure(df, feature_cols, cluster_col, hover_cols, custom_data)

    clusters = sorted(df[cluster_col].dropna().unique())
    unit_options = [{'label': str(u), 'value': u} for u in sorted(df['unit_C'].dropna().unique())] if 'unit_C' in df.columns else []
    part_options = [{'label': str(p), 'value': p} for p in sorted(df['part_C'].dropna().unique())] if 'part_C' in df.columns else []
    type_options = [{'label': str(t), 'value': t} for t in sorted(df['type_C'].dropna().unique())] if 'type_C' in df.columns else []

    algorithm_options = [
        {'label': 't-SNE', 'value': 'tsne'},
        {'label': 'UMAP', 'value': 'umap'},
        {'label': 'PCA', 'value': 'pca'},
    ]

    cluster_metadata = load_cluster_metadata()

    def get_filter_options(selected_clusters):
        """根据已选簇返回联动筛选项（unit/part/type）。"""
        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        units = [{'label': str(u), 'value': u} for u in sorted(dff['unit_C'].dropna().unique())] if 'unit_C' in dff.columns else []
        parts = [{'label': str(p), 'value': p} for p in sorted(dff['part_C'].dropna().unique())] if 'part_C' in dff.columns else []
        types = [{'label': str(t), 'value': t} for t in sorted(dff['type_C'].dropna().unique())] if 'type_C' in dff.columns else []
        return units, parts, types

    # 改为本地提供依赖，避免访问外部 CDN 导致加载过慢
    app = Dash(
        __name__,
        serve_locally=True,
        compress=True,
        assets_folder=str(BASE_DIR / 'assets'),
        assets_url_path='/assets',
    )
    app.title = APP_CONFIG['title']

    # 从上次聚类结果中读取初始值，避免重启后显示默认值
    initial_n_clusters = cluster_metadata.get('n_clusters', 20) if cluster_metadata else 20
    initial_algorithm = cluster_metadata.get('algorithm', 'kmeans') if cluster_metadata else 'kmeans'
    initial_cluster_mode = cluster_metadata.get('cluster_mode', DEFAULT_CLUSTER_MODE) if cluster_metadata else DEFAULT_CLUSTER_MODE

    app.layout = build_layout(
        fig=fig,
        clusters=clusters,
        init_unit_options=unit_options,
        init_part_options=part_options,
        init_type_options=type_options,
        algorithm_options=algorithm_options,
        initial_cluster_mode=initial_cluster_mode,
        initial_n_clusters=initial_n_clusters,
        initial_algorithm=initial_algorithm,
        cluster_metadata=cluster_metadata,
        df=df,
        feature_cols=feature_cols,
        raw_feature_cols=raw_feature_cols,
        cluster_col=cluster_col,
        image_col=image_col,
    )

    server = app.server

    @server.route('/get_full_image')
    def get_full_image():
        """根据查询参数返回大图 base64 数据。"""
        image_path = request.args.get('image_path', '')
        found = find_image_path(image_path)
        if not found or not found.exists():
            return jsonify({'error': 'image_not_found', 'path': Path(image_path).name}), 404
        try:
            data_url = img_to_base64_full(found)
            if not data_url:
                return jsonify({'error': 'encode_failed', 'path': Path(image_path).name}), 500
            return data_url
        except Exception as exc:  # pragma: no cover - runtime safety
            return jsonify({'error': 'server_error', 'detail': str(exc)}), 500

    register_scatter_callbacks(app, csv_path=DATA_CSV, image_root=IMAGE_ROOT, get_filter_options=get_filter_options)
    register_compare_callbacks(app)
    register_cluster_panel_callbacks(app)
    register_analytics_callbacks(app, image_root=IMAGE_ROOT, image_search_dirs=IMAGE_SEARCH_DIRS)
    register_recluster_callbacks(app, features_csv=FEATURES_CSV, image_root=IMAGE_ROOT)

    return app


def main():
    """应用启动入口，读取环境变量并运行 Dash 服务。"""
    port = int(os.environ.get('CERAMIC_PORT', APP_CONFIG['port']))
    debug = os.environ.get('CERAMIC_DEBUG', 'false').lower() == 'true'

    app = create_app()
    # 关闭热重载以降低资源占用；监听 0.0.0.0 便于外部访问
    app.run(debug=debug, port=port, host='0.0.0.0', dev_tools_hot_reload=False)


if __name__ == '__main__':
    main()
