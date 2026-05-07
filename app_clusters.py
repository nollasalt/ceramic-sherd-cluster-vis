"""
Main entry for the Dash clustering app.
负责应用初始化、数据缓存、布局构建与回调注册。
"""

import os
from pathlib import Path

from dash import Dash
import pandas as pd
import plotly.graph_objects as go
from flask import jsonify, request

from app_core.callbacks import (
    register_analytics_callbacks,
    register_cluster_panel_callbacks,
    register_cluster_preview_callbacks,
    register_compare_callbacks,
    register_recluster_callbacks,
    register_scatter_callbacks,
)
from app_core.layout import build_layout
from app_core.data_cache import get_data_cache, set_data_cache
from data_processing import (
    #detect_columns,
    #ensure_sample_ids,
    load_cluster_metadata,
    load_scope_reference,
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


def build_initial_figure():
    """构建散点图页的轻量占位图，避免应用启动时提前计算 UMAP。"""
    fig = go.Figure()
    fig.update_layout(
        title='降维散点图',
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{
            'text': '进入“散点图”标签页后再加载 UMAP / t-SNE / PCA 结果',
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': 0.5,
            'showarrow': False,
            'font': {'size': 16, 'color': '#666'},
        }],
        uirevision='tsne-plot',
        template='plotly_white',
    )
    return fig


def build_app_layout():
    """基于最新聚类结果构建页面布局，并同步服务端缓存。"""
    from app_core.callbacks.analytics.stratigraphy import _sorted_layers

    cluster_metadata = load_cluster_metadata()
    initial_cluster_mode = cluster_metadata.get('cluster_mode', DEFAULT_CLUSTER_MODE) if cluster_metadata else DEFAULT_CLUSTER_MODE
    initial_n_clusters = cluster_metadata.get('n_clusters', 20) if cluster_metadata else 20
    initial_algorithm = cluster_metadata.get('algorithm', 'kmeans') if cluster_metadata else 'kmeans'

    df, feature_cols, raw_feature_cols, cluster_col, image_col = load_dataset(DATA_CSV, initial_cluster_mode)

    # 将数据集与元数据放入服务端缓存，避免前端携带大体量 JSON
    set_data_cache({
        'df': df,
        'feature_cols': feature_cols,
        'raw_feature_cols': raw_feature_cols,
        'cluster_col': cluster_col,
        'image_col': image_col,
        'cluster_mode': initial_cluster_mode,
    })

    fig = build_initial_figure()

    clusters = sorted(df[cluster_col].dropna().unique())
    unit_options = [{'label': str(u), 'value': u} for u in sorted(df['unit_C'].dropna().unique())] if 'unit_C' in df.columns else []
    part_options = [{'label': str(p), 'value': p} for p in sorted(df['part_C'].dropna().unique())] if 'part_C' in df.columns else []
    type_options = [{'label': str(t), 'value': t} for t in sorted(df['type_C'].dropna().unique())] if 'type_C' in df.columns else []
    scope_df = load_scope_reference()
    if scope_df is None:
        scope_df = df
    scope_unit_values = _sorted_layers([u for u in scope_df['unit_C'].dropna().unique() if str(u).strip()]) if 'unit_C' in scope_df.columns else []
    scope_unit_options = [{'label': str(u), 'value': u} for u in scope_unit_values]
    scope_part_values = sorted([p for p in scope_df['part_C'].dropna().unique() if str(p).strip()]) if 'part_C' in scope_df.columns else []
    scope_part_options = [{'label': str(p), 'value': p} for p in scope_part_values]

    algorithm_options = [
        {'label': 't-SNE', 'value': 'tsne'},
        {'label': 'UMAP', 'value': 'umap'},
        {'label': 'PCA', 'value': 'pca'},
    ]

    return build_layout(
        fig=fig,
        clusters=clusters,
        init_unit_options=unit_options,
        init_part_options=part_options,
        init_type_options=type_options,
        init_scope_unit_options=scope_unit_options,
        init_scope_part_options=scope_part_options,
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


def create_app():
    """创建并配置 Dash 应用实例。"""
    def get_filter_options(selected_clusters):
        """根据当前缓存中的数据返回联动筛选项。"""
        data_cache = get_data_cache()
        if not data_cache:
            return [], [], []

        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
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
    app.layout = build_app_layout

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
    register_cluster_preview_callbacks(app, image_root=IMAGE_ROOT)
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
