"""Main entry for the Dash clustering app."""

import os
from pathlib import Path

from dash import Dash
import pandas as pd
import plotly.express as px

from app_core.callbacks import (
    register_analytics_callbacks,
    register_cluster_panel_callbacks,
    register_compare_callbacks,
    register_recluster_callbacks,
    register_scatter_callbacks,
)
from app_core.layout import build_layout
from app_core.utils import CLUSTER_COLORS
from data_processing import (
    build_samples_for_mode,
    detect_columns,
    ensure_dimensionality_reduction,
    ensure_sample_ids,
    load_cluster_metadata,
)
from performance_utils import optimize_dataframe


APP_CONFIG = {
    'title': '陶片聚类交互可视化',
    'port': 8050,
}

BASE_DIR = Path(__file__).parent
DATA_CSV = BASE_DIR / 'sherd_cluster_table_clustered_only.csv'
FEATURES_CSV = BASE_DIR / 'all_features_dinov3.csv'
IMAGE_ROOT = BASE_DIR / 'all_cutouts'
DEFAULT_CLUSTER_MODE = 'merged'


def load_dataset(csv_path: Path, cluster_mode: str):
    df_raw = pd.read_csv(csv_path)
    cluster_col, image_col = detect_columns(df_raw)
    if cluster_col is None or image_col is None:
        raise ValueError('无法检测聚类列或图片列，请检查数据源')

    df_raw = df_raw.dropna(subset=[cluster_col, image_col]).reset_index(drop=True)
    df_raw = ensure_sample_ids(df_raw, image_col)

    raw_feature_cols = [c for c in df_raw.columns if c not in {cluster_col, image_col}]
    df, feature_cols, _ = build_samples_for_mode(
        df_raw,
        raw_feature_cols,
        cluster_col,
        image_col,
        cluster_mode=cluster_mode,
    )
    # Only keep numeric feature columns to avoid encoding errors
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    df = optimize_dataframe(df)
    return df, feature_cols, raw_feature_cols, cluster_col, image_col


def build_initial_figure(df: pd.DataFrame, feature_cols, cluster_col, hover_cols, custom_data):
    df_embed, reduction_key = ensure_dimensionality_reduction(
        df.copy(),
        feature_cols,
        algorithm='umap',
        n_components=2,
        perplexity=None,
        n_neighbors=15,
        min_dist=0.1,
    )

    fig = px.scatter(
        df_embed,
        x=f'{reduction_key}_0',
        y=f'{reduction_key}_1',
        color=df_embed[cluster_col].astype(str),
        hover_data=hover_cols,
        custom_data=custom_data,
        color_discrete_sequence=CLUSTER_COLORS,
        title='降维散点图 (UMAP)'
    )
    fig.update_traces(marker={'size': 8})
    fig.update_layout(uirevision='tsne-plot')
    return fig


def create_app():
    df, feature_cols, raw_feature_cols, cluster_col, image_col = load_dataset(DATA_CSV, DEFAULT_CLUSTER_MODE)

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
        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        units = [{'label': str(u), 'value': u} for u in sorted(dff['unit_C'].dropna().unique())] if 'unit_C' in dff.columns else []
        parts = [{'label': str(p), 'value': p} for p in sorted(dff['part_C'].dropna().unique())] if 'part_C' in dff.columns else []
        types = [{'label': str(t), 'value': t} for t in sorted(dff['type_C'].dropna().unique())] if 'type_C' in dff.columns else []
        return units, parts, types

    app = Dash(__name__)
    app.title = APP_CONFIG['title']

    app.layout = build_layout(
        fig=fig,
        clusters=clusters,
        init_unit_options=unit_options,
        init_part_options=part_options,
        init_type_options=type_options,
        algorithm_options=algorithm_options,
        initial_cluster_mode=DEFAULT_CLUSTER_MODE,
        cluster_metadata=cluster_metadata,
        df=df,
        feature_cols=feature_cols,
        raw_feature_cols=raw_feature_cols,
        cluster_col=cluster_col,
        image_col=image_col,
    )

    register_scatter_callbacks(app, csv_path=DATA_CSV, image_root=IMAGE_ROOT, get_filter_options=get_filter_options)
    register_compare_callbacks(app)
    register_cluster_panel_callbacks(app)
    register_analytics_callbacks(app, image_root=IMAGE_ROOT)
    register_recluster_callbacks(app, features_csv=FEATURES_CSV, image_root=IMAGE_ROOT)

    return app


def main():
    port = int(os.environ.get('CERAMIC_PORT', APP_CONFIG['port']))
    debug = os.environ.get('CERAMIC_DEBUG', 'false').lower() == 'true'

    app = create_app()
    app.run(debug=debug, port=port, host='127.0.0.1')


if __name__ == '__main__':
    main()
