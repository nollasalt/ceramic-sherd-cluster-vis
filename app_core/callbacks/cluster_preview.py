"""簇预览面板回调：显示选中簇的图片缩略图。"""
from pathlib import Path

import dash
from dash import Input, Output, State, html
import pandas as pd

from app_core.data_cache import get_data_cache


def register_cluster_preview_callbacks(app, *, image_root):
    """注册簇预览面板回调。"""
    base_root = Path(__file__).parent.parent.parent
    image_root_abs = Path(image_root)
    if not image_root_abs.is_absolute():
        image_root_abs = base_root / image_root_abs

    def resolve_path(val: str):
        """解析图像路径。"""
        p = Path(str(val))
        if not p.is_absolute():
            p = image_root_abs / p
        if p.exists():
            return p
        alt = base_root / 'all_cutouts' / p.name
        if alt.exists():
            return alt
        alt2 = base_root / 'all_kmeans_new' / p.name
        if alt2.exists():
            return alt2
        return p

    @app.callback(
        Output('cluster-preview-content', 'children'),
        Output('cluster-preview-page', 'data'),
        Output('cluster-preview-page-info', 'children'),
        Output('cluster-preview-prev', 'disabled'),
        Output('cluster-preview-next', 'disabled'),
        Input('cluster-preview-selector', 'value'),
        Input('cluster-preview-pagesize', 'value'),
        Input('cluster-preview-prev', 'n_clicks'),
        Input('cluster-preview-next', 'n_clicks'),
        State('cluster-preview-page', 'data'),
        State('data-store', 'data'),
    )
    def render_cluster_preview(
        selected_cluster, page_size, prev_clicks, next_clicks, page_index, data_store
    ):
        """渲染选中簇的图片缩略图，支持分页。"""
        if selected_cluster is None:
            return html.Div('请选择一个簇', style={'color': '#999', 'padding': '12px', 'textAlign': 'center'}), 1, '', True, True

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
        image_col = data_cache['image_col']

        # 筛选该簇的样本
        try:
            cluster_df = df[df[cluster_col] == selected_cluster]
        except Exception:
            cluster_df = df[df[cluster_col].astype(str) == str(selected_cluster)]

        if len(cluster_df) == 0:
            return html.Div('该簇无样本', style={'color': '#999', 'padding': '12px', 'textAlign': 'center'}), 1, '', True, True

        page_size = int(page_size or 12)
        page_index = int(page_index or 1)
        total_pages = max(1, (len(cluster_df) + page_size - 1) // page_size)

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        if trigger_id == 'cluster-preview-prev':
            page_index = max(1, page_index - 1)
        elif trigger_id == 'cluster-preview-next':
            page_index = min(total_pages, page_index + 1)
        elif trigger_id in ['cluster-preview-selector', 'cluster-preview-pagesize']:
            page_index = 1

        page_index = max(1, min(page_index, total_pages))
        start_idx = (page_index - 1) * page_size
        page_df = cluster_df.iloc[start_idx:start_idx + page_size]

        thumbs = []
        for _, row in page_df.iterrows():
            img_val = row.get('image_name') if 'image_name' in row else row.get(image_col)
            fname = Path(resolve_path(img_val)).name

            # 显示样本ID或文件名
            sample_id = row.get('sample_id', row.get('piece_id', fname))

            thumbs.append(html.Div([
                html.Img(
                    src=f'/img/{fname}',
                    style={
                        'width': '100%',
                        'height': '100px',
                        'objectFit': 'cover',
                        'border': '1px solid #ddd',
                        'borderRadius': '4px',
                        'backgroundColor': '#fff',
                    },
                    title=str(img_val),
                ),
                html.Div(
                    str(sample_id)[:12],
                    style={
                        'fontSize': '10px',
                        'color': '#666',
                        'marginTop': '3px',
                        'textAlign': 'center',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'whiteSpace': 'nowrap',
                    }
                ),
            ], style={'marginBottom': '8px'}))

        page_info = f'{page_index}/{total_pages} 页 ({len(cluster_df)} 个样本)'

        return (
            html.Div(thumbs, style={'display': 'flex', 'flexDirection': 'column', 'gap': '4px'}),
            page_index,
            page_info,
            page_index <= 1,
            page_index >= total_pages,
        )
