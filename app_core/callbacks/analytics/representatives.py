"""代表样本展示回调。"""
from pathlib import Path

import dash
from dash import ALL, Input, Output, State, html
import numpy as np
import pandas as pd

from app_core.data_cache import get_data_cache


def register_representatives_callbacks(app, *, image_root):
    base_root = Path(__file__).parent.parent.parent.parent

    image_root_abs = Path(image_root)
    if not image_root_abs.is_absolute():
        image_root_abs = base_root / image_root_abs

    def resolve_path(val: str):
        """解析图像路径并在常用目录中兜底查找。"""
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
        Output('representative-grid', 'children'),
        Output('rep-page-index', 'data'),
        Output('rep-page-status', 'children'),
        Output('rep-page-prev', 'disabled'),
        Output('rep-page-next', 'disabled'),
        [Input('visualization-tabs', 'value'),
         Input('rep-samples-per-cluster', 'value'),
         Input('rep-strategy', 'value'),
         Input('rep-page-prev', 'n_clicks'),
         Input('rep-page-next', 'n_clicks'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('rep-page-index', 'data'),
        State('data-store', 'data')
    )
    def render_representatives(
        tab_value, samples_per_cluster, strategy,
        prev_clicks, next_clicks,
        selected_clusters, selected_units, selected_parts, selected_types,
        page_index, data_store,
    ):
        """渲染代表样本，支持分页增量加载簇。图像通过 /img/ 路由直接请求，不再 base64 编码。"""
        if tab_value != 'representatives':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        page_size = 8
        page_index = max(1, int(page_index or 1))

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
        image_col = data_cache['image_col']
        feature_cols = data_cache.get('feature_cols', [])

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        _empty = html.Div('暂无数据', style={'color': '#666', 'padding': '8px'})

        if cluster_col not in dff.columns or len(dff) == 0:
            return _empty, 1, '第 0/0 页（0 个簇）', True, True

        clusters = sorted(dff[cluster_col].dropna().unique())
        if len(clusters) == 0:
            return _empty, 1, '第 0/0 页（0 个簇）', True, True

        total_pages = max(1, (len(clusters) + page_size - 1) // page_size)

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        if trigger_id == 'rep-page-prev':
            page_index = max(1, page_index - 1)
        elif trigger_id == 'rep-page-next':
            page_index = min(total_pages, page_index + 1)
        else:
            page_index = 1

        page_index = max(1, min(page_index, total_pages))
        start_idx = (page_index - 1) * page_size
        active_clusters = clusters[start_idx:start_idx + page_size]

        if len(active_clusters) == 0:
            return _empty, 1, '第 0/0 页（0 个簇）', True, True

        n_per = max(1, min(12, int(samples_per_cluster or 1)))
        thumb_size = 120
        cards = []

        for c in active_clusters:
            subset_all = dff[dff[cluster_col] == c]
            subset_feat = subset_all.dropna(subset=feature_cols) if feature_cols else subset_all
            try:
                cluster_id_for_url = int(c)
            except Exception:
                cluster_id_for_url = str(c)
            assemble_url = f'http://127.0.0.1:12800/?cluster_id={cluster_id_for_url}'

            if strategy == 'center' and feature_cols and len(subset_feat) > 0:
                center_vec = subset_feat[feature_cols].mean().values
                distances = np.linalg.norm(subset_feat[feature_cols].values - center_vec, axis=1)
                chosen = subset_feat.assign(_dist=distances).nsmallest(n_per, '_dist')
            elif strategy == 'random':
                chosen = subset_all.sample(n=min(n_per, len(subset_all)), random_state=42)
            else:
                chosen = subset_all.head(n_per)

            if len(chosen) < n_per and len(subset_all) > len(chosen):
                extra = subset_all.drop(chosen.index, errors='ignore').head(n_per - len(chosen))
                chosen = pd.concat([chosen, extra])

            thumbs = []
            for _, row in chosen.head(n_per).iterrows():
                img_val = row.get('image_name') if 'image_name' in row else row.get(image_col)
                fname = Path(resolve_path(img_val)).name
                thumbs.append(html.Img(
                    src=f'/img/{fname}',
                    style={
                        'height': f'{thumb_size}px', 'border': '1px solid #ddd',
                        'borderRadius': '4px', 'backgroundColor': '#fafafa',
                    },
                    **{'data-image-path': fname},
                    title=str(img_val),
                ))

            while len(thumbs) < n_per:
                thumbs.append(html.Div('样本不足', style={
                    'height': f'{thumb_size}px', 'minWidth': '84px',
                    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                    'border': '1px dashed #d0d0d0', 'borderRadius': '4px',
                    'backgroundColor': '#f8f8f8', 'fontSize': '12px', 'color': '#999',
                }))

            cards.append(html.Div([
                html.Div([
                    html.Div(f'簇 {c}', style={'fontSize': '13px', 'fontWeight': '600'}),
                    html.Div([
                        html.Button(
                            '查看',
                            id={'type': 'rep-view-cluster', 'index': str(c)},
                            n_clicks=0,
                            style={
                                'padding': '4px 10px', 'fontSize': '12px',
                                'backgroundColor': '#0066cc', 'color': 'white',
                                'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer',
                            },
                        ),
                        html.A(
                            '尝试拼对', href=assemble_url, target='_blank',
                            style={
                                'display': 'inline-block', 'padding': '4px 10px',
                                'fontSize': '12px', 'backgroundColor': '#28a745',
                                'color': 'white', 'borderRadius': '4px',
                                'textDecoration': 'none', 'marginLeft': '6px',
                            },
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                ], style={
                    'display': 'flex', 'justifyContent': 'space-between',
                    'alignItems': 'center', 'marginBottom': '6px',
                }),
                html.Div(thumbs, style={'display': 'flex', 'gap': '6px', 'flexWrap': 'wrap'}),
            ], style={
                'padding': '10px', 'border': '1px solid #e0e0e0',
                'borderRadius': '8px', 'minWidth': '180px', 'backgroundColor': '#fff',
            }))

        page_status = (
            f'第 {page_index}/{total_pages} 页｜'
            f'簇 {start_idx + 1}-{start_idx + len(active_clusters)} / {len(clusters)}'
        )
        return cards, page_index, page_status, page_index <= 1, page_index >= total_pages

    @app.callback(
        Output('visualization-tabs', 'value', allow_duplicate=True),
        Output('cluster-filter', 'value', allow_duplicate=True),
        Output('rep-last-view-click', 'data', allow_duplicate=True),
        Input({'type': 'rep-view-cluster', 'index': ALL}, 'n_clicks'),
        State('rep-last-view-click', 'data'),
        prevent_initial_call=True,
    )
    def view_cluster_from_representatives(_n_clicks, last_click):
        """从代表样本页跳转到散点页并自动筛选目标簇。"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update

        trigger_value = ctx.triggered[0].get('value')
        if not isinstance(trigger_value, (int, float)) or trigger_value <= 0:
            return dash.no_update, dash.no_update, dash.no_update

        trigger_id = ctx.triggered_id
        if not isinstance(trigger_id, dict):
            return dash.no_update, dash.no_update, dash.no_update

        cluster_id = trigger_id.get('index')
        if cluster_id is None:
            return dash.no_update, dash.no_update, dash.no_update

        last_click = last_click or {}
        if last_click.get('cluster') == cluster_id and int(last_click.get('count', 0)) == int(trigger_value):
            return dash.no_update, dash.no_update, dash.no_update

        try:
            cluster_value = int(cluster_id)
        except Exception:
            cluster_value = cluster_id

        return 'scatter', [cluster_value], {'cluster': cluster_id, 'count': int(trigger_value)}
