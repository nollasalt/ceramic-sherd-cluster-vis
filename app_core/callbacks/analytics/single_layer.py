"""单层详情分析回调。"""
from pathlib import Path

import dash
from dash import Input, Output, State, html, dcc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_core.data_cache import get_data_cache
from app_core.callbacks.analytics.stratigraphy import _sorted_layers
from app_core.utils import CLUSTER_COLORS
from performance_utils import cache_plot_result


def register_single_layer_callbacks(app, *, image_root):
    """注册单层详情分析回调。"""
    base_root = Path(__file__).parent.parent.parent.parent
    image_root_abs = Path(image_root)
    if not image_root_abs.is_absolute():
        image_root_abs = base_root / image_root_abs

    def resolve_path(val: str):
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
        Output('single-layer-selector', 'options'),
        Input('visualization-tabs', 'value'),
        State('data-store', 'data'),
    )
    def init_layer_selector(tab_value, _):
        """初始化地层选择器。"""
        if tab_value != 'single-layer':
            return dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        units = [u for u in df['unit_C'].dropna().unique() if str(u).strip()]
        units_sorted = _sorted_layers(units)
        return [{'label': str(u), 'value': u} for u in units_sorted]

    @app.callback(
        Output('single-layer-content', 'children'),
        Input('single-layer-selector', 'value'),
        State('data-store', 'data'),
    )
    def render_single_layer(selected_layer, _):
        """渲染单层详情分析。"""
        if not selected_layer:
            return html.Div('请选择一个地层', style={'color': '#999', 'padding': '20px', 'textAlign': 'center'})

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
        image_col = data_cache['image_col']
        feature_cols = data_cache.get('feature_cols', [])

        # 筛选该层数据（astype(str) 避免 Categorical 类型匹配失败）
        layer_df = df[df['unit_C'].astype(str) == str(selected_layer)].copy()
        if len(layer_df) == 0:
            return html.Div(f'地层 {selected_layer} 无数据', style={'color': '#999', 'padding': '20px'})

        # 统计簇分布
        cluster_counts = layer_df[cluster_col].value_counts().sort_index()
        clusters = cluster_counts.index.tolist()

        # ── 条形图：簇大小分布 ────────────────────────────────────────────
        bar_fig = go.Figure(go.Bar(
            x=[f'簇{c}' for c in clusters],
            y=cluster_counts.values,
            marker=dict(color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(clusters))]),
            text=cluster_counts.values,
            textposition='outside',
            hovertemplate='%{x}<br>%{y} 片<extra></extra>',
        ))
        bar_fig.update_layout(
            title=f'{selected_layer} 簇大小分布（共 {len(layer_df)} 片）',
            xaxis_title='簇',
            yaxis_title='样本数',
            margin=dict(l=50, r=20, t=50, b=60),
            height=350,
            showlegend=False,
        )

        # ── 纵向流动图：相邻层关系 ────────────────────────────────────────
        flow_fig = None
        all_units = _sorted_layers([u for u in df['unit_C'].dropna().unique() if str(u).strip()])
        if selected_layer in all_units:
            layer_idx = all_units.index(selected_layer)
            adjacent_layers = []
            if layer_idx > 0:
                adjacent_layers.append(all_units[layer_idx - 1])
            adjacent_layers.append(selected_layer)
            if layer_idx < len(all_units) - 1:
                adjacent_layers.append(all_units[layer_idx + 1])

            if len(adjacent_layers) > 1:
                # 简化版：直接显示各层的簇分布，不追踪单个样本
                # 构建堆叠条形图
                layer_data = {}
                all_clusters_in_range = set()

                for unit in adjacent_layers:
                    unit_df = df[df['unit_C'].astype(str) == str(unit)]
                    unit_counts = unit_df[cluster_col].value_counts()
                    layer_data[unit] = unit_counts
                    all_clusters_in_range.update(unit_counts.index)

                all_clusters_sorted = sorted(all_clusters_in_range)

                # 创建分组条形图（每个簇一组，不同层并排显示）
                flow_fig = go.Figure()
                for unit in adjacent_layers:
                    y_vals = [layer_data[unit].get(c, 0) for c in all_clusters_sorted]
                    flow_fig.add_trace(go.Bar(
                        name=str(unit),
                        x=[f'簇{c}' for c in all_clusters_sorted],
                        y=y_vals,
                        hovertemplate='%{x}<br>' + str(unit) + ': %{y}片<extra></extra>',
                    ))

                flow_fig.update_layout(
                    title=f'{selected_layer} 与相邻层的簇分布对比',
                    xaxis_title='簇',
                    yaxis_title='样本数',
                    barmode='group',  # 分组模式，同一簇的不同层并排显示
                    margin=dict(l=50, r=20, t=50, b=80),
                    height=450,
                    showlegend=True,
                    legend=dict(title='地层'),
                )

        # ── 代表样本网格 ──────────────────────────────────────────────────
        samples_per_cluster = 4
        sample_cards = []

        for c in clusters:
            cluster_df = layer_df[layer_df[cluster_col].astype(str) == str(c)]

            # 选择代表样本（距中心最近）
            if feature_cols and len(cluster_df) > 0:
                cluster_feat = cluster_df.dropna(subset=feature_cols)
                if len(cluster_feat) > 0:
                    center_vec = cluster_feat[feature_cols].mean().values
                    distances = np.linalg.norm(cluster_feat[feature_cols].values - center_vec, axis=1)
                    chosen = cluster_feat.assign(_dist=distances).nsmallest(samples_per_cluster, '_dist')
                else:
                    chosen = cluster_df.head(samples_per_cluster)
            else:
                chosen = cluster_df.head(samples_per_cluster)

            thumbs = []
            for _, row in chosen.iterrows():
                img_val = row.get('image_name') if 'image_name' in row else row.get(image_col)
                fname = Path(resolve_path(img_val)).name
                thumbs.append(html.Img(
                    src=f'/img/{fname}',
                    style={
                        'height': '80px',
                        'border': '1px solid #ddd',
                        'borderRadius': '4px',
                        'backgroundColor': '#fafafa',
                    },
                    title=str(img_val),
                ))

            sample_cards.append(html.Div([
                html.Div(f'簇 {c}（{len(cluster_df)} 片）', style={
                    'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '6px', 'color': '#333'
                }),
                html.Div(thumbs, style={'display': 'flex', 'gap': '4px', 'flexWrap': 'wrap'}),
            ], style={
                'padding': '10px',
                'border': '1px solid #e0e0e0',
                'borderRadius': '6px',
                'backgroundColor': '#fff',
                'minWidth': '200px',
            }))

        # ── 簇间相似度热力图 ──────────────────────────────────────────────
        heatmap_fig = None
        if feature_cols and len(clusters) > 1:
            # 计算各簇中心
            centers = []
            for c in clusters:
                cluster_feat = layer_df[layer_df[cluster_col].astype(str) == str(c)].dropna(subset=feature_cols)
                if len(cluster_feat) > 0:
                    centers.append(cluster_feat[feature_cols].mean().values)
                else:
                    centers.append(np.zeros(len(feature_cols)))

            centers = np.array(centers)

            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(centers)

            heatmap_fig = go.Figure(go.Heatmap(
                z=sim_matrix,
                x=[f'簇{c}' for c in clusters],
                y=[f'簇{c}' for c in clusters],
                colorscale='RdYlGn',
                zmin=0,
                zmax=1,
                colorbar=dict(title='相似度'),
                hovertemplate='%{y} × %{x}<br>相似度: %{z:.3f}<extra></extra>',
            ))
            heatmap_fig.update_layout(
                title=f'{selected_layer} 簇间相似度（余弦）',
                margin=dict(l=80, r=20, t=50, b=60),
                height=400,
                xaxis=dict(side='bottom'),
            )

        # ── 组装页面 ──────────────────────────────────────────────────────
        components = [
            html.Div([
                html.Div([
                    dcc.Graph(figure=bar_fig, style={'width': '70%'}),
                    html.Div([
                        html.Div(f'总样本数: {len(layer_df)}', style={'fontSize': '14px', 'marginBottom': '8px'}),
                        html.Div(f'簇数量: {len(clusters)}', style={'fontSize': '14px', 'marginBottom': '8px'}),
                        html.Div(f'平均簇大小: {len(layer_df) / len(clusters):.1f}', style={'fontSize': '14px', 'marginBottom': '8px'}),
                    ], style={'width': '30%', 'padding': '20px'}),
                ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '16px'}),
            ]),
        ]

        if flow_fig:
            components.append(html.Div([
                dcc.Graph(figure=flow_fig),
            ], style={'marginBottom': '16px'}))

        components.append(
            html.Div([
                html.H4('各簇代表样本', style={'fontSize': '15px', 'fontWeight': '600', 'marginBottom': '10px', 'color': '#444'}),
                html.Div(sample_cards, style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px'}),
            ], style={'marginBottom': '16px'})
        )

        if heatmap_fig:
            components.append(dcc.Graph(figure=heatmap_fig))

        return html.Div(components)
