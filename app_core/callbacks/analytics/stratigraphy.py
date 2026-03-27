"""地层流动分析回调：Sankey 图、跨层热力图、统计摘要。"""
import unicodedata

import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app_core.data_cache import get_data_cache
from app_core.utils import CLUSTER_COLORS
from performance_utils import cache_plot_result


# ── 层位排序辅助 ──────────────────────────────────────────────────────────────

def _layer_sort_key(s):
    """对层位字符串排序：⑭排最前（最新/最浅），①排最后（最早/最深）。"""
    if not s or not isinstance(s, str):
        return (9999, s or '')
    for ch in s:
        try:
            n = unicodedata.numeric(ch)
            if n == int(n):
                return (-int(n), s)  # 负号使大数字排前面
        except (ValueError, TypeError):
            pass
    return (9998, s)  # 无圆圈数字（H690混 等）放倒数第二


def _sorted_layers(layers):
    """返回按地层顺序排列的层位列表（⑭ → ①）。"""
    return sorted(layers, key=_layer_sort_key)


# ── 统计计算辅助 ───────────────────────────────────────────────────────────────

def _shannon_entropy(counts):
    """计算归一化 Shannon 熵（0=完全集中，1=均匀分散）。"""
    total = counts.sum()
    if total == 0 or len(counts) <= 1:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    raw = -float(np.sum(probs * np.log2(probs)))
    return raw / np.log2(len(counts))


def _simpson_concentration(counts):
    """Simpson 集中度（越高越集中于少数类别）。"""
    total = counts.sum()
    if total <= 1:
        return 1.0
    return float(np.sum((counts / total) ** 2))


# ── 主回调 ────────────────────────────────────────────────────────────────────

def register_stratigraphy_callbacks(app):

    # 初始化下拉框选项
    @app.callback(
        Output('strat-unit-filter', 'options'),
        Output('strat-cluster-filter', 'options'),
        Input('visualization-tabs', 'value'),
        State('data-store', 'data'),
    )
    def init_strat_filters(tab_value, _data_store):
        """当切换到地层流动标签页时填充层位和簇的下拉选项。"""
        if tab_value != 'stratigraphy':
            return dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']

        units = [u for u in df['unit_C'].dropna().unique() if str(u).strip()]
        units_sorted = _sorted_layers(units)
        unit_opts = [{'label': str(u), 'value': u} for u in units_sorted]

        clusters = sorted(df[cluster_col].dropna().unique())
        cluster_opts = [{'label': f'簇 {c}', 'value': c} for c in clusters]

        return unit_opts, cluster_opts

    # 主渲染回调
    @app.callback(
        Output('stratigraphy-sankey', 'figure'),
        Output('stratigraphy-heatmap', 'figure'),
        Output('stratigraphy-stats', 'children'),
        Input('visualization-tabs', 'value'),
        Input('strat-unit-filter', 'value'),
        Input('strat-cluster-filter', 'value'),
        Input('strat-heatmap-mode', 'value'),
        Input('strat-min-link', 'value'),
        State('data-store', 'data'),
    )
    @cache_plot_result
    def render_stratigraphy(tab_value, sel_units, sel_clusters, heatmap_mode, min_link, _data_store):
        """渲染地层流动 Sankey、热力图和统计摘要。"""
        if tab_value != 'stratigraphy':
            return dash.no_update, dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']

        # 过滤无效行
        dff = df[df['unit_C'].notna() & df['unit_C'].astype(str).str.strip().ne('')].copy()
        if cluster_col not in dff.columns or len(dff) == 0:
            empty = px.scatter(title='暂无数据')
            return empty, empty, html.Div('暂无数据', style={'color': '#666'})

        if sel_units:
            dff = dff[dff['unit_C'].isin(sel_units)]
        if sel_clusters:
            dff = dff[dff[cluster_col].isin(sel_clusters)]

        if len(dff) == 0:
            empty = px.scatter(title='筛选后暂无数据')
            return empty, empty, html.Div('筛选后无数据', style={'color': '#666'})

        # 构建层位 × 簇计数矩阵
        pivot = (
            dff.groupby(['unit_C', cluster_col], observed=True)
            .size()
            .reset_index(name='count')
        )
        matrix = pivot.pivot_table(
            index='unit_C', columns=cluster_col, values='count', fill_value=0
        )

        layers_sorted = _sorted_layers(matrix.index.tolist())
        matrix = matrix.reindex(layers_sorted)
        clusters_sorted = sorted(matrix.columns.tolist())
        matrix = matrix[clusters_sorted]

        # ── Sankey ────────────────────────────────────────────────────────────
        min_link = int(min_link or 5)
        n_layers = len(layers_sorted)
        layer_idx = {lyr: i for i, lyr in enumerate(layers_sorted)}
        cluster_idx = {c: n_layers + i for i, c in enumerate(clusters_sorted)}

        sources, targets, values, link_labels = [], [], [], []
        for _, row in pivot.iterrows():
            lyr, cid, cnt = row['unit_C'], row[cluster_col], int(row['count'])
            if cnt < min_link:
                continue
            if lyr not in layer_idx or cid not in cluster_idx:
                continue
            sources.append(layer_idx[lyr])
            targets.append(cluster_idx[cid])
            values.append(cnt)
            link_labels.append(f'{lyr} → 簇{cid}：{cnt} 片')

        # 节点颜色
        layer_colors = [f'rgba(31,119,180,{0.5 + 0.4 * i / max(n_layers - 1, 1)})' for i in range(n_layers)]
        cluster_colors = [
            CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] if str(c).lstrip('-').isdigit() else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for i, c in enumerate(clusters_sorted)
        ]
        node_labels = [str(lyr) for lyr in layers_sorted] + [f'簇{c}' for c in clusters_sorted]
        node_colors = layer_colors + cluster_colors

        if values:
            sankey_fig = go.Figure(go.Sankey(
                arrangement='snap',
                node=dict(
                    label=node_labels,
                    color=node_colors,
                    pad=12,
                    thickness=18,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    label=link_labels,
                    color='rgba(180,180,180,0.35)',
                ),
            ))
            sankey_fig.update_layout(
                title=f'层位 → 簇 流向图｜最小连线 ≥ {min_link} 片',
                margin=dict(l=20, r=20, t=50, b=20),
                font_size=12,
            )
        else:
            sankey_fig = go.Figure()
            sankey_fig.update_layout(
                title='无满足阈值的连线，请降低最小连线值',
                margin=dict(l=20, r=20, t=50, b=20),
            )

        # ── 热力图 ────────────────────────────────────────────────────────────
        hm_data = matrix.values.astype(float)

        if heatmap_mode == 'by_layer':
            row_sums = hm_data.sum(axis=1, keepdims=True)
            hm_data = np.where(row_sums > 0, hm_data / row_sums, 0)
            color_label = '层内占比'
        elif heatmap_mode == 'by_cluster':
            col_sums = hm_data.sum(axis=0, keepdims=True)
            hm_data = np.where(col_sums > 0, hm_data / col_sums, 0)
            color_label = '簇内占比'
        else:
            color_label = '陶片数'

        heatmap_fig = px.imshow(
            hm_data,
            x=[str(c) for c in clusters_sorted],
            y=[str(lyr) for lyr in layers_sorted],
            color_continuous_scale='YlOrRd',
            labels={'x': '簇', 'y': '地层', 'color': color_label},
            title=f'簇跨层分布｜{color_label}',
            aspect='auto',
        )
        heatmap_fig.update_layout(
            margin=dict(l=80, r=20, t=50, b=60),
            xaxis_title='簇编号',
            yaxis_title='地层层位',
        )
        heatmap_fig.update_xaxes(side='bottom')

        # ── 统计摘要 ──────────────────────────────────────────────────────────
        count_matrix = matrix.values  # 原始计数，用于统计

        # 每个簇跨越的层数（persistence）
        persistence = (count_matrix > 0).sum(axis=0)
        most_persistent_idx = int(np.argmax(persistence))
        most_persistent_cluster = clusters_sorted[most_persistent_idx]
        most_persistent_layers = int(persistence[most_persistent_idx])

        # 每个簇的 Simpson 集中度（最集中 = 最可能是某层特有）
        concentrations = [_simpson_concentration(count_matrix[:, j]) for j in range(len(clusters_sorted))]
        most_concentrated_idx = int(np.argmax(concentrations))
        most_concentrated_cluster = clusters_sorted[most_concentrated_idx]
        most_concentrated_val = concentrations[most_concentrated_idx]

        # 每层的 Shannon 多样性
        layer_entropy = [_shannon_entropy(count_matrix[i, :]) for i in range(len(layers_sorted))]

        # 每层的主导簇
        dominant = []
        for i, lyr in enumerate(layers_sorted):
            row = count_matrix[i, :]
            total = row.sum()
            if total == 0:
                continue
            dom_idx = int(np.argmax(row))
            dom_cluster = clusters_sorted[dom_idx]
            dom_pct = row[dom_idx] / total
            ent = layer_entropy[i]
            dominant.append((lyr, dom_cluster, dom_pct, ent, int(total)))

        stat_items = [
            html.Div([
                html.Span('跨层最广', style={'color': '#666', 'fontSize': '12px'}),
                html.Div(
                    f'簇 {most_persistent_cluster}（{most_persistent_layers} 层）',
                    style={'fontWeight': '600', 'fontSize': '15px', 'color': '#1f77b4'}
                ),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span('最集中于单层', style={'color': '#666', 'fontSize': '12px'}),
                html.Div(
                    f'簇 {most_concentrated_cluster}（集中度 {most_concentrated_val:.2f}）',
                    style={'fontWeight': '600', 'fontSize': '15px', 'color': '#d62728'}
                ),
            ], style={'marginBottom': '14px'}),
        ]

        if dominant:
            rows = []
            for lyr, dom_c, dom_pct, ent, total in dominant:
                rows.append(html.Tr([
                    html.Td(str(lyr), style={'padding': '3px 6px', 'fontSize': '12px'}),
                    html.Td(f'簇{dom_c}', style={'padding': '3px 6px', 'fontSize': '12px', 'fontWeight': '600'}),
                    html.Td(f'{dom_pct:.0%}', style={'padding': '3px 6px', 'fontSize': '12px'}),
                    html.Td(f'{ent:.2f}', style={'padding': '3px 6px', 'fontSize': '12px', 'color': '#555'}),
                    html.Td(str(total), style={'padding': '3px 6px', 'fontSize': '12px', 'color': '#888'}),
                ]))
            stat_items.append(html.Div('各层概况', style={
                'fontWeight': '600', 'fontSize': '12px', 'color': '#444', 'marginBottom': '4px'
            }))
            stat_items.append(html.Table([
                html.Thead(html.Tr([
                    html.Th('层位', style={'padding': '3px 6px', 'fontSize': '11px', 'color': '#888'}),
                    html.Th('主导簇', style={'padding': '3px 6px', 'fontSize': '11px', 'color': '#888'}),
                    html.Th('占比', style={'padding': '3px 6px', 'fontSize': '11px', 'color': '#888'}),
                    html.Th('多样性', style={'padding': '3px 6px', 'fontSize': '11px', 'color': '#888'}),
                    html.Th('总片数', style={'padding': '3px 6px', 'fontSize': '11px', 'color': '#888'}),
                ])),
                html.Tbody(rows),
            ], style={'borderCollapse': 'collapse', 'width': '100%'}))

        return sankey_fig, heatmap_fig, html.Div(stat_items)
