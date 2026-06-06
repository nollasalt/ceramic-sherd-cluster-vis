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


def _cluster_sort_key(value):
    """按簇名称排序，数字优先按数值排，其余按字符串排。"""
    if value is None:
        return (2, '')
    text = str(value).strip()
    if text == '':
        return (2, '')
    if text.lstrip('-').isdigit():
        return (0, int(text))
    return (1, text)


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


def _color_with_alpha(color, alpha):
    """将 Plotly 颜色字符串转换为带指定透明度的 rgba。"""
    if not color:
        return f'rgba(180,180,180,{alpha})'

    color = str(color).strip()
    if color.startswith('rgba('):
        parts = [p.strip() for p in color[5:-1].split(',')]
        if len(parts) >= 3:
            return f'rgba({parts[0]},{parts[1]},{parts[2]},{alpha})'
    if color.startswith('rgb('):
        parts = [p.strip() for p in color[4:-1].split(',')]
        if len(parts) >= 3:
            return f'rgba({parts[0]},{parts[1]},{parts[2]},{alpha})'
    if color.startswith('#'):
        hex_color = color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join(ch * 2 for ch in hex_color)
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f'rgba({r},{g},{b},{alpha})'
            except ValueError:
                pass

    return color


def _sankey_node_style(n_nodes):
    """根据节点数量动态调整 Sankey 节点厚度、间距和图高。"""
    n_nodes = max(int(n_nodes or 0), 1)
    if n_nodes >= 20:
        thickness = 9
        pad = 4
    elif n_nodes >= 16:
        thickness = 10
        pad = 5
    elif n_nodes >= 14:
        thickness = 12
        pad = 6
    elif n_nodes >= 10:
        thickness = 14
        pad = 8
    else:
        thickness = 18
        pad = 12

    height = max(560, 160 + n_nodes * (thickness + pad + 12))
    font_size = 10 if n_nodes >= 18 else 11 if n_nodes >= 14 else 12
    return thickness, pad, height, font_size


def _ordered_node_positions(weights, top=0.03, bottom=0.97, min_gap=0.012):
    """按节点总流量分配纵向锚点，尽量在 snap 下保持既定顺序。"""
    weights = [max(float(w or 0), 0.0) for w in weights]
    n_nodes = len(weights)
    if n_nodes <= 0:
        return []
    if n_nodes == 1:
        return [0.5]

    total = sum(weights)
    if total <= 0:
        return np.linspace(top, bottom, n_nodes).tolist()

    span = max(bottom - top, 0.2)
    gap = min(min_gap, span / max(n_nodes * 3, 3))
    usable = max(span - gap * (n_nodes - 1), span * 0.5)
    heights = [(w / total) * usable for w in weights]

    positions = []
    cursor = top
    for height in heights:
        positions.append(cursor)
        cursor += height + gap
    return positions


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

        clusters = sorted(df[cluster_col].dropna().unique(), key=_cluster_sort_key)
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
        clusters_sorted = sorted(matrix.columns.tolist(), key=_cluster_sort_key)
        matrix = matrix[clusters_sorted]

        # ── Sankey ────────────────────────────────────────────────────────────
        min_link = int(min_link or 5)
        filtered_links = []
        active_layers = []
        active_clusters = []
        active_layer_set = set()
        active_cluster_set = set()

        for _, row in pivot.iterrows():
            lyr, cid, cnt = row['unit_C'], row[cluster_col], int(row['count'])
            if cnt < min_link:
                continue
            filtered_links.append((lyr, cid, cnt))
            if lyr not in active_layer_set:
                active_layer_set.add(lyr)
                active_layers.append(lyr)
            if cid not in active_cluster_set:
                active_cluster_set.add(cid)
                active_clusters.append(cid)

        active_layers = [lyr for lyr in layers_sorted if lyr in active_layer_set]
        active_clusters = [cid for cid in clusters_sorted if cid in active_cluster_set]

        n_layers = len(active_layers)
        n_clusters = len(active_clusters)
        layer_idx = {lyr: i for i, lyr in enumerate(active_layers)}
        cluster_idx = {c: n_layers + i for i, c in enumerate(active_clusters)}
        filtered_links.sort(key=lambda item: (layer_idx.get(item[0], 10**9), _cluster_sort_key(item[1])))

        layer_colors = [
            _color_with_alpha(CLUSTER_COLORS[i % len(CLUSTER_COLORS)], 0.78)
            for i in range(n_layers)
        ]

        sources, targets, values, link_labels, link_colors = [], [], [], [], []
        for lyr, cid, cnt in filtered_links:
            if lyr not in layer_idx or cid not in cluster_idx:
                continue
            layer_color = layer_colors[layer_idx[lyr]]
            sources.append(layer_idx[lyr])
            targets.append(cluster_idx[cid])
            values.append(cnt)
            link_labels.append(f'{lyr} → 簇{cid}：{cnt} 片')
            link_colors.append(_color_with_alpha(layer_color, 0.48))

        # 节点颜色
        cluster_colors = [
            CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] if str(c).lstrip('-').isdigit() else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for i, c in enumerate(active_clusters)
        ]
        node_labels = [str(lyr) for lyr in active_layers] + [f'簇{c}' for c in active_clusters]
        node_colors = layer_colors + cluster_colors
        layer_weights = [sum(cnt for lyr, _, cnt in filtered_links if lyr == active_layer) for active_layer in active_layers]
        cluster_weights = [sum(cnt for _, cid, cnt in filtered_links if cid == active_cluster) for active_cluster in active_clusters]
        node_x = ([0.08] * n_layers) + ([0.92] * n_clusters)
        node_y = _ordered_node_positions(layer_weights) + _ordered_node_positions(cluster_weights)
        max_column_nodes = max(n_layers, n_clusters)
        node_thickness, node_pad, sankey_height, sankey_font_size = _sankey_node_style(max_column_nodes)

        if values:
            sankey_fig = go.Figure(go.Sankey(
                arrangement='snap',
                node=dict(
                    label=node_labels,
                    color=node_colors,
                    pad=node_pad,
                    thickness=node_thickness,
                    x=node_x,
                    y=node_y,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    label=link_labels,
                    color=link_colors,
                ),
            ))
            sankey_fig.update_layout(
                title=f'层位 → 簇 流向图｜最小连线 ≥ {min_link} 片',
                margin=dict(l=48, r=48, t=60, b=36),
                font_size=sankey_font_size,
                height=sankey_height,
            )
        else:
            sankey_fig = go.Figure()
            sankey_fig.update_layout(
                title='无满足阈值的连线，请降低最小连线值',
                margin=dict(l=48, r=48, t=60, b=36),
                height=500,
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
