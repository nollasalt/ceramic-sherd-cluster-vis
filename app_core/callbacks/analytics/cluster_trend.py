"""跨层簇趋势分析回调：折线图 + 趋势线 + 自动分类（兴起/衰落/瞬间/稳定）。"""

import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app_core.data_cache import get_data_cache
from app_core.callbacks.analytics.stratigraphy import _sorted_layers
from app_core.utils import CLUSTER_COLORS
from performance_utils import cache_plot_result


# ── 趋势分类 ──────────────────────────────────────────────────────────────────

_TYPE_META = {
    'rising':   {'label': '兴起型 ↑', 'color': '#27ae60', 'dash': 'solid'},
    'declining': {'label': '衰落型 ↓', 'color': '#e74c3c', 'dash': 'solid'},
    'transient': {'label': '瞬间型 ⚡', 'color': '#e67e22', 'dash': 'dot'},
    'stable':   {'label': '稳定型 ─', 'color': '#2980b9', 'dash': 'solid'},
}


def _classify_trend(xs, ys, n_layers_present, slope, slope_thresh=0.005, transient_max=2):
    """将一条趋势线分类为兴起/衰落/瞬间/稳定。

    Args:
        xs: 层位 X 坐标序列（整数索引）
        ys: 各层占比/数量序列
        n_layers_present: 出现层数
        slope: 线性拟合斜率（基于归一化 X）
        slope_thresh: 斜率阈值（相对占比单位）
        transient_max: 最多出现多少层算"瞬间型"
    """
    if n_layers_present <= transient_max:
        return 'transient'
    if slope > slope_thresh:
        return 'rising'
    if slope < -slope_thresh:
        return 'declining'
    return 'stable'


# ── 主回调 ────────────────────────────────────────────────────────────────────

def register_cluster_trend_callbacks(app):

    # ── 初始化下拉选项 ─────────────────────────────────────────────────────
    @app.callback(
        Output('trend-unit-filter', 'options'),
        Output('trend-cluster-filter', 'options'),
        Input('visualization-tabs', 'value'),
        State('data-store', 'data'),
    )
    def init_trend_filters(tab_value, _):
        if tab_value != 'cluster-trend':
            return dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']

        units = [u for u in df['unit_C'].dropna().unique() if str(u).strip()]
        unit_opts = [{'label': str(u), 'value': u} for u in _sorted_layers(units)]

        clusters = sorted(df[cluster_col].dropna().unique())
        cluster_opts = [{'label': f'簇 {c}', 'value': c} for c in clusters]

        return unit_opts, cluster_opts

    # ── 快捷筛选按钮 ───────────────────────────────────────────────────
    @app.callback(
        Output('trend-unit-filter', 'value'),
        Input('trend-unit-select-all', 'n_clicks'),
        Input('trend-unit-clear', 'n_clicks'),
        Input('trend-unit-main', 'n_clicks'),
        State('trend-unit-filter', 'options'),
        prevent_initial_call=True,
    )
    def handle_unit_shortcuts(n_all, n_clear, n_main, options):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'trend-unit-select-all':
            return [opt['value'] for opt in options]
        elif button_id == 'trend-unit-clear':
            return []
        elif button_id == 'trend-unit-main':
            # 仅主要层：排除包含"混"、"扰"等关键字的层位
            exclude_keywords = ['混', '扰', '乱']
            return [opt['value'] for opt in options
                    if not any(kw in str(opt['value']) for kw in exclude_keywords)]

        return dash.no_update

    # ── 主渲染回调 ─────────────────────────────────────────────────────────
    @app.callback(
        Output('cluster-trend-chart', 'figure'),
        Output('cluster-trend-summary', 'children'),
        Output('cluster-trend-detail', 'children'),
        Input('visualization-tabs', 'value'),
        Input('trend-unit-filter', 'value'),
        Input('trend-cluster-filter', 'value'),
        Input('trend-type-filter', 'value'),
        Input('trend-y-mode', 'value'),
        Input('trend-min-layers', 'value'),
        State('data-store', 'data'),
    )
    @cache_plot_result
    def render_cluster_trend(
        tab_value, sel_units, sel_clusters, type_filter,
        y_mode, min_layers, _data_store,
    ):
        if tab_value != 'cluster-trend':
            return dash.no_update, dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']

        # 只取需要的列
        dff = df[[cluster_col, 'unit_C']].dropna(subset=['unit_C'])
        dff = dff[dff['unit_C'].astype(str).str.strip() != '']

        if len(dff) == 0:
            empty = go.Figure()
            empty.update_layout(title='暂无地层数据（需要 unit_C 列）')
            msg = html.Div('暂无数据', style={'color': '#888', 'padding': '8px'})
            return empty, msg, msg

        if sel_units:
            dff = dff[dff['unit_C'].isin(sel_units)]
        if sel_clusters:
            dff = dff[dff[cluster_col].isin(sel_clusters)]

        # ── 构建 layer × cluster 计数矩阵 ────────────────────────────────
        pivot = (
            dff.groupby(['unit_C', cluster_col], observed=True)
            .size()
            .reset_index(name='count')
        )
        matrix = pivot.pivot_table(
            index='unit_C', columns=cluster_col, values='count', fill_value=0, observed=True
        )
        layers_sorted = _sorted_layers(matrix.index.tolist())
        matrix = matrix.reindex(layers_sorted)

        # 层位 X 轴：从左=最新(浅)到右=最早(深)
        x_indices = list(range(len(layers_sorted)))  # 用于拟合
        x_labels = [str(lyr) for lyr in layers_sorted]

        # 层总量（用于按层归一化）
        layer_totals = dff.groupby('unit_C', observed=True).size().reindex(layers_sorted, fill_value=0)

        min_layers = int(min_layers or 2)
        type_filter = set(type_filter or list(_TYPE_META.keys()))

        # ── 计算各簇趋势 ──────────────────────────────────────────────────
        clusters_in_matrix = [c for c in matrix.columns]
        stats = []  # 存储每个簇的统计信息

        for cid in clusters_in_matrix:
            counts = matrix[cid].values.astype(float)
            totals = layer_totals.values.astype(float)

            if y_mode == 'by_layer':
                ys = np.where(totals > 0, counts / totals, 0.0)
            else:
                ys = counts

            n_present = int((counts > 0).sum())
            if n_present < min_layers:
                continue

            # 线性趋势拟合（仅用有数据的层）
            mask = counts > 0
            if mask.sum() >= 2:
                xi = np.array(x_indices)[mask]
                yi = ys[mask]
                # 归一化 X 到 [0,1] 使斜率具有可比性
                xi_norm = (xi - xi.min()) / max(xi.max() - xi.min(), 1)
                coeffs = np.polyfit(xi_norm, yi, 1)
                slope = float(coeffs[0])
                trend_y = np.polyval(coeffs, (np.array(x_indices) - xi.min()) / max(xi.max() - xi.min(), 1))
            else:
                slope = 0.0
                trend_y = ys.copy()

            trend_type = _classify_trend(x_indices, ys, n_present, slope)
            stats.append({
                'cluster': cid,
                'ys': ys,
                'trend_y': trend_y,
                'slope': slope,
                'n_present': n_present,
                'type': trend_type,
                'peak_layer': layers_sorted[int(np.argmax(ys))],
                'peak_val': float(np.max(ys)),
            })

        if not stats:
            empty = go.Figure()
            empty.update_layout(title='无满足条件的簇（调整最少出现层数或簇筛选）')
            msg = html.Div('无数据', style={'color': '#888', 'padding': '8px'})
            return empty, msg, msg

        # ── 按类型过滤 ────────────────────────────────────────────────────
        stats_show = [s for s in stats if s['type'] in type_filter]

        # ── 折线图 ────────────────────────────────────────────────────────
        fig = go.Figure()

        # 若显示簇数过多给出警告（仍渲染，但提示用户）
        too_many = len(stats_show) > 30

        for i, s in enumerate(stats_show):
            cid = s['cluster']
            meta = _TYPE_META[s['type']]
            color = meta['color']
            cid_int = int(cid) if str(cid).lstrip('-').isdigit() else i
            line_color = CLUSTER_COLORS[cid_int % len(CLUSTER_COLORS)]

            y_label = '层内占比' if y_mode == 'by_layer' else '陶片数'
            hover_text = [
                f'簇 {cid}<br>{x_labels[j]}<br>{y_label}: {s["ys"][j]:.2%}'
                if y_mode == 'by_layer' else
                f'簇 {cid}<br>{x_labels[j]}<br>数量: {int(s["ys"][j])}'
                for j in range(len(x_labels))
            ]

            # 实线：实际占比
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=s['ys'],
                mode='lines+markers',
                name=f'簇{cid} [{meta["label"]}]',
                line=dict(color=line_color, width=2),
                marker=dict(size=5, color=line_color),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=str(cid),
            ))

            # 虚线：趋势线
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=s['trend_y'],
                mode='lines',
                name=f'簇{cid} 趋势',
                line=dict(color=line_color, width=1.5, dash='dash'),
                opacity=0.55,
                showlegend=False,
                hoverinfo='skip',
                legendgroup=str(cid),
            ))

        y_axis_title = '该层内占比 (%)' if y_mode == 'by_layer' else '陶片数'
        title_txt = f'簇跨层占比趋势｜共 {len(stats_show)} 条'
        if too_many:
            title_txt += '（建议用簇筛选缩小范围）'

        fig.update_layout(
            title=title_txt,
            xaxis=dict(
                title='地层层位（左=新，右=老）',
                tickangle=-30,
                tickmode='array',
                tickvals=x_indices,
                ticktext=x_labels,
                range=[-0.5, len(x_indices) - 0.5],
            ),
            yaxis=dict(title=y_axis_title, tickformat='.1%' if y_mode == 'by_layer' else ''),
            legend=dict(
                orientation='v', x=1.01, y=1,
                font=dict(size=11),
                tracegroupgap=2,
            ),
            margin=dict(l=60, r=180, t=60, b=80),
            hovermode='x unified' if len(stats_show) <= 10 else 'closest',
        )

        # ── 趋势分类汇总 ──────────────────────────────────────────────────
        from collections import Counter
        type_counts = Counter(s['type'] for s in stats)  # 全量统计
        summary_cards = []
        for ttype, meta in _TYPE_META.items():
            cnt_all = type_counts.get(ttype, 0)
            cnt_show = sum(1 for s in stats_show if s['type'] == ttype)
            color = meta['color']
            summary_cards.append(html.Div([
                html.Div(str(cnt_all), style={
                    'fontSize': '22px', 'fontWeight': '700', 'color': color,
                }),
                html.Div(meta['label'], style={
                    'fontSize': '11px', 'color': '#555', 'marginTop': '2px',
                }),
                html.Div(f'显示 {cnt_show}', style={
                    'fontSize': '10px', 'color': '#aaa',
                }),
            ], style={
                'padding': '8px 10px', 'textAlign': 'center',
                'border': f'1px solid {color}44',
                'borderRadius': '8px', 'backgroundColor': color + '0d',
                'flex': '1', 'minWidth': '80px',
            }))

        n_total = len(stats)
        summary_block = html.Div([
            html.Div(
                f'共 {n_total} 个有效簇（≥{min_layers} 层），显示 {len(stats_show)} 个',
                style={'fontSize': '11px', 'color': '#999', 'marginBottom': '10px'},
            ),
            html.Div(summary_cards, style={
                'display': 'flex', 'gap': '6px', 'flexWrap': 'wrap', 'marginBottom': '12px',
            }),
            html.Div([
                html.Div('分类标准：', style={'fontWeight': '600', 'fontSize': '11px', 'marginBottom': '4px'}),
                html.Div('兴起型：线性斜率 > 0.5%/层', style={'fontSize': '11px', 'color': '#555'}),
                html.Div('衰落型：线性斜率 < −0.5%/层', style={'fontSize': '11px', 'color': '#555'}),
                html.Div(f'瞬间型：出现层数 ≤ {min(2, min_layers)} 层', style={'fontSize': '11px', 'color': '#555'}),
                html.Div('稳定型：斜率接近 0', style={'fontSize': '11px', 'color': '#555'}),
            ], style={
                'backgroundColor': '#fff', 'borderRadius': '8px',
                'padding': '8px 10px', 'border': '1px solid #e4e8ef',
                'fontSize': '11px',
            }),
        ])

        # ── 逐簇明细表 ────────────────────────────────────────────────────
        _TH = {
            'padding': '6px 8px', 'backgroundColor': '#2c3e50', 'color': '#fff',
            'fontSize': '11px', 'fontWeight': '600', 'textAlign': 'left', 'whiteSpace': 'nowrap',
        }
        _TD = {
            'padding': '5px 8px', 'fontSize': '11px', 'color': '#333',
            'borderBottom': '1px solid #f0f0f0', 'verticalAlign': 'middle',
        }

        # 按斜率绝对值降序排列，最显著的排前面
        stats_sorted = sorted(stats, key=lambda s: -abs(s['slope']))

        rows = []
        for i, s in enumerate(stats_sorted):
            meta = _TYPE_META[s['type']]
            color = meta['color']
            badge = html.Span(meta['label'], style={
                'color': color, 'fontWeight': '600',
                'backgroundColor': color + '18',
                'padding': '1px 6px', 'borderRadius': '8px',
                'border': f'1px solid {color}44', 'fontSize': '10px',
            })
            slope_str = f'{s["slope"]:+.3%}' if y_mode == 'by_layer' else f'{s["slope"]:+.1f}'
            peak_str = f'{s["peak_val"]:.1%}' if y_mode == 'by_layer' else str(int(s['peak_val']))
            rows.append(html.Tr([
                html.Td(str(s['cluster']), style={**_TD, 'fontWeight': '600'}),
                html.Td(badge, style=_TD),
                html.Td(str(s['n_present']), style=_TD),
                html.Td(slope_str, style={**_TD, 'fontFamily': 'monospace'}),
                html.Td(str(s['peak_layer']), style=_TD),
                html.Td(peak_str, style=_TD),
            ], style={'backgroundColor': '#fafbfc' if i % 2 == 0 else '#fff'}))

        detail_table = html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th(t, style=_TH)
                    for t in ['簇', '类型', '出现层数', '趋势斜率', '峰值层', '峰值']
                ])),
                html.Tbody(rows),
            ], style={'borderCollapse': 'collapse', 'width': '100%'}),
        ], style={
            'overflowX': 'auto', 'maxHeight': '380px', 'overflowY': 'auto',
            'border': '1px solid #e4e8ef', 'borderRadius': '10px', 'overflow': 'hidden',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.05)',
        })

        return fig, summary_block, detail_table
