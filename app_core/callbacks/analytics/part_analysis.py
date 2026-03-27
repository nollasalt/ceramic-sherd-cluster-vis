"""器部分布分析回调：part_C × cluster_id 混淆热力图 + 熵值语义标签。"""

import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.express as px

from app_core.data_cache import get_data_cache
from performance_utils import cache_plot_result


def _entropy(probs):
    """Shannon 熵（以 2 为底），用于衡量器部分布的均匀程度。"""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def _semantic_label(dominant_part, dominant_pct, entropy_val, n_parts):
    """根据主导器部、主导比例和熵值生成语义标签。"""
    max_entropy = np.log2(max(n_parts, 2))
    rel_entropy = entropy_val / max_entropy if max_entropy > 0 else 0

    if dominant_pct >= 0.80:
        return f'单一器部（{dominant_part}）', '#27ae60'
    if dominant_pct >= 0.55:
        return f'以{dominant_part}为主', '#2980b9'
    if rel_entropy >= 0.85:
        return '器部均匀混合', '#e67e22'
    return f'多器部混合', '#8e44ad'


def register_part_analysis_callbacks(app):
    @app.callback(
        [Output('part-val-heatmap', 'figure'),
         Output('part-val-metrics', 'children'),
         Output('part-val-detail', 'children')],
        [Input('visualization-tabs', 'value'),
         Input('part-val-norm', 'value'),
         Input('part-val-min-samples', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data'),
    )
    @cache_plot_result
    def render_part_analysis(
        tab_value, norm_mode, min_samples,
        selected_clusters, selected_units, selected_parts, selected_types,
        data_store,
    ):
        """渲染器部分布矩阵、熵值统计和逐簇明细表。"""
        if tab_value != 'part-analysis':
            return dash.no_update, dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']

        if 'part_C' not in df.columns:
            empty = px.imshow([[0]], title='无 part_C 列', color_continuous_scale='Blues')
            msg = html.Div('数据中无 part_C 字段', style={'color': '#888', 'padding': '8px'})
            return empty, msg, msg

        # 只取需要的列
        need_cols = [cluster_col, 'part_C']
        for c in ['unit_C', 'type_C']:
            if c in df.columns:
                need_cols.append(c)
        dff = df[need_cols]

        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        _empty_fig = px.imshow([[0]], title='暂无数据', color_continuous_scale='RdYlGn')
        _empty_msg = html.Div('暂无数据', style={'color': '#888', 'padding': '8px'})

        work = dff[[cluster_col, 'part_C']].dropna()
        if len(work) < 2:
            return _empty_fig, _empty_msg, _empty_msg

        # ── 构建 pivot（行=器部，列=簇），过滤小簇 ────────────────────────
        min_samples = int(min_samples or 5)
        cluster_counts = work[cluster_col].value_counts()
        valid_clusters = cluster_counts[cluster_counts >= min_samples].index
        work = work[work[cluster_col].isin(valid_clusters)].copy()
        work[cluster_col] = work[cluster_col].astype(str)

        if len(work) == 0:
            return _empty_fig, _empty_msg, _empty_msg

        parts_order = sorted(work['part_C'].unique())
        clusters_order = sorted(work[cluster_col].unique(), key=lambda x: int(x) if x.isdigit() else x)

        pivot = (
            work.groupby(['part_C', cluster_col], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(index=parts_order, columns=clusters_order, fill_value=0)
        )

        # ── 归一化 ────────────────────────────────────────────────────────
        norm_mode = norm_mode or 'by_cluster'
        if norm_mode == 'by_part':
            row_sums = pivot.sum(axis=1).replace(0, np.nan)
            plot_matrix = pivot.div(row_sums, axis=0).fillna(0)
            colorbar_title = '比例（按器部）'
            text_fmt = '.1%'
            title_suffix = '按器部行归一化'
        elif norm_mode == 'by_cluster':
            col_sums = pivot.sum(axis=0).replace(0, np.nan)
            plot_matrix = pivot.div(col_sums, axis=1).fillna(0)
            colorbar_title = '占簇内比例'
            text_fmt = '.1%'
            title_suffix = '按簇列归一化'
        else:
            plot_matrix = pivot.astype(float)
            colorbar_title = '陶片数'
            text_fmt = '.0f'
            title_suffix = '绝对数'

        # ── 热力图 ────────────────────────────────────────────────────────
        fig = px.imshow(
            plot_matrix,
            labels=dict(x='簇编号', y='器部（part_C）', color=colorbar_title),
            aspect='auto',
            color_continuous_scale='RdYlGn_r',
            title=f'器部 × 簇 分布矩阵（{title_suffix}，≥{min_samples}样本簇）',
        )
        n_cells = plot_matrix.shape[0] * plot_matrix.shape[1]
        if n_cells <= 300:
            fig.update_traces(
                texttemplate='%{z:' + text_fmt + '}',
                textfont_size=9,
            )
        fig.update_layout(
            margin=dict(l=100, r=40, t=60, b=80),
            xaxis=dict(title='簇编号', tickangle=-45, side='bottom'),
            yaxis=dict(title='器部'),
            coloraxis_colorbar=dict(title=colorbar_title, len=0.8),
        )

        # ── 逐簇熵值与语义标签 ────────────────────────────────────────────
        n_parts = len(parts_order)
        cluster_stats = []
        for cid in clusters_order:
            if cid not in pivot.columns:
                continue
            counts = pivot[cid]
            total = int(counts.sum())
            if total == 0:
                continue
            probs = counts / total
            ent = _entropy(probs.values)
            dom_part = counts.idxmax()
            dom_pct = float(counts.max()) / total
            label, color = _semantic_label(dom_part, dom_pct, ent, n_parts)
            cluster_stats.append({
                'cluster': cid,
                'total': total,
                'dominant': dom_part,
                'dom_pct': dom_pct,
                'entropy': ent,
                'label': label,
                'color': color,
            })

        # ── 语义分布汇总卡片 ──────────────────────────────────────────────
        from collections import Counter
        label_counts = Counter(s['label'].split('（')[0] for s in cluster_stats)
        summary_items = []
        label_colors = {
            '单一器部': '#27ae60',
            '以': '#2980b9',
            '器部均匀混合': '#e67e22',
            '多器部混合': '#8e44ad',
        }
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            color = next((v for k, v in label_colors.items() if lbl.startswith(k)), '#555')
            summary_items.append(html.Div([
                html.Span(f'{cnt} 簇', style={
                    'fontSize': '20px', 'fontWeight': '700', 'color': color,
                }),
                html.Div(lbl, style={'fontSize': '11px', 'color': '#666', 'marginTop': '2px'}),
            ], style={
                'padding': '8px 12px',
                'border': f'1px solid {color}33',
                'borderRadius': '8px',
                'backgroundColor': color + '0d',
                'flex': '1', 'minWidth': '100px', 'textAlign': 'center',
            }))

        avg_entropy = np.mean([s['entropy'] for s in cluster_stats]) if cluster_stats else 0
        max_entropy = np.log2(max(n_parts, 2))
        metrics_block = html.Div([
            html.Div('器部语义分布', style={
                'fontSize': '13px', 'fontWeight': '700', 'color': '#2c3e50', 'marginBottom': '10px',
            }),
            html.Div(
                f'共 {len(cluster_stats)} 个有效簇（≥{min_samples}样本），'
                f'器部 {n_parts} 种，平均熵 {avg_entropy:.2f} / {max_entropy:.2f}',
                style={'fontSize': '11px', 'color': '#999', 'marginBottom': '8px'},
            ),
            html.Div(summary_items, style={
                'display': 'flex', 'gap': '6px', 'flexWrap': 'wrap', 'marginBottom': '10px',
            }),
            html.Div(
                '熵值越低说明该簇器部越单一（可能捕获器形特征）；'
                '熵值越高说明器部混合（可能捕获纹饰或胎土特征）。',
                style={
                    'fontSize': '12px', 'color': '#555', 'lineHeight': '1.7',
                    'backgroundColor': '#fff', 'borderRadius': '8px',
                    'padding': '8px 12px', 'border': '1px solid #e4e8ef',
                },
            ),
        ])

        # ── 逐簇明细表 ────────────────────────────────────────────────────
        _TH = {
            'padding': '7px 10px', 'backgroundColor': '#2c3e50', 'color': '#fff',
            'fontSize': '12px', 'fontWeight': '600', 'textAlign': 'left', 'whiteSpace': 'nowrap',
        }
        _TD = {
            'padding': '6px 10px', 'fontSize': '12px', 'color': '#333',
            'borderBottom': '1px solid #f0f0f0', 'verticalAlign': 'middle',
        }
        table_rows = []
        for i, s in enumerate(cluster_stats):
            color = s['color']
            label_badge = html.Span(s['label'], style={
                'color': color, 'fontWeight': '600',
                'backgroundColor': color + '18',
                'padding': '2px 7px', 'borderRadius': '10px',
                'border': f'1px solid {color}44',
                'fontSize': '11px',
            })
            row_bg = '#fafbfc' if i % 2 == 0 else '#fff'
            table_rows.append(html.Tr([
                html.Td(str(s['cluster']), style={**_TD, 'fontWeight': '600'}),
                html.Td(str(s['total']), style=_TD),
                html.Td(str(s['dominant']), style=_TD),
                html.Td(f"{s['dom_pct']:.1%}", style=_TD),
                html.Td(f"{s['entropy']:.2f}", style=_TD),
                html.Td(label_badge, style=_TD),
            ], style={'backgroundColor': row_bg}))

        detail_table = html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th(t, style=_TH)
                    for t in ['簇', '样本数', '主导器部', '主导占比', '熵值', '语义']
                ])),
                html.Tbody(table_rows),
            ], style={'borderCollapse': 'collapse', 'width': '100%'}),
        ], style={
            'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto',
            'border': '1px solid #e4e8ef', 'borderRadius': '10px', 'overflow': 'hidden',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.05)',
        })

        return fig, metrics_block, detail_table
