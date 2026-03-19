"""器类验证矩阵回调：type_C × cluster_id 混淆热力图 + ARI/NMI 纯度指数。"""

import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.express as px

from app_core.data_cache import get_data_cache
from performance_utils import cache_plot_result


# ── 阈值颜色辅助 ───────────────────────────────────────────────────────────────

def _purity_color(pct):
    """根据纯度百分比返回颜色（绿/橙/红）。"""
    if pct >= 0.70:
        return '#27ae60'
    if pct >= 0.40:
        return '#e67e22'
    return '#e74c3c'


def _metric_color(key, val):
    """根据指标名称和数值返回颜色。"""
    if key == 'ARI':
        return '#27ae60' if val > 0.4 else ('#e67e22' if val > 0.1 else '#e74c3c')
    return '#27ae60' if val > 0.6 else ('#e67e22' if val > 0.3 else '#e74c3c')


# ── 指标解读 ───────────────────────────────────────────────────────────────────

def _interpret_metrics(metrics):
    """根据 ARI/NMI 等指标生成自然语言解读。"""
    if not metrics:
        return '指标计算失败，样本可能过少或类别不足。'

    ari = metrics.get('ARI', 0.0)
    hom = metrics.get('同质性', 0.0)
    comp = metrics.get('完整性', 0.0)

    lines = []
    if ari > 0.4:
        lines.append(
            'ARI 较高（>0.4），算法分组与人工器类标注一致性强，'
            '模型已学到考古学家认可的分类规律。'
        )
    elif ari > 0.1:
        lines.append(
            'ARI 中等（0.1~0.4），分组有参考价值，'
            '但存在明显跨类混入，可结合热力图找出"问题器类"。'
        )
    else:
        lines.append(
            'ARI 偏低（<0.1），算法分组与器类标注差异较大。'
            '可能存在视觉相似但类型不同的器物——这恰是值得深入研究的发现。'
        )

    if hom > comp + 0.15:
        lines.append(
            '同质性 > 完整性：每簇内器类较纯，但同一器类被拆散到多个簇，'
            '可考虑适当增大 K 值。'
        )
    elif comp > hom + 0.15:
        lines.append(
            '完整性 > 同质性：同类器物趋于集中，但每簇混入了多种类型，'
            '可考虑适当减小 K 值。'
        )

    return ' '.join(lines) if lines else '请结合热力图和明细表进行解读。'


# ── 主回调 ────────────────────────────────────────────────────────────────────

def register_type_validation_callbacks(app):
    @app.callback(
        [Output('type-val-heatmap', 'figure'),
         Output('type-val-metrics', 'children'),
         Output('type-val-detail', 'children')],
        [Input('visualization-tabs', 'value'),
         Input('type-val-norm', 'value'),
         Input('type-val-topn', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data'),
    )
    @cache_plot_result
    def render_type_validation(
        tab_value, norm_mode, topn,
        selected_clusters, selected_units, selected_parts, selected_types,
        data_store,
    ):
        """渲染器类验证矩阵、纯度指数和逐器类明细。"""
        if tab_value != 'category-breakdown':
            return dash.no_update, dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        _empty_fig = px.imshow([[0]], title='暂无数据', color_continuous_scale='Blues')
        _empty_msg = html.Div('暂无数据', style={'color': '#888', 'padding': '8px'})

        if 'type_C' not in dff.columns or len(dff) == 0:
            return _empty_fig, _empty_msg, _empty_msg

        work = dff[[cluster_col, 'type_C']].dropna()
        if len(work) < 2:
            return _empty_fig, _empty_msg, _empty_msg

        # ── 筛选 Top-N 器类 ────────────────────────────────────────────────
        topn = int(topn or 15)
        topn = max(5, min(30, topn))
        type_counts = work['type_C'].value_counts()
        top_types = type_counts.head(topn).index.tolist()
        work_top = work[work['type_C'].isin(top_types)].copy()

        # ── 构建 pivot 矩阵（行=器类，列=簇） ──────────────────────────────
        clusters_sorted = sorted(work_top[cluster_col].unique())
        pivot = (
            work_top.groupby(['type_C', cluster_col])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=clusters_sorted, fill_value=0)
        )
        pivot = pivot.loc[[t for t in top_types if t in pivot.index]]

        # ── 归一化 ────────────────────────────────────────────────────────
        norm_mode = norm_mode or 'by_type'
        if norm_mode == 'by_type':
            row_sums = pivot.sum(axis=1).replace(0, np.nan)
            plot_matrix = pivot.div(row_sums, axis=0).fillna(0)
            colorbar_title = '比例（按器类）'
            text_fmt = '.1%'
            title_suffix = '按器类行归一化'
        elif norm_mode == 'by_cluster':
            col_sums = pivot.sum(axis=0).replace(0, np.nan)
            plot_matrix = pivot.div(col_sums, axis=1).fillna(0)
            colorbar_title = '比例（按簇）'
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
            labels=dict(x='簇编号', y='器类（type_C）', color=colorbar_title),
            aspect='auto',
            color_continuous_scale='Blues',
            title=f'器类 × 簇 混淆矩阵（Top-{topn}，{title_suffix}）',
        )
        fig.update_traces(
            texttemplate='%{z:' + text_fmt + '}',
            textfont_size=10,
        )
        fig.update_layout(
            margin=dict(l=120, r=40, t=60, b=80),
            xaxis=dict(title='簇编号', tickangle=-45, side='bottom'),
            yaxis=dict(title='器类'),
            coloraxis_colorbar=dict(title=colorbar_title, len=0.8),
        )

        # ── sklearn 纯度指数（全量，非 Top-N） ─────────────────────────────
        metrics = {}
        try:
            from sklearn.metrics import (
                adjusted_rand_score,
                normalized_mutual_info_score,
                homogeneity_completeness_v_measure,
            )
            labels_true = work['type_C'].astype(str).values
            labels_pred = work[cluster_col].astype(str).values
            metrics['ARI'] = float(adjusted_rand_score(labels_true, labels_pred))
            metrics['NMI'] = float(normalized_mutual_info_score(labels_true, labels_pred))
            h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)
            metrics['同质性'] = float(h)
            metrics['完整性'] = float(c)
            metrics['V-measure'] = float(v)
        except Exception:
            pass

        _METRIC_HINTS = {
            'ARI':     '[-1,1]，越高越一致',
            'NMI':     '[0,1]，越高越一致',
            '同质性':  '[0,1]，每簇只含一种器类',
            '完整性':  '[0,1]，同类全在一簇',
            'V-measure': 'H 与 C 的调和平均',
        }

        metric_cards = []
        for key, val in metrics.items():
            color = _metric_color(key, val)
            metric_cards.append(html.Div([
                html.Div(key, style={
                    'fontSize': '12px', 'color': '#666', 'marginBottom': '4px',
                }),
                html.Div(f'{val:.4f}', style={
                    'fontSize': '20px', 'fontWeight': '700', 'color': color,
                }),
                html.Div(_METRIC_HINTS.get(key, ''), style={
                    'fontSize': '11px', 'color': '#999', 'marginTop': '3px',
                }),
            ], style={
                'padding': '10px 14px',
                'border': f'1px solid {color}33',
                'borderRadius': '8px',
                'backgroundColor': color + '0d',
                'minWidth': '110px',
                'flex': '1',
            }))

        interpretation = _interpret_metrics(metrics)
        sample_hint = f'（基于全量 {len(work)} 条有效记录，器类 {work["type_C"].nunique()} 种，簇 {work[cluster_col].nunique()} 个）'

        metrics_block = html.Div([
            html.Div('纯度指数', style={
                'fontSize': '13px', 'fontWeight': '700',
                'color': '#2c3e50', 'marginBottom': '10px',
            }),
            html.Div(sample_hint, style={
                'fontSize': '11px', 'color': '#999', 'marginBottom': '8px',
            }),
            html.Div(metric_cards, style={
                'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '10px',
            }),
            html.Div(interpretation, style={
                'fontSize': '12px', 'color': '#555', 'lineHeight': '1.7',
                'backgroundColor': '#fff', 'borderRadius': '8px',
                'padding': '8px 12px', 'border': '1px solid #e4e8ef',
            }),
        ])

        # ── 逐器类纯度明细表 ───────────────────────────────────────────────
        _TH = {
            'padding': '7px 10px', 'backgroundColor': '#2c3e50', 'color': '#fff',
            'fontSize': '12px', 'fontWeight': '600', 'textAlign': 'left', 'whiteSpace': 'nowrap',
        }
        _TD = {
            'padding': '6px 10px', 'fontSize': '12px', 'color': '#333',
            'borderBottom': '1px solid #f0f0f0', 'verticalAlign': 'middle',
        }

        table_rows = []
        for i, type_name in enumerate(top_types):
            if type_name not in pivot.index:
                continue
            row_counts = pivot.loc[type_name]
            total = int(row_counts.sum())
            if total == 0:
                continue
            dom_cluster = row_counts.idxmax()
            dom_pct = float(row_counts.max()) / total
            spread = int((row_counts > 0).sum())

            color = _purity_color(dom_pct)
            purity_badge = html.Span(f'{dom_pct:.1%}', style={
                'color': color, 'fontWeight': '700',
                'backgroundColor': color + '18',
                'padding': '2px 7px', 'borderRadius': '10px',
                'border': f'1px solid {color}44',
            })
            row_bg = '#fafbfc' if i % 2 == 0 else '#fff'
            table_rows.append(html.Tr([
                html.Td(str(type_name), style={**_TD, 'fontWeight': '600'}),
                html.Td(str(total), style=_TD),
                html.Td(str(dom_cluster), style=_TD),
                html.Td(purity_badge, style=_TD),
                html.Td(str(spread), style=_TD),
            ], style={'backgroundColor': row_bg}))

        purity_hint = html.Div([
            html.Span('纯度 = 主导簇陶片数 / 该器类总数。', style={'color': '#888', 'fontSize': '11px'}),
            html.Span(' 绿≥70%，橙≥40%，红<40%', style={'color': '#aaa', 'fontSize': '11px'}),
        ], style={'marginTop': '8px', 'padding': '0 2px'})

        detail_table = html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th(t, style=_TH)
                    for t in ['器类', '样本数', '主导簇', '纯度', '分布到 N 簇']
                ])),
                html.Tbody(table_rows),
            ], style={'borderCollapse': 'collapse', 'width': '100%'}),
            purity_hint,
        ], style={
            'overflowX': 'auto',
            'border': '1px solid #e4e8ef',
            'borderRadius': '10px',
            'overflow': 'hidden',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.05)',
        })

        return fig, metrics_block, detail_table
