"""簇共现分析回调：共现矩阵热力图、聚类树状图、统计摘要。"""

import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from app_core.data_cache import get_data_cache
from app_core.callbacks.analytics.stratigraphy import _sorted_layers
from performance_utils import cache_plot_result


# ── 共现矩阵计算 ──────────────────────────────────────────────────────────────

def _build_cooc_matrix(df, cluster_col, sel_units=None):
    """返回 (matrix_df, unit_cluster_sets)。

    matrix_df: DataFrame，index/columns 均为簇编号，值为共现层数。
    unit_cluster_sets: {unit: set(clusters)} 原始共现数据。
    """
    dff = df[df['unit_C'].notna() & df['unit_C'].astype(str).str.strip().ne('')].copy()
    if sel_units:
        dff = dff[dff['unit_C'].isin(sel_units)]

    # 每个层位包含的簇集合
    unit_cluster_sets = (
        dff.groupby('unit_C', observed=True)[cluster_col]
        .apply(lambda s: set(s.dropna().unique()))
        .to_dict()
    )

    all_clusters = sorted(dff[cluster_col].dropna().unique())
    n = len(all_clusters)
    idx = {c: i for i, c in enumerate(all_clusters)}
    matrix = np.zeros((n, n), dtype=int)

    for unit, clusters in unit_cluster_sets.items():
        cl = list(clusters)
        for i in range(len(cl)):
            for j in range(i, len(cl)):
                a, b = idx[cl[i]], idx[cl[j]]
                matrix[a][b] += 1
                if a != b:
                    matrix[b][a] += 1

    return pd.DataFrame(matrix, index=all_clusters, columns=all_clusters), unit_cluster_sets


def _normalize(matrix_df, mode):
    """归一化共现矩阵。返回 float DataFrame。"""
    m = matrix_df.values.astype(float)
    diag = np.diag(m).copy()  # 每个簇自身出现的层数

    if mode == 'raw':
        return pd.DataFrame(m, index=matrix_df.index, columns=matrix_df.columns)

    if mode == 'jaccard':
        n = len(diag)
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                union = diag[i] + diag[j] - m[i][j]
                out[i][j] = m[i][j] / union if union > 0 else 0.0
        return pd.DataFrame(out, index=matrix_df.index, columns=matrix_df.columns)

    if mode == 'conditional':
        # P(j|i) = cooc(i,j) / occ(i)
        out = np.where(diag[:, None] > 0, m / diag[:, None], 0.0)
        return pd.DataFrame(out, index=matrix_df.index, columns=matrix_df.columns)

    return pd.DataFrame(m, index=matrix_df.index, columns=matrix_df.columns)


# ── 回调注册 ──────────────────────────────────────────────────────────────────

def register_cooccurrence_callbacks(app):

    @app.callback(
        Output('cooc-unit-filter', 'options'),
        Input('visualization-tabs', 'value'),
        State('data-store', 'data'),
    )
    def init_cooc_unit_filter(tab_value, _):
        if tab_value != 'cooccurrence':
            return dash.no_update
        data_cache = get_data_cache()
        df = data_cache['df']
        units = [u for u in df['unit_C'].dropna().unique() if str(u).strip()]
        units_sorted = _sorted_layers(units)
        return [{'label': str(u), 'value': u} for u in units_sorted]

    @app.callback(
        Output('cooc-dendrogram', 'figure'),
        Output('cooc-heatmap', 'figure'),
        Output('cooc-stats', 'children'),
        Input('visualization-tabs', 'value'),
        Input('cooc-unit-filter', 'value'),
        Input('cooc-norm-mode', 'value'),
        Input('cooc-min-count', 'value'),
        Input('cooc-linkage', 'value'),
        State('data-store', 'data'),
    )
    @cache_plot_result
    def render_cooccurrence(tab_value, sel_units, norm_mode, min_count, linkage_method, _):
        if tab_value != 'cooccurrence':
            return dash.no_update, dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
        min_count = int(min_count or 1)

        raw_matrix, unit_cluster_sets = _build_cooc_matrix(df, cluster_col, sel_units)

        if raw_matrix.empty or len(raw_matrix) < 2:
            empty = go.Figure().update_layout(title='数据不足，无法构建共现矩阵')
            return empty, empty, html.Div('暂无数据', style={'color': '#666'})

        # 按最小共现层数过滤（对角线 = 自身出现层数，不参与过滤）
        keep_mask = np.zeros(len(raw_matrix), dtype=bool)
        m = raw_matrix.values
        for i in range(len(m)):
            # 只要与任一其他簇共现次数 >= min_count 则保留
            off_diag = np.concatenate([m[i, :i], m[i, i+1:]])
            if off_diag.max() >= min_count:
                keep_mask[i] = True

        if keep_mask.sum() < 2:
            empty = go.Figure().update_layout(title=f'无簇满足最小共现层数 ≥ {min_count}，请降低阈值')
            return empty, empty, html.Div('降低最小共现层数后重试', style={'color': '#888'})

        raw_filtered = raw_matrix.loc[keep_mask, keep_mask]
        norm_df = _normalize(raw_filtered, norm_mode)

        clusters_list = list(norm_df.index)
        labels = [f'簇{c}' for c in clusters_list]
        n = len(clusters_list)

        # ── 层次聚类排序 ─────────────────────────────────────────────────────
        np.fill_diagonal(norm_df.values, 1.0 if norm_mode != 'raw' else norm_df.values.max())

        # 距离矩阵（1 - 相似度，对角线置 0）
        sim = norm_df.values.copy()
        np.fill_diagonal(sim, 1.0)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0, None)

        # 对 ward 方法要求欧式距离，改用 average 兜底
        lm = linkage_method
        if lm == 'ward' and norm_mode != 'raw':
            lm = 'average'

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method=lm)
        dendro_info = dendrogram(Z, no_plot=True)
        leaf_order = dendro_info['leaves']

        reordered_labels = [labels[i] for i in leaf_order]
        reordered_sim = norm_df.values[np.ix_(leaf_order, leaf_order)]
        # 对角线设回 1（自身与自身相似度为 1）
        np.fill_diagonal(reordered_sim, 1.0 if norm_mode != 'raw' else raw_filtered.values.diagonal().max())

        # ── 树状图 ────────────────────────────────────────────────────────────
        icoord = np.array(dendro_info['icoord'])
        dcoord = np.array(dendro_info['dcoord'])
        dendro_traces = []
        for xs, ys in zip(icoord, dcoord):
            dendro_traces.append(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color='#2c6fad', width=1.5),
                hoverinfo='skip',
            ))

        # X 轴刻度对应叶节点位置（icoord 以 5 为间隔）
        tick_vals = np.arange(5, 10 * n, 10)
        dendro_fig = go.Figure(data=dendro_traces)
        dendro_fig.update_layout(
            showlegend=False,
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=reordered_labels,
                tickangle=45,
                showgrid=False,
            ),
            yaxis=dict(title='距离', showgrid=False),
            margin=dict(l=50, r=20, t=30, b=60),
            plot_bgcolor='white',
            title=f'簇聚类树状图（{linkage_method} 联接，基于 {norm_mode} 共现相似度）',
        )

        # ── 热力图 ────────────────────────────────────────────────────────────
        color_label_map = {
            'raw': '共现层数',
            'jaccard': 'Jaccard 相似度',
            'conditional': '条件概率 P(j|i)',
        }
        color_label = color_label_map.get(norm_mode, '共现值')

        heatmap_fig = go.Figure(go.Heatmap(
            z=reordered_sim,
            x=reordered_labels,
            y=reordered_labels,
            colorscale='Blues',
            colorbar=dict(title=color_label),
            hovertemplate='%{y} × %{x}<br>' + color_label + ': %{z:.3f}<extra></extra>',
        ))
        heatmap_fig.update_layout(
            title=f'共现矩阵（{color_label}，行列按树状图排序）',
            margin=dict(l=80, r=20, t=50, b=80),
            xaxis=dict(tickangle=45, side='bottom'),
            yaxis=dict(autorange='reversed'),
        )

        # ── 统计摘要 ──────────────────────────────────────────────────────────
        raw_m = raw_filtered.values.copy()
        np.fill_diagonal(raw_m, 0)  # 排除自身

        # 最高共现对（原始计数）
        upper = np.triu(raw_m, k=1)
        if upper.max() > 0:
            top_idx = np.unravel_index(upper.argmax(), upper.shape)
            top_a = clusters_list[top_idx[0]]
            top_b = clusters_list[top_idx[1]]
            top_val = int(upper[top_idx])
        else:
            top_a = top_b = '-'
            top_val = 0

        # 最高 Jaccard 对
        if norm_mode == 'jaccard':
            jac_m = norm_df.values.copy()
            np.fill_diagonal(jac_m, 0)
            jac_upper = np.triu(jac_m, k=1)
            if jac_upper.max() > 0:
                jac_idx = np.unravel_index(jac_upper.argmax(), jac_upper.shape)
                jac_a = clusters_list[jac_idx[0]]
                jac_b = clusters_list[jac_idx[1]]
                jac_val = float(jac_upper[jac_idx])
            else:
                jac_a = jac_b = '-'
                jac_val = 0.0
        else:
            jac_a = jac_b = None

        # 孤立簇：与所有其他簇共现均为 0
        isolated = [clusters_list[i] for i in range(len(clusters_list)) if raw_m[i].max() == 0]

        # 平均共现层数（非对角线均值）
        off_vals = upper[upper > 0]
        avg_cooc = float(off_vals.mean()) if len(off_vals) > 0 else 0.0

        n_units_used = len(unit_cluster_sets)

        items = [
            _stat_row('分析层位数', str(n_units_used)),
            _stat_row('参与簇数', str(n)),
            _stat_row('平均共现层数', f'{avg_cooc:.1f}'),
            html.Hr(style={'margin': '10px 0'}),
            _stat_row('最高共现对（原始）',
                      f'簇{top_a} & 簇{top_b}', sub=f'{top_val} 层共现'),
        ]
        if jac_a is not None:
            items.append(_stat_row('最高 Jaccard 对',
                                   f'簇{jac_a} & 簇{jac_b}', sub=f'{jac_val:.3f}'))
        if isolated:
            items.append(html.Hr(style={'margin': '10px 0'}))
            items.append(_stat_row('孤立簇（无共现）',
                                   '、'.join(f'簇{c}' for c in isolated[:6]),
                                   sub=f'共 {len(isolated)} 个' if len(isolated) > 1 else ''))

        return dendro_fig, heatmap_fig, html.Div(items)


def _stat_row(label, value, sub=None):
    return html.Div([
        html.Div(label, style={'fontSize': '11px', 'color': '#888', 'marginBottom': '1px'}),
        html.Div(value, style={'fontWeight': '600', 'fontSize': '14px', 'color': '#222'}),
        *([] if not sub else [html.Div(sub, style={'fontSize': '11px', 'color': '#666'})]),
    ], style={'marginBottom': '10px'})
