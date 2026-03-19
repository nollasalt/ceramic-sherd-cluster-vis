"""簇分析表、特征差异图和模式洞察回调。"""
import dash
from dash import ALL, Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.express as px

from app_core.data_cache import get_data_cache
from performance_utils import cache_plot_result


def build_cluster_pattern_insights(dff, cluster_col, feature_cols, selected_cluster, sil_means):
    """基于当前筛选数据生成可读的簇模式洞察文本。"""
    if cluster_col not in dff.columns or len(dff) == 0:
        return html.Div('暂无可分析的模式', style={'color': '#666'})

    clusters = sorted(dff[cluster_col].dropna().unique())
    if len(clusters) == 0:
        return html.Div('暂无可分析的模式', style={'color': '#666'})

    insights = []
    size_series = dff[cluster_col].value_counts()
    if len(size_series) > 0:
        largest_id = size_series.index[0]
        largest_n = int(size_series.iloc[0])
        smallest_id = size_series.index[-1]
        smallest_n = int(size_series.iloc[-1])
        insights.append(
            f"规模结构：最大簇 {largest_id}（{largest_n}）与最小簇 {smallest_id}（{smallest_n}）差异明显，可优先检查大簇内部是否还含子模式。"
        )

    cat_fields = [c for c in ['part_C', 'type_C', 'unit_C'] if c in dff.columns and dff[c].notna().any()]
    best_field = None
    best_avg_purity = -1.0
    best_field_purity = {}
    for field in cat_fields:
        per_cluster = {}
        for cid, grp in dff[[cluster_col, field]].dropna().groupby(cluster_col):
            vc = grp[field].value_counts(normalize=True)
            if len(vc) > 0:
                per_cluster[cid] = (float(vc.iloc[0]), str(vc.index[0]))
        if per_cluster:
            avg_purity = float(np.mean([v[0] for v in per_cluster.values()]))
            if avg_purity > best_avg_purity:
                best_avg_purity = avg_purity
                best_field = field
                best_field_purity = per_cluster

    if best_field and best_field_purity:
        high_purity = [
            (cid, ratio, label)
            for cid, (ratio, label) in best_field_purity.items()
            if ratio >= 0.70
        ]
        high_purity.sort(key=lambda x: x[1], reverse=True)
        if high_purity:
            top_text = '；'.join([f"簇 {cid}: {label} ({ratio:.1%})" for cid, ratio, label in high_purity[:3]])
            insights.append(f"类别模式：按 {best_field} 统计，存在高纯度簇（≥70%），如 {top_text}。")
        else:
            insights.append(f"类别模式：按 {best_field} 统计，各簇纯度整体偏低，可能呈连续过渡而非离散分组。")

    if feature_cols and len(clusters) >= 2:
        work = dff.dropna(subset=feature_cols)
        if len(work) >= 2:
            centers_df = work.groupby(cluster_col)[feature_cols].mean()
            if len(centers_df) >= 2:
                centers = centers_df.values
                center_ids = centers_df.index.to_list()
                diff = centers[:, None, :] - centers[None, :, :]
                dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dist_mat, np.inf)
                min_pos = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
                c1 = center_ids[min_pos[0]]
                c2 = center_ids[min_pos[1]]
                dmin = float(dist_mat[min_pos])
                insights.append(f"邻近关系：簇 {c1} 与簇 {c2} 的中心最接近（距离 {dmin:.3f}），建议重点比较这两个簇的边界样本。")

    valid_sil = [(cid, val) for cid, val in sil_means.items() if not pd.isna(val)]
    if valid_sil:
        valid_sil.sort(key=lambda x: x[1], reverse=True)
        best_c, best_s = valid_sil[0]
        worst_c, worst_s = valid_sil[-1]
        insights.append(f"分离质量：轮廓系数最好的是簇 {best_c}（{best_s:.3f}），最弱的是簇 {worst_c}（{worst_s:.3f}）。")

    selected_detail = None
    if selected_cluster is not None and feature_cols:
        try:
            cluster_center = dff[dff[cluster_col] == selected_cluster][feature_cols].mean().values
            global_center = dff[feature_cols].mean().values
            diff = cluster_center - global_center
            abs_diff = np.abs(diff)
            idx = np.argsort(abs_diff)[-3:][::-1]
            if len(idx) > 0:
                top_desc = '、'.join([
                    f"{feature_cols[i]}({'高' if diff[i] >= 0 else '低'})"
                    for i in idx
                ])
                selected_detail = html.Div(
                    f"当前选中簇 {selected_cluster} 的主要区分特征：{top_desc}。",
                    style={'marginTop': '8px', 'color': '#444'}
                )
        except Exception:
            selected_detail = None

    if not insights:
        return html.Div('当前筛选下样本不足，暂时无法提取稳定模式。', style={'color': '#666'})

    # 每条洞察对应图标与颜色
    _ICONS = ['📊', '🏷️', '🔗', '⭐']
    _COLORS = ['#1a6fad', '#2a7a4a', '#7d3c98', '#b7770d']

    insight_cards = []
    for idx_i, text in enumerate(insights):
        ic = _ICONS[idx_i % len(_ICONS)]
        col = _COLORS[idx_i % len(_COLORS)]
        insight_cards.append(html.Div([
            html.Span(ic, style={'fontSize': '16px', 'marginRight': '8px', 'flexShrink': '0'}),
            html.Span(text, style={'fontSize': '12px', 'color': '#333', 'lineHeight': '1.6'}),
        ], style={
            'display': 'flex', 'alignItems': 'flex-start',
            'padding': '9px 12px',
            'borderRadius': '8px',
            'backgroundColor': col + '0d',
            'border': f'1px solid {col}33',
            'marginBottom': '7px',
        }))

    content = [
        html.Div('自动模式洞察', style={
            'fontWeight': '700', 'fontSize': '13px', 'color': '#2c3e50', 'marginBottom': '8px',
        }),
        html.Div(insight_cards),
    ]
    if selected_detail is not None:
        content.append(selected_detail)
    return html.Div(content)


def register_cluster_analysis_callbacks(app):
    @app.callback(
        [Output('cluster-quality-table', 'children'),
         Output('feature-diff-graph', 'figure'),
         Output('cluster-pattern-insights', 'children'),
         Output('analysis-cluster-selector', 'options'),
         Output('analysis-cluster-selector', 'value'),
         Output('analysis-table-page-index', 'data'),
         Output('analysis-table-page-status', 'children'),
         Output('analysis-table-prev', 'disabled'),
         Output('analysis-table-next', 'disabled')],
        [Input('visualization-tabs', 'value'),
         Input('analysis-cluster-selector', 'value'),
         Input('feature-diff-mode', 'value'),
         Input('feature-topk-slider', 'value'),
         Input('analysis-table-prev', 'n_clicks'),
         Input('analysis-table-next', 'n_clicks'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('analysis-table-page-index', 'data'),
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_cluster_analysis(tab_value, selected_cluster, diff_mode, topk, prev_clicks, next_clicks, selected_clusters, selected_units, selected_parts, selected_types, page_index, data_store):
        """渲染簇分析表、特征差异图和自动模式洞察。"""
        if tab_value != 'cluster-analysis':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        page_index = int(page_index or 1)
        page_index = max(1, page_index)

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
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

        if cluster_col not in dff.columns or len(dff) == 0:
            empty_fig = px.bar(title='暂无数据')
            return html.Div('暂无数据'), empty_fig, html.Div('暂无可分析模式'), [], None, 1, '第 0/0 页', True, True

        clusters = sorted(dff[cluster_col].dropna().unique())
        options = [{'label': str(c), 'value': c} for c in clusters]
        if selected_cluster not in clusters:
            selected_cluster = clusters[0] if clusters else None

        purity_field = None
        for cand in ['part_C', 'type_C', 'unit_C']:
            if cand in dff.columns and dff[cand].notna().any():
                purity_field = cand
                break

        size_series = dff[cluster_col].value_counts().sort_index()

        purity_data = {}
        if purity_field:
            grp = dff[[cluster_col, purity_field]].dropna().groupby(cluster_col)[purity_field]
            for cid, series in grp:
                vc = series.value_counts(normalize=True)
                purity = float(vc.iloc[0]) if len(vc) > 0 else np.nan
                top_label = str(vc.index[0]) if len(vc) > 0 else ''
                purity_data[cid] = (purity, top_label)
        else:
            purity_data = {cid: (np.nan, '') for cid in clusters}

        sil_means = {cid: np.nan for cid in clusters}
        if feature_cols and len(feature_cols) > 1 and len(dff) >= 3 and len(clusters) >= 2:
            try:
                work = dff.dropna(subset=feature_cols)
                X = work[feature_cols].values
                labels = work[cluster_col].values
                if len(np.unique(labels)) >= 2 and len(X) >= 3:
                    from sklearn.metrics import silhouette_samples
                    max_samples = 4000
                    if len(X) > max_samples:
                        idx = np.random.default_rng(42).choice(len(X), size=max_samples, replace=False)
                        X = X[idx]
                        labels = labels[idx]
                    sil_samples = silhouette_samples(X, labels, metric='euclidean')
                    for cid in np.unique(labels):
                        mask = labels == cid
                        if np.any(mask):
                            sil_means[cid] = float(np.mean(sil_samples[mask]))
            except Exception:
                pass

        rows = []
        for cid in clusters:
            size = int(size_series.get(cid, 0))
            purity, top_lbl = purity_data.get(cid, (np.nan, ''))
            sil = sil_means.get(cid, np.nan)
            rows.append((cid, size, purity, top_lbl, sil))

        rows.sort(key=lambda x: -x[1])

        def fmt(v):
            """格式化显示值，浮点数保留 3 位小数。"""
            if isinstance(v, float):
                return f"{v:.3f}" if not np.isnan(v) else '-'
            return str(v)

        page_size = 12
        total_rows = len(rows)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)

        ctx = dash.callback_context
        trigger_id = None
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'analysis-table-prev':
            page_index = max(1, page_index - 1)
        elif trigger_id == 'analysis-table-next':
            page_index = min(total_pages, page_index + 1)
        else:
            page_index = 1

        page_index = max(1, min(page_index, total_pages))
        start_idx = (page_index - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_rows = rows[start_idx:end_idx]

        table = html.Table([
            html.Thead(html.Tr([
                html.Th('簇'), html.Th('规模'), html.Th('纯度'), html.Th('主类别'), html.Th('簇内轮廓')
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(str(cid)),
                    html.Td(size),
                    html.Td(fmt(purity)),
                    html.Td(top_lbl),
                    html.Td(fmt(sil))
                ]) for cid, size, purity, top_lbl, sil in page_rows
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse'})

        topk = int(topk or 5)
        topk = max(3, min(30, topk))
        feat_fig = px.bar(title='特征差异')
        if feature_cols and selected_cluster is not None:
            try:
                cluster_center = dff[dff[cluster_col] == selected_cluster][feature_cols].mean().values
                global_center = dff[feature_cols].mean().values
                if diff_mode == 'zscore':
                    global_std = dff[feature_cols].std(ddof=0).replace(0, np.nan).values
                    diff = (cluster_center - global_center) / (global_std + 1e-8)
                    title_mode = 'z-score'
                else:
                    diff = cluster_center - global_center
                    title_mode = '均值差'
                abs_diff = np.abs(diff)
                idx = np.argsort(abs_diff)[-topk:][::-1]
                data = {
                    'feature': [feature_cols[i] for i in idx],
                    'delta': [float(diff[i]) for i in idx]
                }
                feat_fig = px.bar(data, x='feature', y='delta', title=f"簇 {selected_cluster} 特征差异 Top-{topk}（{title_mode}）")
                feat_fig.update_layout(margin=dict(l=40, r=30, t=60, b=120))
                feat_fig.update_traces(marker_color='#3366cc')
            except Exception:
                feat_fig = px.bar(title='特征差异计算失败')

        pattern_insights = build_cluster_pattern_insights(
            dff=dff,
            cluster_col=cluster_col,
            feature_cols=feature_cols,
            selected_cluster=selected_cluster,
            sil_means=sil_means,
        )

        page_status = f"第 {page_index}/{total_pages} 页｜簇 {start_idx + 1}-{start_idx + len(page_rows)} / {total_rows}"
        prev_disabled = page_index <= 1
        next_disabled = page_index >= total_pages

        return table, feat_fig, pattern_insights, options, selected_cluster, page_index, page_status, prev_disabled, next_disabled
