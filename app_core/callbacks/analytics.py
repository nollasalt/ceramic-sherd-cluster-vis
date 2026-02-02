"""Analytical callbacks split from the main app module."""

from pathlib import Path

import dash
import dash
from dash import Input, Output, State, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px

from app_core.data_cache import get_data_cache
from app_core.utils import CLUSTER_COLORS
from data_processing import create_cluster_pattern_heatmap, img_to_base64, img_to_base64_full
from performance_utils import cache_plot_result, image_cache

try:
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def register_analytics_callbacks(app, *, image_root):
    """Register cluster analytics, heatmap, similarity, and modal callbacks."""

    @app.callback(
        Output('cluster-size-graph', 'figure'),
        [Input('visualization-tabs', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_cluster_size(tab_value, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'cluster-size':
            return dash.no_update

        # Pull dataset from server-side cache to avoid large client payloads
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

        if len(dff) == 0 or cluster_col not in dff.columns:
            empty_fig = px.bar(title='暂无数据')
            empty_fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))
            return empty_fig

        counts = dff[cluster_col].value_counts().sort_index()
        plot_df = counts.reset_index()
        plot_df.columns = ['cluster', 'count']
        plot_df['cluster_label'] = plot_df['cluster'].astype(str)

        def to_int_or_index(lbl, fallback_idx):
            try:
                return int(float(lbl))
            except Exception:
                return fallback_idx

        color_map = {}
        for i, lbl in enumerate(plot_df['cluster_label']):
            color_idx = to_int_or_index(lbl, i) % len(CLUSTER_COLORS)
            color_map[lbl] = CLUSTER_COLORS[color_idx]

        total = int(counts.sum())
        max_count = int(counts.max()) if len(counts) > 0 else 0
        max_ratio = max_count / total if total > 0 else 0
        sorted_counts = counts.sort_values()
        half = max(1, len(sorted_counts) // 2)
        tail_share = sorted_counts.head(half).sum() / total if total > 0 else 0

        fig = px.bar(
            plot_df,
            x='cluster_label',
            y='count',
            text='count',
            color='cluster_label',
            color_discrete_map=color_map
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            title=f"簇规模分布｜样本 {len(dff)}，簇 {len(counts)}｜最大簇占比 {max_ratio:.2%}｜长尾占比 {tail_share:.2%}",
            xaxis_title='簇 ID',
            yaxis_title='样本数',
            bargap=0.3,
            showlegend=False,
            margin=dict(l=40, r=30, t=60, b=80)
        )
        return fig

    @app.callback(
        Output('cluster-quality-cards', 'children'),
        Output('cluster-quality-bars', 'figure'),
        Output('cluster-quality-detail', 'children'),
        [Input('visualization-tabs', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_cluster_quality(tab_value, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'cluster-quality':
            return dash.no_update, dash.no_update, dash.no_update

        # Use cached df/feature_cols for metric computation
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

        if not feature_cols or cluster_col not in dff.columns or len(dff) < 3:
            empty = html.Div('暂无足够数据计算指标', style={'color': '#666', 'padding': '8px'})
            return empty, dash.no_update, dash.no_update

        dff = dff.dropna(subset=feature_cols)
        if len(dff) < 3:
            empty = html.Div('样本过少，无法计算指标', style={'color': '#666', 'padding': '8px'})
            return empty, dash.no_update, dash.no_update

        X = dff[feature_cols].values
        labels = dff[cluster_col].values

        if len(np.unique(labels)) < 2:
            empty = html.Div('簇数不足 2，无法计算指标', style={'color': '#666', 'padding': '8px'})
            return empty, dash.no_update, dash.no_update

        max_samples = 3000
        if len(X) > max_samples:
            sample_idx = np.random.default_rng(42).choice(len(X), size=max_samples, replace=False)
            X = X[sample_idx]
            labels = labels[sample_idx]

        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        def safe_metric(fn, default=np.nan):
            try:
                return float(fn(X, labels))
            except Exception:
                return default

        sil = safe_metric(silhouette_score)
        ch = safe_metric(calinski_harabasz_score)
        db = safe_metric(davies_bouldin_score)

        def card(title, value, hint):
            txt = '无法计算' if np.isnan(value) else f"{value:.4f}"
            return html.Div([
                html.Div(title, style={'fontSize': '13px', 'color': '#666', 'marginBottom': '6px'}),
                html.Div(txt, style={'fontSize': '22px', 'fontWeight': '600'}),
                html.Div(hint, style={'fontSize': '12px', 'color': '#888', 'marginTop': '4px'})
            ], style={
                'padding': '12px 14px',
                'border': '1px solid #e0e0e0',
                'borderRadius': '8px',
                'minWidth': '180px',
                'backgroundColor': '#fafafa'
            })

        cards = [
            card('Silhouette', sil, '越接近 1 越好'),
            card('Calinski-Harabasz', ch, '越大越好'),
            card('Davies-Bouldin', db, '越低越好')
        ]

        summary = html.Div(
            f"样本 {len(X)}｜簇 {len(np.unique(labels))}",
            style={'fontSize': '13px', 'color': '#555', 'marginBottom': '8px'}
        )

        from sklearn.metrics import silhouette_samples

        sil_per_cluster = {}
        try:
            max_samples_detail = 4000
            X_detail = X
            labels_detail = labels
            if len(X_detail) > max_samples_detail:
                idx = np.random.default_rng(42).choice(len(X_detail), size=max_samples_detail, replace=False)
                X_detail = X_detail[idx]
                labels_detail = labels_detail[idx]
            sil_samples = silhouette_samples(X_detail, labels_detail, metric='euclidean')
            for cid in np.unique(labels_detail):
                mask = labels_detail == cid
                if np.any(mask):
                    sil_per_cluster[cid] = float(np.mean(sil_samples[mask]))
        except Exception:
            sil_per_cluster = {}

        centers_df = dff.groupby(cluster_col)[feature_cols].mean()
        centers = centers_df.values
        center_ids = centers_df.index.to_numpy()
        inter_min = {}
        if len(center_ids) > 1:
            diff = centers[:, None, :] - centers[None, :, :]
            dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
            for i, cid in enumerate(center_ids):
                mask = np.ones(len(center_ids), dtype=bool)
                mask[i] = False
                inter_min[cid] = float(np.min(dist_mat[i][mask])) if np.any(mask) else np.nan

        intra_mean = {}
        for cid, group in dff.groupby(cluster_col):
            if len(group) == 0:
                intra_mean[cid] = np.nan
                continue
            center_vec = group[feature_cols].mean().values
            distances = np.linalg.norm(group[feature_cols].values - center_vec, axis=1)
            intra_mean[cid] = float(np.mean(distances))

        records = []
        for cid in sorted(dff[cluster_col].unique()):
            records.append({
                'cluster': cid,
                'size': int((dff[cluster_col] == cid).sum()),
                'silhouette': sil_per_cluster.get(cid, np.nan),
                'intra_mean': intra_mean.get(cid, np.nan),
                'inter_min': inter_min.get(cid, np.nan)
            })

        detail_df = pd.DataFrame(records)
        detail_df['cluster_label'] = detail_df['cluster'].astype(str)
        detail_df['looseness'] = detail_df['intra_mean'] / (detail_df['inter_min'] + 1e-8)

        def status_color(looseness, sil):
            if pd.isna(looseness):
                return '#cccccc'
            if looseness < 0.3 and (pd.isna(sil) or sil >= 0.2):
                return '#4caf50'
            if looseness < 0.6 or (not pd.isna(sil) and sil >= 0.0):
                return '#ffb300'
            return '#e53935'

        detail_df['status_color'] = detail_df.apply(lambda r: status_color(r['looseness'], r['silhouette']), axis=1)

        plot_df = detail_df.sort_values('looseness', ascending=False)
        bar_fig = px.bar(
            plot_df,
            x='cluster_label',
            y='looseness',
            text='looseness',
            color='status_color',
            color_discrete_map='identity',
            labels={'cluster_label': '簇', 'looseness': '松散度（越低越紧凑）'},
            title='簇紧凑度/黏连风险（颜色：绿=清晰，黄=需关注，红=混杂）'
        )
        bar_fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        bar_fig.update_layout(margin=dict(l=40, r=30, t=60, b=80), showlegend=False)

        def fmt_val(v, digits=3):
            return '-' if pd.isna(v) else f"{v:.{digits}f}"

        table_rows = []
        header = html.Tr([
            html.Th('簇'), html.Th('规模'), html.Th('簇内均距'), html.Th('最近簇距'), html.Th('轮廓系数'), html.Th('松散度比')
        ])
        for _, row in detail_df.sort_values('looseness', ascending=False).iterrows():
            color = row['status_color'] if pd.notna(row['status_color']) else '#cccccc'
            table_rows.append(html.Tr([
                html.Td(str(row['cluster'])),
                html.Td(str(int(row['size']))),
                html.Td(fmt_val(row['intra_mean'])),
                html.Td(fmt_val(row['inter_min'])),
                html.Td(fmt_val(row['silhouette'])),
                html.Td(fmt_val(row['looseness']))
            ], style={'backgroundColor': '#fdfdfd', 'borderLeft': f'6px solid {color}'}))

        detail_table = html.Table([
            html.Thead(header),
            html.Tbody(table_rows)
        ], style={'borderCollapse': 'collapse', 'width': '100%', 'marginTop': '6px'})

        detail_hint = html.Div(
            '颜色含义：绿=簇紧凑且与邻簇分开；黄=轻微分散或稍粘连；红=分散或与邻簇混杂。松散度比 = 簇内平均距离 / 最近簇中心距离，越低越清晰。',
            style={'color': '#666', 'marginTop': '4px'}
        )

        return [summary] + cards, bar_fig, html.Div([detail_hint, detail_table])

    @app.callback(
        Output('category-breakdown-graph', 'figure'),
        [Input('visualization-tabs', 'value'),
         Input('category-field-selector', 'value'),
         Input('category-x-axis', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_category_breakdown(tab_value, category_field, x_axis_mode, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'category-breakdown':
            return dash.no_update

        # Category breakdown also reads from cached df
        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']

        if category_field not in df.columns:
            fig = px.bar(title='所选类别字段不存在')
            fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))
            return fig

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        dff = dff[pd.notna(dff[category_field])]

        if len(dff) == 0 or cluster_col not in dff.columns:
            empty_fig = px.bar(title='暂无数据')
            empty_fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))
            return empty_fig

        x_axis_mode = x_axis_mode or 'cluster'
        x_field = cluster_col if x_axis_mode == 'cluster' else 'unit_C'

        if x_field not in dff.columns:
            fig = px.bar(title='所选横轴字段不存在')
            fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))
            return fig

        grouped = (
            dff
            .groupby([x_field, category_field])
            .size()
            .reset_index(name='count')
        )

        grouped['x_label'] = grouped[x_field].astype(str)
        grouped = grouped.sort_values([x_field, category_field])

        fig = px.bar(
            grouped,
            x='x_label',
            y='count',
            color=category_field,
            text='count',
            barmode='stack'
        )
        fig.update_traces(textposition='outside', cliponaxis=False)
        fig.update_layout(
            title=f"类别构成（{category_field}）｜样本 {len(dff)}",
            xaxis_title='簇' if x_axis_mode == 'cluster' else '单位 (unit_C)',
            yaxis_title='样本数',
            bargap=0.25,
            margin=dict(l=40, r=30, t=60, b=80),
            legend_title=category_field
        )
        return fig

    @app.callback(
        [Output('cluster-quality-table', 'children'),
         Output('feature-diff-graph', 'figure'),
         Output('analysis-cluster-selector', 'options'),
         Output('analysis-cluster-selector', 'value')],
        [Input('visualization-tabs', 'value'),
         Input('analysis-cluster-selector', 'value'),
         Input('feature-diff-mode', 'value'),
         Input('feature-topk-slider', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_cluster_analysis(tab_value, selected_cluster, diff_mode, topk, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'cluster-analysis':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Use server-side cache to compute purity/feature diffs
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
            return html.Div('暂无数据'), empty_fig, [], None

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
            if isinstance(v, float):
                return f"{v:.3f}" if not np.isnan(v) else '-'
            return str(v)

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
                ]) for cid, size, purity, top_lbl, sil in rows
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

        return table, feat_fig, options, selected_cluster

    @app.callback(
        Output('representative-grid', 'children'),
        Output('outlier-list', 'children'),
        [Input('visualization-tabs', 'value'),
         Input('rep-samples-per-cluster', 'value'),
         Input('rep-strategy', 'value'),
         Input('outlier-count', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    def render_representatives(tab_value, samples_per_cluster, strategy, outlier_count, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'representatives':
            return dash.no_update, dash.no_update

        # Thumbnails and outliers are derived from cached df to keep responses small
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

        if cluster_col not in dff.columns or len(dff) == 0:
            empty_div = html.Div('暂无数据', style={'color': '#666', 'padding': '8px'})
            return empty_div, empty_div

        clusters = sorted(dff[cluster_col].dropna().unique())
        if len(clusters) == 0:
            empty_div = html.Div('暂无数据', style={'color': '#666', 'padding': '8px'})
            return empty_div, empty_div

        n_per = int(samples_per_cluster or 1)
        n_per = max(1, min(12, n_per))
        outlier_k = int(outlier_count or 1)
        outlier_k = max(1, min(5, outlier_k))

        max_total = 200
        if len(clusters) * n_per > max_total:
            n_per = max(1, max_total // len(clusters))

        base_root = Path(__file__).parent.parent.parent
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

        cards = []
        outlier_blocks = []
        thumb_size = 120
        for c in clusters:
            subset_all = dff[dff[cluster_col] == c]
            subset_feat = subset_all.dropna(subset=feature_cols) if feature_cols else subset_all

            chosen = subset_all
            if strategy == 'center' and feature_cols and len(subset_feat) > 0:
                center_vec = subset_feat[feature_cols].mean().values
                distances = np.linalg.norm(subset_feat[feature_cols].values - center_vec, axis=1)
                subset_feat = subset_feat.assign(_dist=distances)
                chosen = subset_feat.nsmallest(n_per, '_dist')
            elif strategy == 'random':
                chosen = subset_all.sample(n=min(n_per, len(subset_all)), random_state=42) if len(subset_all) > 0 else subset_all
            else:
                chosen = subset_all.head(n_per)

            if len(chosen) < n_per and len(subset_all) > len(chosen):
                extra = subset_all.drop(chosen.index, errors='ignore').head(n_per - len(chosen))
                chosen = pd.concat([chosen, extra])

            thumbs = []
            for _, row in chosen.head(n_per).iterrows():
                img_val = row.get('image_name') if 'image_name' in row else row.get(image_col)
                path = resolve_path(img_val)
                cache_key = f"rep_thumb_{Path(path).name}_{thumb_size}"
                b64 = image_cache.get(cache_key) if image_cache else None
                if b64 is None:
                    b64 = img_to_base64(path, max_size=thumb_size)
                    if image_cache and b64:
                        image_cache.set(cache_key, b64)
                if b64:
                    thumbs.append(html.Img(
                        src=b64,
                        style={'height': f'{thumb_size}px', 'border': '1px solid #ddd', 'borderRadius': '4px', 'backgroundColor': '#fafafa'},
                        **{'data-image-path': Path(path).name},
                        title=str(img_val)
                    ))
                else:
                    thumbs.append(html.Div(str(Path(path).name), style={'fontSize': '12px', 'color': '#999'}))

            if len(thumbs) == 0:
                thumbs.append(html.Div('无可用图片', style={'fontSize': '12px', 'color': '#999'}))

            cards.append(html.Div([
                html.Div(f"簇 {c}", style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '6px'}),
                html.Div(thumbs, style={'display': 'flex', 'gap': '6px', 'flexWrap': 'wrap'})
            ], style={
                'padding': '10px',
                'border': '1px solid #e0e0e0',
                'borderRadius': '8px',
                'minWidth': '180px',
                'backgroundColor': '#fff'
            }))

            if feature_cols and len(subset_feat) > 0:
                center_vec = subset_feat[feature_cols].mean().values
                distances = np.linalg.norm(subset_feat[feature_cols].values - center_vec, axis=1)
                subset_feat = subset_feat.assign(_dist=distances)
                outliers = subset_feat.nlargest(outlier_k, '_dist')
                items = []
                for _, r in outliers.iterrows():
                    img_val = r.get('image_name') if 'image_name' in r else r.get(image_col)
                    path = resolve_path(img_val)
                    cache_key = f"outlier_thumb_{Path(path).name}_{thumb_size}"
                    b64 = image_cache.get(cache_key) if image_cache else None
                    if b64 is None:
                        b64 = img_to_base64(path, max_size=thumb_size)
                        if image_cache and b64:
                            image_cache.set(cache_key, b64)
                    label_text = f"样本 {r.get('sample_id', img_val)}｜距离 {r['_dist']:.3f}"
                    thumb = html.Img(src=b64, style={'height': '60px', 'border': '1px solid #ddd', 'borderRadius': '4px', 'marginRight': '6px'}) if b64 else None
                    items.append(html.Li([
                        thumb if thumb else html.Span(str(Path(path).name), style={'marginRight': '6px'}),
                        html.Span(label_text)
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '4px'}))
                outlier_blocks.append(html.Div([
                    html.Div(f"簇 {c} 离群样本", style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '4px'}),
                    html.Ul(items, style={'paddingLeft': '16px', 'marginTop': '0', 'marginBottom': '8px'})
                ], style={'marginBottom': '8px'}))

        if len(outlier_blocks) == 0:
            outlier_blocks = html.Div('缺少特征列，无法计算离群样本', style={'color': '#666', 'padding': '4px'})

        return cards, outlier_blocks

    @app.callback(
        Output('heatmap-container', 'children'),
        Input('visualization-tabs', 'value'),
        State('cluster-metadata-store', 'data')
    )
    def update_heatmap(tab_value, cluster_metadata):
        if tab_value != 'heatmap' or cluster_metadata is None:
            return html.Div('请选择"聚类特征热力图"选项卡')

        try:
            cluster_centers = np.array(cluster_metadata.get('cluster_centers', []))
            if cluster_centers.shape[0] == 0:
                return html.Div('未找到聚类中心数据')

            if cluster_centers.shape[1] > 50:
                cluster_centers = cluster_centers[:, :50]

            fig = create_cluster_pattern_heatmap(cluster_centers)
            return dcc.Graph(figure=fig)
        except Exception as exc:
            return html.Div(f'生成热力图时出错: {exc}')

    @app.callback(
        Output('similarity-graph', 'figure'),
        Output('nearest-cluster-list', 'children'),
        [Input('visualization-tabs', 'value'),
         Input('similarity-metric', 'value'),
         Input('similarity-options', 'value'),
         Input('similarity-neighbor-k', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def update_similarity_matrix(tab_value, metric, options, neighbor_k, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'similarity':
            return dash.no_update, dash.no_update

        metric = metric or 'cosine'
        options = options or []
        annotate = 'annotate' in options
        reorder_requested = 'reorder' in options
        neighbor_k = int(neighbor_k or 3)

        # Compute cluster-centroid similarity using cached df and feature columns
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

        if cluster_col not in dff.columns or not feature_cols:
            fig = px.imshow([[0]], title='缺少簇列或特征列')
            return fig, ""

        dff = dff.dropna(subset=feature_cols)
        if len(dff) == 0:
            fig = px.imshow([[0]], title='暂无数据')
            return fig, ""

        centers_df = dff.groupby(cluster_col)[feature_cols].mean()
        clusters = centers_df.index.to_numpy()
        centers = centers_df.values

        if centers.shape[0] == 0:
            fig = px.imshow([[0]], title='暂无簇')
            return fig, ""

        if metric == 'euclidean':
            diff = centers[:, None, :] - centers[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            mat = dist
            neighbor_matrix = dist
            neighbor_is_distance = True
            title = f"簇中心距离矩阵｜簇 {len(clusters)}"
            color_scale = 'Viridis'
            zmin = None
            zmax = None
        else:
            norm = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
            normed = centers / norm
            sim = normed @ normed.T
            mat = sim
            neighbor_matrix = sim
            neighbor_is_distance = False
            title = f"簇中心相似度矩阵｜簇 {len(clusters)}"
            color_scale = 'RdBu'
            zmin = -1
            zmax = 1

        labels = np.array([str(c) for c in clusters])

        reordered = False
        if reorder_requested and SCIPY_AVAILABLE and len(labels) > 1:
            try:
                if neighbor_is_distance:
                    dist_mat = neighbor_matrix
                else:
                    sim01 = (neighbor_matrix + 1) / 2
                    dist_mat = 1 - sim01
                condensed = squareform(dist_mat, checks=False)
                order = leaves_list(linkage(condensed, method='average'))
                mat = mat[np.ix_(order, order)]
                neighbor_matrix = neighbor_matrix[np.ix_(order, order)]
                labels = labels[order]
                reordered = True
            except Exception:
                reordered = False

        fig = px.imshow(
            mat,
            x=labels,
            y=labels,
            color_continuous_scale=color_scale,
            zmin=zmin,
            zmax=zmax,
            labels={'x': '簇', 'y': '簇', 'color': '值'}
        )
        title_suffix = '（已重排）' if reordered else ''
        fig.update_layout(
            title=f"{title}{title_suffix}",
            margin=dict(l=40, r=30, t=60, b=60)
        )
        fig.update_xaxes(side='top')
        fig.update_yaxes(autorange='reversed')

        if annotate:
            text = np.round(mat, 3)
            fig.update_traces(text=text, texttemplate="%{text}")

        if neighbor_matrix.shape[0] > 1:
            k = max(1, min(neighbor_k, neighbor_matrix.shape[0] - 1))
            nearest_children = []
            for i, cid in enumerate(labels):
                if neighbor_is_distance:
                    order = np.argsort(neighbor_matrix[i])
                    nearest_idx = [idx for idx in order if idx != i][:k]
                    neighbors = [f"{labels[j]}（距离 {neighbor_matrix[i][j]:.3f}）" for j in nearest_idx]
                else:
                    order = np.argsort(-neighbor_matrix[i])
                    nearest_idx = [idx for idx in order if idx != i][:k]
                    neighbors = [f"{labels[j]}（相似度 {neighbor_matrix[i][j]:.3f}）" for j in nearest_idx]
                nearest_children.append(html.Li(f"簇 {cid}: " + ", ".join(neighbors)))
            nearest_list = html.Ul(nearest_children)
        else:
            nearest_list = ""

        return fig, nearest_list

    @app.callback(
        Output('modal-image', 'src'),
        [Input('image-path-input', 'value')],
        prevent_initial_call=True
    )
    def load_full_image(image_path):
        if not image_path or image_path == '':
            return dash.no_update
        try:
            full_path = Path(image_root) / image_path
            if full_path.exists():
                full_res_image = img_to_base64_full(str(full_path))
                return full_res_image
            return dash.no_update
        except Exception:
            return dash.no_update

    return app