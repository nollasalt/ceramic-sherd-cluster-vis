"""聚类质量指标（轮廓系数、CH 指数、DB 指数）及紧凑度风险图回调。"""
import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.express as px

from app_core.data_cache import get_data_cache
from performance_utils import cache_plot_result


def register_cluster_quality_callbacks(app):
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
        """计算并渲染聚类质量指标、风险条图和明细表。"""
        if tab_value != 'cluster-quality':
            return dash.no_update, dash.no_update, dash.no_update

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

        # 高维特征空间中距离度量失效（维度诅咒），先降到 50 维再计算指标
        if X.shape[1] > 50:
            from sklearn.decomposition import PCA
            n_comp = min(50, X.shape[0] - 1, X.shape[1])
            X = PCA(n_components=n_comp, random_state=42).fit_transform(X)

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
            """安全执行 sklearn 指标函数，异常时返回默认值。"""
            try:
                return float(fn(X, labels))
            except Exception:
                return default

        sil = safe_metric(silhouette_score)
        ch = safe_metric(calinski_harabasz_score)
        db = safe_metric(davies_bouldin_score)

        def card(title, value, hint):
            """构建单个质量指标卡片。"""
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
            """根据松散度与轮廓系数返回风险颜色。"""
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
            """格式化数值，缺失值显示为 '-'。"""
            return '-' if pd.isna(v) else f"{v:.{digits}f}"

        _TH = {
            'padding': '8px 12px', 'backgroundColor': '#2c3e50', 'color': '#fff',
            'fontSize': '12px', 'fontWeight': '600', 'textAlign': 'left',
            'whiteSpace': 'nowrap', 'letterSpacing': '0.03em',
        }
        _TD = {
            'padding': '7px 12px', 'fontSize': '12px', 'color': '#333',
            'borderBottom': '1px solid #f0f0f0', 'verticalAlign': 'middle',
        }

        def _sil_badge(v):
            if pd.isna(v): return html.Span('-', style={'color': '#aaa'})
            color = '#27ae60' if v >= 0.3 else ('#e67e22' if v >= 0 else '#e74c3c')
            return html.Span(f"{v:.3f}", style={
                'color': color, 'fontWeight': '700',
                'backgroundColor': color + '18',
                'padding': '2px 7px', 'borderRadius': '10px',
                'border': f'1px solid {color}44',
            })

        table_rows = []
        for i, (_, row) in enumerate(detail_df.sort_values('looseness', ascending=False).iterrows()):
            color = row['status_color'] if pd.notna(row['status_color']) else '#cccccc'
            row_bg = '#fafbfc' if i % 2 == 0 else '#fff'
            dot = html.Span('●', style={'color': color, 'marginRight': '6px', 'fontSize': '14px'})
            table_rows.append(html.Tr([
                html.Td([dot, html.Span(str(row['cluster']))], style={**_TD, 'fontWeight': '600'}),
                html.Td(str(int(row['size'])), style=_TD),
                html.Td(fmt_val(row['intra_mean']), style=_TD),
                html.Td(fmt_val(row['inter_min']), style=_TD),
                html.Td(_sil_badge(row['silhouette']), style=_TD),
                html.Td(fmt_val(row['looseness']), style=_TD),
            ], style={'backgroundColor': row_bg}))

        detail_table = html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th(t, style=_TH)
                    for t in ['簇', '规模', '簇内均距', '最近簇距', '轮廓系数', '松散度比']
                ])),
                html.Tbody(table_rows),
            ], style={'borderCollapse': 'collapse', 'width': '100%'}),
        ], style={
            'overflowX': 'auto', 'border': '1px solid #e4e8ef',
            'borderRadius': '10px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.05)',
            'marginTop': '10px', 'overflow': 'hidden',
        })

        detail_hint = html.Div([
            html.Span('颜色说明：', style={'fontWeight': '600', 'color': '#444'}),
            html.Span('● ', style={'color': '#4caf50'}), html.Span('清晰  ', style={'color': '#555'}),
            html.Span('● ', style={'color': '#ffb300'}), html.Span('需关注  ', style={'color': '#555'}),
            html.Span('● ', style={'color': '#e53935'}), html.Span('混杂  ', style={'color': '#555'}),
            html.Span('｜ 松散度 = 簇内均距 ÷ 最近簇距，越低越清晰', style={'color': '#888', 'fontSize': '11px'}),
        ], style={'marginTop': '8px', 'fontSize': '12px', 'padding': '0 2px'})

        return [summary] + cards, bar_fig, html.Div([detail_hint, detail_table])
