"""
Analytical callbacks split from the main app module.
负责分析页签、质量评估、代表样本与相似度矩阵等回调。
"""
"""
这个文件是仪表盘的“分析/洞察”回调集合，负责在前端点击不同标签页时生成各类分析视图。
核心职责概览（逻辑都注册为 Dash 回调，数据来自服务端缓存）：

簇规模/质量：生成簇规模柱状图、簇质量指标卡片与紧凑度条形图 analytics.py:16-238。
类别分布：按簇或 unit_C 汇总指定类别字段的堆叠柱状图 analytics.py:240-320。
簇分析表：计算簇纯度、簇内轮廓系数，展示特征差异 Top-K 柱状图 analytics.py:322-456。
代表样本与离群点：为每个簇挑选代表图（中心/随机/顺序）并列出离群样本缩略图，带图像缓存 analytics.py:458-618。
簇中心相似度/距离矩阵：支持余弦或欧氏、可选层次重排与标注，列出最近邻簇列表 analytics.py:620-760。
热力图与大图：根据聚类中心生成特征热力图、按路径加载大图查看 analytics.py:762-835。
"""
from pathlib import Path

import dash
import dash
from dash import ALL, Input, Output, State, dcc, html
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


def register_analytics_callbacks(app, *, image_root, image_search_dirs=None):
    """注册分析相关回调（规模、质量、簇分析、代表样本、相似度等）。"""

    search_dirs = []
    if image_root:
        search_dirs.append(Path(image_root))
    if image_search_dirs:
        search_dirs.extend(Path(p) for p in image_search_dirs)
    # remove duplicates while keeping order
    seen_dirs = []
    for p in search_dirs:
        if p not in seen_dirs:
            seen_dirs.append(p)
    search_dirs = seen_dirs

    def resolve_full_path(image_path: str) -> Path | None:
        """在配置的图像目录中解析大图文件路径。"""
        if not image_path:
            return None
        target = Path(image_path)
        candidates = [target]
        if not target.is_absolute():
            candidates.append(Path(target.name))

        for base in search_dirs:
            base = Path(base)
            if not base.exists():
                continue
            for cand in candidates:
                cand_path = base / cand
                if cand_path.exists():
                    return cand_path
            try:
                match = next(base.rglob(target.name))
                if match.exists():
                    return match
            except StopIteration:
                pass
        return None

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

        content = [
            html.Div('自动模式洞察', style={'fontWeight': '600', 'marginBottom': '6px'}),
            html.Ul([html.Li(text) for text in insights], style={'margin': '0', 'paddingLeft': '18px', 'color': '#333'})
        ]
        if selected_detail is not None:
            content.append(selected_detail)
        return html.Div(content)

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
        """渲染簇规模分布图，并给出最大簇与长尾占比信息。"""
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
            """将簇标签安全转为整数索引，失败时回退默认索引。"""
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
        """计算并渲染聚类质量指标、风险条图和明细表。"""
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
        """按类别字段渲染构成分布图（按簇或按 unit）。"""
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
         Output('cluster-pattern-insights', 'children'),
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
        """渲染簇分析表、特征差异图和自动模式洞察。"""
        if tab_value != 'cluster-analysis':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

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
            return html.Div('暂无数据'), empty_fig, html.Div('暂无可分析模式'), [], None

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

        pattern_insights = build_cluster_pattern_insights(
            dff=dff,
            cluster_col=cluster_col,
            feature_cols=feature_cols,
            selected_cluster=selected_cluster,
            sil_means=sil_means,
        )

        return table, feat_fig, pattern_insights, options, selected_cluster

    @app.callback(
        [Output('unit-compare-a', 'options'),
         Output('unit-compare-a', 'value'),
         Output('unit-compare-b', 'options'),
         Output('unit-compare-b', 'value'),
         Output('unit-compare-summary', 'children'),
         Output('unit-compare-graph', 'figure')],
        [Input('visualization-tabs', 'value'),
         Input('unit-compare-a', 'value'),
         Input('unit-compare-b', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_unit_layer_analysis(tab_value, unit_a, unit_b, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        """比较两个 Unit 层的簇构成差异并生成解读摘要。"""
        if tab_value != 'cluster-analysis':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

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

        if 'unit_C' not in dff.columns or len(dff) == 0:
            empty_fig = px.bar(title='暂无 Unit 数据')
            return [], None, [], None, html.Div('暂无 Unit 层可比较数据', style={'color': '#666'}), empty_fig

        units = sorted(dff['unit_C'].dropna().unique())
        options = [{'label': str(u), 'value': u} for u in units]

        if not units:
            empty_fig = px.bar(title='暂无 Unit 数据')
            return [], None, [], None, html.Div('暂无 Unit 层可比较数据', style={'color': '#666'}), empty_fig

        if unit_a not in units:
            unit_a = units[0]
        if unit_b not in units:
            unit_b = units[1] if len(units) > 1 else units[0]
        if unit_a == unit_b and len(units) > 1:
            for candidate in units:
                if candidate != unit_a:
                    unit_b = candidate
                    break

        a_df = dff[dff['unit_C'] == unit_a]
        b_df = dff[dff['unit_C'] == unit_b]

        if len(a_df) == 0 or len(b_df) == 0:
            empty_fig = px.bar(title='所选 Unit 样本不足')
            return options, unit_a, options, unit_b, html.Div('所选 Unit 样本不足，无法比较', style={'color': '#666'}), empty_fig

        a_cluster = a_df[cluster_col].value_counts(normalize=True)
        b_cluster = b_df[cluster_col].value_counts(normalize=True)
        all_clusters = sorted(set(a_cluster.index).union(set(b_cluster.index)))
        records = []
        for cid in all_clusters:
            pa = float(a_cluster.get(cid, 0.0))
            pb = float(b_cluster.get(cid, 0.0))
            records.append({
                'cluster': str(cid),
                'delta_pct': (pa - pb) * 100.0,
                'unit': f'{unit_a} - {unit_b}'
            })
        diff_df = pd.DataFrame(records)
        diff_df['abs_delta'] = diff_df['delta_pct'].abs()
        plot_df = diff_df.sort_values('abs_delta', ascending=False).head(12).sort_values('delta_pct', ascending=False)

        fig = px.bar(
            plot_df,
            x='cluster',
            y='delta_pct',
            color='delta_pct',
            color_continuous_scale='RdBu',
            title=f'Unit 层簇构成差异（{unit_a} - {unit_b}，单位：百分点）'
        )
        fig.update_layout(margin=dict(l=40, r=20, t=60, b=70), coloraxis_showscale=False)
        fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')

        def top_label_text(frame, field_name):
            """提取指定类别字段的主导标签及占比。"""
            if field_name not in frame.columns:
                return None
            series = frame[field_name].dropna().astype(str)
            if len(series) == 0:
                return None
            vc = series.value_counts(normalize=True)
            return str(vc.index[0]), float(vc.iloc[0])

        summary_items = [
            html.Li(f"样本规模：{unit_a} 有 {len(a_df)} 片，{unit_b} 有 {len(b_df)} 片。"),
        ]

        a_dom_cluster = a_df[cluster_col].value_counts(normalize=True)
        b_dom_cluster = b_df[cluster_col].value_counts(normalize=True)
        if len(a_dom_cluster) > 0 and len(b_dom_cluster) > 0:
            summary_items.append(
                html.Li(
                    f"主导簇：{unit_a} 以簇 {a_dom_cluster.index[0]} 为主（{a_dom_cluster.iloc[0]:.1%}），"
                    f"{unit_b} 以簇 {b_dom_cluster.index[0]} 为主（{b_dom_cluster.iloc[0]:.1%}）。"
                )
            )

        for field, label in [('part_C', '器型部位'), ('type_C', '器类类型')]:
            a_top = top_label_text(a_df, field)
            b_top = top_label_text(b_df, field)
            if a_top and b_top:
                summary_items.append(
                    html.Li(
                        f"{label}偏好：{unit_a} 更集中在“{a_top[0]}”（{a_top[1]:.1%}），"
                        f"{unit_b} 更集中在“{b_top[0]}”（{b_top[1]:.1%}）。"
                    )
                )

        confidence = '高'
        if min(len(a_df), len(b_df)) < 30:
            confidence = '中'
        if min(len(a_df), len(b_df)) < 12:
            confidence = '低'
        summary_items.append(html.Li(f"结果置信度：{confidence}（受样本量影响）。"))

        summary = html.Div([
            html.Div('地层差异解读', style={'fontWeight': '600', 'marginBottom': '6px'}),
            html.Ul(summary_items, style={'margin': '0', 'paddingLeft': '18px'})
        ])

        return options, unit_a, options, unit_b, summary, fig

    @app.callback(
        Output('representative-grid', 'children'),
        Output('outlier-list', 'children'),
        Output('rep-visible-clusters', 'data'),
        Output('rep-load-status', 'children'),
        Output('rep-load-more-btn', 'disabled'),
        [Input('visualization-tabs', 'value'),
         Input('rep-samples-per-cluster', 'value'),
         Input('rep-strategy', 'value'),
         Input('outlier-count', 'value'),
         Input('rep-load-more-btn', 'n_clicks'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('rep-visible-clusters', 'data'),
        State('data-store', 'data')
    )
    def render_representatives(tab_value, samples_per_cluster, strategy, outlier_count, load_more_clicks, selected_clusters, selected_units, selected_parts, selected_types, visible_clusters, data_store):
        """渲染代表样本与离群样本，并支持分页增量加载簇。"""
        if tab_value != 'representatives':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        page_size = 8
        visible_clusters = int(visible_clusters or page_size)
        visible_clusters = max(page_size, visible_clusters)

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
            return empty_div, empty_div, page_size, '已显示 0/0 个簇', True

        clusters = sorted(dff[cluster_col].dropna().unique())
        if len(clusters) == 0:
            empty_div = html.Div('暂无数据', style={'color': '#666', 'padding': '8px'})
            return empty_div, empty_div, page_size, '已显示 0/0 个簇', True

        ctx = dash.callback_context
        trigger_id = None
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'rep-load-more-btn':
            visible_clusters = min(len(clusters), visible_clusters + page_size)
        else:
            visible_clusters = min(len(clusters), page_size)

        active_clusters = clusters[:visible_clusters]

        n_per = int(samples_per_cluster or 1)
        n_per = max(1, min(12, n_per))
        outlier_k = int(outlier_count or 1)
        outlier_k = max(1, min(5, outlier_k))

        base_root = Path(__file__).parent.parent.parent
        image_root_abs = Path(image_root)
        if not image_root_abs.is_absolute():
            image_root_abs = base_root / image_root_abs

        def resolve_path(val: str):
            """解析图像路径并在常用目录中兜底查找。"""
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
        for c in active_clusters:
            subset_all = dff[dff[cluster_col] == c]
            subset_feat = subset_all.dropna(subset=feature_cols) if feature_cols else subset_all
            try:
                cluster_id_for_url = int(c)
            except Exception:
                cluster_id_for_url = str(c)
            assemble_url = f"http://127.0.0.1:12800/?cluster_id={cluster_id_for_url}"

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

            while len(thumbs) < n_per:
                thumbs.append(
                    html.Div(
                        '样本不足',
                        style={
                            'height': f'{thumb_size}px',
                            'minWidth': '84px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'border': '1px dashed #d0d0d0',
                            'borderRadius': '4px',
                            'backgroundColor': '#f8f8f8',
                            'fontSize': '12px',
                            'color': '#999'
                        }
                    )
                )

            if len(thumbs) == 0:
                thumbs.append(html.Div('无可用图片', style={'fontSize': '12px', 'color': '#999'}))

            cards.append(html.Div([
                html.Div([
                    html.Div(f"簇 {c}", style={'fontSize': '13px', 'fontWeight': '600'}),
                    html.Div([
                        html.Button(
                            '查看',
                            id={'type': 'rep-view-cluster', 'index': str(c)},
                            n_clicks=0,
                            style={
                                'padding': '4px 10px',
                                'fontSize': '12px',
                                'backgroundColor': '#0066cc',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '4px',
                                'cursor': 'pointer'
                            }
                        ),
                        html.A(
                            '尝试拼对',
                            href=assemble_url,
                            target='_blank',
                            style={
                                'display': 'inline-block',
                                'padding': '4px 10px',
                                'fontSize': '12px',
                                'backgroundColor': '#28a745',
                                'color': 'white',
                                'borderRadius': '4px',
                                'textDecoration': 'none',
                                'marginLeft': '6px'
                            }
                        )
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '6px'}),
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

        load_status = f"已显示 {len(active_clusters)}/{len(clusters)} 个簇（每次加载 {page_size} 个）"
        disable_load_more = len(active_clusters) >= len(clusters)

        return cards, outlier_blocks, visible_clusters, load_status, disable_load_more

    @app.callback(
        Output('visualization-tabs', 'value', allow_duplicate=True),
        Output('cluster-filter', 'value', allow_duplicate=True),
        Output('rep-last-view-click', 'data', allow_duplicate=True),
        Input({'type': 'rep-view-cluster', 'index': ALL}, 'n_clicks'),
        State('rep-last-view-click', 'data'),
        prevent_initial_call=True,
    )
    def view_cluster_from_representatives(_n_clicks, last_click):
        """从代表样本页跳转到散点页并自动筛选目标簇。"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update

        trigger_value = ctx.triggered[0].get('value')
        if not isinstance(trigger_value, (int, float)) or trigger_value <= 0:
            return dash.no_update, dash.no_update, dash.no_update

        trigger_id = ctx.triggered_id
        if not isinstance(trigger_id, dict):
            return dash.no_update, dash.no_update, dash.no_update

        cluster_id = trigger_id.get('index')
        if cluster_id is None:
            return dash.no_update, dash.no_update, dash.no_update

        last_click = last_click or {}
        if last_click.get('cluster') == cluster_id and int(last_click.get('count', 0)) == int(trigger_value):
            return dash.no_update, dash.no_update, dash.no_update

        try:
            cluster_value = int(cluster_id)
        except Exception:
            cluster_value = cluster_id

        return 'scatter', [cluster_value], {'cluster': cluster_id, 'count': int(trigger_value)}

    @app.callback(
        Output('heatmap-container', 'children'),
        Input('visualization-tabs', 'value'),
        State('cluster-metadata-store', 'data')
    )
    def update_heatmap(tab_value, cluster_metadata):
        """生成簇中心热力图并返回图形组件。"""
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
        """计算簇中心相似度/距离矩阵，并输出最近邻簇列表。"""
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
        """加载并返回原图的高分辨率 base64 数据。"""
        if not image_path or image_path == '':
            return dash.no_update
        try:
            full_path = resolve_full_path(image_path)
            if full_path and full_path.exists():
                full_res_image = img_to_base64_full(str(full_path))
                return full_res_image
            return dash.no_update
        except Exception:
            return dash.no_update

    return app