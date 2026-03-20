"""簇间距离网络图回调：质心 kNN 网络 + 点击查看近邻。"""

import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from app_core.data_cache import get_data_cache
from app_core.utils import CLUSTER_COLORS
from performance_utils import cache_plot_result


def _cosine_distance_matrix(X):
    """计算行向量两两余弦距离矩阵，返回 (N, N) float32。"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    X_norm = X / norms
    sim = X_norm @ X_norm.T
    return 1.0 - np.clip(sim, -1, 1)


def _euclidean_distance_matrix(X):
    """计算欧氏距离矩阵（标准化后）。"""
    from sklearn.preprocessing import StandardScaler
    X_s = StandardScaler().fit_transform(X)
    sq = np.sum(X_s ** 2, axis=1, keepdims=True)
    dist_sq = np.maximum(sq + sq.T - 2 * (X_s @ X_s.T), 0)
    return np.sqrt(dist_sq)


def _layout_pca(centroids):
    """用 PCA 把质心投影到 2D。"""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(centroids)


def _layout_umap(centroids):
    """用 UMAP 把质心投影到 2D（fallback PCA）。"""
    try:
        from umap import UMAP
        return UMAP(n_components=2, n_neighbors=min(10, len(centroids) - 1),
                    random_state=42).fit_transform(centroids)
    except Exception:
        return _layout_pca(centroids)


def _dominant_label(df, cluster_col, cid, col):
    """取某簇在指定列中最多的标签值，缺失返回 '?'。"""
    if col not in df.columns:
        return '?'
    sub = df[df[cluster_col] == cid][col].dropna()
    if len(sub) == 0:
        return '?'
    return str(sub.mode().iloc[0])


def register_cluster_network_callbacks(app):

    @app.callback(
        Output('cluster-network-graph', 'figure'),
        Output('cluster-network-detail', 'children'),
        Input('visualization-tabs', 'value'),
        Input('net-node-color', 'value'),
        Input('net-metric', 'value'),
        Input('net-layout', 'value'),
        Input('net-knn', 'value'),
        Input('net-max-clusters', 'value'),
        Input('cluster-filter', 'value'),
        State('data-store', 'data'),
    )
    @cache_plot_result
    def render_cluster_network(
        tab_value, node_color_by, metric, layout_algo,
        knn, max_clusters,
        selected_clusters, _data_store,
    ):
        if tab_value != 'cluster-network':
            return dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
        feature_cols = data_cache['feature_cols']

        # ── 筛选簇 ────────────────────────────────────────────────────────
        dff = df
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]

        # 按样本数降序取 top max_clusters
        max_clusters = int(max_clusters or 100)
        cluster_sizes = dff[cluster_col].value_counts()
        top_clusters = cluster_sizes.head(max_clusters).index.tolist()
        dff = dff[dff[cluster_col].isin(top_clusters)]

        clusters = sorted(dff[cluster_col].unique())
        n = len(clusters)

        if n < 3:
            empty = go.Figure()
            empty.update_layout(title='簇数量不足，请放宽筛选条件')
            return empty, html.Div('暂无数据')

        cid_to_idx = {c: i for i, c in enumerate(clusters)}

        # ── 计算质心 ──────────────────────────────────────────────────────
        centroids = np.array([
            dff[dff[cluster_col] == c][feature_cols].mean().values
            for c in clusters
        ], dtype=np.float32)

        # ── 距离矩阵 → 相似度 ─────────────────────────────────────────────
        knn = int(knn or 3)
        if metric == 'cosine':
            dist_mat = _cosine_distance_matrix(centroids)
        else:
            dist_mat = _euclidean_distance_matrix(centroids)

        # 转相似度 [0,1]
        d_max = dist_mat.max()
        sim_mat = 1.0 - dist_mat / (d_max + 1e-10)
        np.fill_diagonal(sim_mat, 0)

        # ── kNN 边 ────────────────────────────────────────────────────────
        edges = set()
        for i in range(n):
            neighbors = np.argsort(dist_mat[i])[:knn + 1]
            for j in neighbors:
                if j != i:
                    edges.add((min(i, j), max(i, j)))

        # ── 布局（质心低维投影） ───────────────────────────────────────────
        if layout_algo == 'umap':
            pos = _layout_umap(centroids)
        else:
            pos = _layout_pca(centroids)

        # ── 节点属性 ──────────────────────────────────────────────────────
        node_sizes = np.array([int(cluster_sizes.get(c, 1)) for c in clusters])
        node_sizes_scaled = 8 + 22 * (node_sizes / node_sizes.max())

        if node_color_by == 'type' and 'type_C' in df.columns:
            labels = [_dominant_label(dff, cluster_col, c, 'type_C') for c in clusters]
        elif node_color_by == 'part' and 'part_C' in df.columns:
            labels = [_dominant_label(dff, cluster_col, c, 'part_C') for c in clusters]
        else:
            labels = [str(c) for c in clusters]

        # 给每个唯一标签分配颜色
        unique_labels = list(dict.fromkeys(labels))
        label_color = {
            lbl: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for i, lbl in enumerate(unique_labels)
        }
        node_colors = [label_color[lbl] for lbl in labels]

        # ── 绘图 ──────────────────────────────────────────────────────────
        fig = go.Figure()

        # 边（按相似度分组，减少 trace 数量）
        sim_values = [sim_mat[i, j] for i, j in edges]
        if sim_values:
            sim_arr = np.array(sim_values)
            # 分 3 组宽度
            thresholds = np.percentile(sim_arr, [33, 66])
            for group_idx, (width, alpha, label_suffix) in enumerate([
                (0.8, 0.15, '弱'),
                (1.5, 0.25, '中'),
                (2.5, 0.45, '强'),
            ]):
                ex, ey = [], []
                for k, (i, j) in enumerate(edges):
                    s = sim_values[k]
                    if group_idx == 0 and s > thresholds[0]:
                        continue
                    if group_idx == 1 and not (thresholds[0] < s <= thresholds[1]):
                        continue
                    if group_idx == 2 and s <= thresholds[1]:
                        continue
                    ex += [pos[i, 0], pos[j, 0], None]
                    ey += [pos[i, 1], pos[j, 1], None]
                if ex:
                    fig.add_trace(go.Scatter(
                        x=ex, y=ey,
                        mode='lines',
                        line=dict(width=width, color=f'rgba(120,120,140,{alpha})'),
                        hoverinfo='skip',
                        showlegend=False,
                    ))

        # 节点（按标签分组，便于图例）
        for lbl in unique_labels:
            idxs = [i for i, l in enumerate(labels) if l == lbl]
            hover = [
                f'簇 {clusters[i]}<br>样本数: {node_sizes[i]}<br>标签: {labels[i]}'
                for i in idxs
            ]
            fig.add_trace(go.Scatter(
                x=pos[idxs, 0],
                y=pos[idxs, 1],
                mode='markers+text',
                name=lbl,
                marker=dict(
                    size=[node_sizes_scaled[i] for i in idxs],
                    color=label_color[lbl],
                    line=dict(width=1, color='white'),
                    opacity=0.88,
                ),
                text=[str(clusters[i]) for i in idxs],
                textposition='middle center',
                textfont=dict(size=8, color='white'),
                customdata=[clusters[i] for i in idxs],
                hovertext=hover,
                hovertemplate='%{hovertext}<extra></extra>',
            ))

        layout_label = 'PCA' if layout_algo == 'pca' else 'UMAP'
        fig.update_layout(
            title=f'簇质心 kNN 网络（{metric}，k={knn}，{layout_label} 布局，共 {n} 簇）',
            showlegend=len(unique_labels) <= 20,
            legend=dict(
                orientation='v', x=1.01, y=1,
                font=dict(size=10), itemsizing='constant',
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=160, t=50, b=20),
            plot_bgcolor='#f8fafc',
            paper_bgcolor='#fff',
            hovermode='closest',
        )

        # ── 点击节点近邻说明（静态展示距离最近的 K 簇） ──────────────────
        # 取全局距离最小的 top-5 对
        flat = [(dist_mat[i, j], clusters[i], clusters[j])
                for i in range(n) for j in range(i + 1, n)]
        flat.sort()
        top_pairs = flat[:10]

        rows = []
        for d, ca, cb in top_pairs:
            s = 1 - d / (d_max + 1e-10)
            rows.append(html.Tr([
                html.Td(f'簇 {ca}', style={'padding': '4px 8px', 'fontSize': '12px', 'fontWeight': '600'}),
                html.Td('↔', style={'padding': '4px 4px', 'color': '#aaa'}),
                html.Td(f'簇 {cb}', style={'padding': '4px 8px', 'fontSize': '12px', 'fontWeight': '600'}),
                html.Td(f'{s:.3f}', style={
                    'padding': '4px 8px', 'fontSize': '12px',
                    'color': '#27ae60' if s > 0.7 else '#e67e22',
                }),
            ]))

        detail = html.Div([
            html.Div('全局最相似簇对（Top 10）', style={
                'fontSize': '12px', 'fontWeight': '700', 'color': '#2c3e50', 'marginBottom': '8px',
            }),
            html.Table([
                html.Thead(html.Tr([
                    html.Th('簇 A', style={'padding': '4px 8px', 'fontSize': '11px', 'color': '#888'}),
                    html.Th('', style={'padding': '4px 4px'}),
                    html.Th('簇 B', style={'padding': '4px 8px', 'fontSize': '11px', 'color': '#888'}),
                    html.Th('相似度', style={'padding': '4px 8px', 'fontSize': '11px', 'color': '#888'}),
                ])),
                html.Tbody(rows),
            ], style={'borderCollapse': 'collapse'}),
        ])

        return fig, detail
