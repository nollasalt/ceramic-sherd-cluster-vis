"""簇中心相似度/距离矩阵及最近邻列表回调。"""
import dash
from dash import Input, Output, State, html
import numpy as np
import plotly.express as px

from app_core.data_cache import get_data_cache
from performance_utils import cache_plot_result

try:
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def register_similarity_callbacks(app):
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
