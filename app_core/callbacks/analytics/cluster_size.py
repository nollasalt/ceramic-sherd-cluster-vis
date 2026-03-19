"""簇规模分布图回调。"""
import dash
from dash import Input, Output, State
import plotly.express as px

from app_core.data_cache import get_data_cache
from app_core.utils import CLUSTER_COLORS
from performance_utils import cache_plot_result


def register_cluster_size_callbacks(app):
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
