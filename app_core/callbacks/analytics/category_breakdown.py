"""类别构成分布图回调。"""
import dash
from dash import Input, Output, State
import pandas as pd
import plotly.express as px

from app_core.data_cache import get_data_cache
from performance_utils import cache_plot_result


def register_category_breakdown_callbacks(app):
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
            .groupby([x_field, category_field], observed=True)
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
