"""聚类特征热力图回调。"""
import dash
from dash import Input, Output, State, dcc, html
import numpy as np

from data_processing import create_cluster_pattern_heatmap


def register_heatmap_callbacks(app):
    @app.callback(
        Output('heatmap-container', 'children'),
        [Input('visualization-tabs', 'value'),
         Input('reload-trigger', 'data')],
        State('cluster-metadata-store', 'data')
    )
    def update_heatmap(tab_value, _reload, cluster_metadata):
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
