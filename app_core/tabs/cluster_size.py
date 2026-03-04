"""簇规模分布标签页布局定义。"""

from dash import dcc, html


def build_cluster_size_tab():
    """构建簇规模可视化页面。"""
    return dcc.Tab(
        label='簇规模分布',
        value='cluster-size',
        children=[
            html.Div([
                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='cluster-size-graph', style={'height': 'calc(100vh - 200px)'}),
                )
            ], style={'marginTop': '12px'}),
        ],
    )
