"""簇规模分布标签页布局定义。"""

from dash import dcc, html


def build_cluster_size_tab():
    """构建簇规模可视化页面。"""
    return dcc.Tab(
        label='簇规模分布',
        value='cluster-size',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面展示各簇的样本数量分布，帮助识别大簇和小簇。柱状图直观显示每个簇包含的陶片数量，便于评估聚类结果的均衡性。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#fce4ec', 'border': '1px solid #f48fb1',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='cluster-size-graph', style={'height': 'calc(100vh - 200px)'}),
                )
            ], style={'marginTop': '12px'}),
        ],
    )
