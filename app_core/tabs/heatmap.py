from dash import dcc, html


def build_heatmap_tab():
    return dcc.Tab(
        label='聚类特征热力图',
        value='heatmap',
        children=[
            html.Div([
                dcc.Loading(
                    type='default',
                    children=html.Div(id='heatmap-container', style={'height': 'calc(100vh - 180px)', 'width': '100%'}),
                )
            ], style={'marginTop': '12px'}),
        ],
    )
