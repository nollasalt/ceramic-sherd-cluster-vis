from dash import dcc, html


def build_similarity_tab():
    return dcc.Tab(
        label='聚类相似度矩阵',
        value='similarity',
        children=[
            html.Div([
                html.Div([
                    html.Label('矩阵类型:'),
                    dcc.RadioItems(
                        id='similarity-metric',
                        options=[
                            {'label': '相似度 (余弦)', 'value': 'cosine'},
                            {'label': '距离 (欧氏)', 'value': 'euclidean'},
                        ],
                        value='cosine',
                        labelStyle={'marginRight': '12px'},
                    ),
                    dcc.Checklist(
                        id='similarity-options',
                        options=[
                            {'label': '层次重排', 'value': 'reorder'},
                            {'label': '显示数值', 'value': 'annotate'},
                        ],
                        value=[],
                        style={'marginTop': '4px'},
                    ),
                    html.Div([
                        html.Label('最近邻簇数量'),
                        dcc.Slider(
                            id='similarity-neighbor-k',
                            min=1,
                            max=10,
                            step=1,
                            value=3,
                            marks={1: '1', 3: '3', 5: '5', 7: '7', 10: '10'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'marginTop': '6px'}),
                ], style={'marginBottom': '8px'}),
                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='similarity-graph', style={'height': 'calc(100vh - 240px)', 'width': '100%'}),
                ),
                html.Div(id='nearest-cluster-list', style={'marginTop': '8px', 'fontSize': '13px', 'color': '#333'}),
            ], style={'marginTop': '12px'}),
        ],
    )
