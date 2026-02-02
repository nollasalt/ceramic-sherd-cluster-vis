from dash import dcc, html


def build_cluster_quality_tab():
    return dcc.Tab(
        label='聚类质量',
        value='cluster-quality',
        children=[
            html.Div([
                dcc.Loading(
                    type='default',
                    children=html.Div([
                        html.Div(id='cluster-quality-cards', style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),
                        dcc.Graph(id='cluster-quality-bars', style={'height': '380px', 'width': '100%', 'marginBottom': '8px'}),
                        html.Div(id='cluster-quality-detail', style={'fontSize': '13px', 'color': '#333', 'padding': '0 4px'}),
                    ]),
                ),
                html.Div('颜色指示: 绿=清晰，黄=需关注，红=混杂/易粘连', style={'fontSize': '12px', 'color': '#666', 'marginTop': '4px', 'padding': '0 4px'}),
            ], style={'marginTop': '12px', 'padding': '0 8px'}),
        ],
    )
