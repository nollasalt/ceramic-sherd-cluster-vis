from dash import dcc, html


def build_cluster_analysis_tab():
    return dcc.Tab(
        label='簇分析',
        value='cluster-analysis',
        children=[
            html.Div([
                html.Div([
                    html.Label('选择簇'),
                    dcc.Dropdown(id='analysis-cluster-selector', placeholder='选择一个簇查看特征差异'),
                ], style={'width': '200px', 'marginRight': '12px'}),
                html.Div([
                    html.Label('Top-K 特征'),
                    dcc.Slider(
                        id='feature-topk-slider',
                        min=3,
                        max=20,
                        step=1,
                        value=8,
                        marks={3: '3', 5: '5', 8: '8', 12: '12', 16: '16', 20: '20'},
                        tooltip={'placement': 'bottom', 'always_visible': False},
                    ),
                ], style={'flex': '1', 'minWidth': '240px'}),
                html.Div([
                    html.Label('差异度量'),
                    dcc.RadioItems(
                        id='feature-diff-mode',
                        options=[
                            {'label': '均值差', 'value': 'mean'},
                            {'label': 'z-score', 'value': 'zscore'},
                        ],
                        value='mean',
                        labelStyle={'marginRight': '12px'},
                    ),
                ], style={'width': '200px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '12px', 'padding': '0 8px'}),
            html.Div([
                html.Div(id='cluster-quality-table', style={'flex': '1', 'minWidth': '320px', 'padding': '0 8px'}),
                html.Div(
                    dcc.Loading(id='feature-diff-loading', type='default', children=dcc.Graph(id='feature-diff-graph', style={'height': '420px'})),
                    style={'flex': '1', 'minWidth': '320px', 'padding': '0 8px'},
                ),
            ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'flex-start'}),
        ],
    )
