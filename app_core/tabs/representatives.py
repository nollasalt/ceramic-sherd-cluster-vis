from dash import dcc, html


def build_representatives_tab():
    return dcc.Tab(
        label='代表样本',
        value='representatives',
        children=[
            html.Div([
                html.Div([
                    html.Label('每簇展示张数'),
                    dcc.Slider(
                        id='rep-samples-per-cluster',
                        min=1,
                        max=12,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in [1, 2, 3, 4, 6, 8, 10, 12]},
                        tooltip={'placement': 'bottom', 'always_visible': False},
                    ),
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Label('代表样本选择'),
                    dcc.RadioItems(
                        id='rep-strategy',
                        options=[
                            {'label': '最近中心', 'value': 'center'},
                            {'label': '随机', 'value': 'random'},
                        ],
                        value='center',
                        labelStyle={'marginRight': '12px'},
                    ),
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Label('每簇离群样本数'),
                    dcc.Slider(
                        id='outlier-count',
                        min=1,
                        max=5,
                        step=1,
                        value=2,
                        marks={1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
                        tooltip={'placement': 'bottom', 'always_visible': False},
                    ),
                ], style={'marginBottom': '8px'}),
                dcc.Loading(
                    id='rep-grid-loading',
                    type='default',
                    children=html.Div(id='representative-grid', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px'}),
                ),
                html.Div(id='outlier-list', style={'marginTop': '12px', 'fontSize': '13px', 'color': '#333'}),
            ], style={'marginTop': '12px', 'padding': '0 8px'}),
        ],
    )
