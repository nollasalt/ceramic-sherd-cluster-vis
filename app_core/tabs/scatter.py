from dash import dcc, html


def build_scatter_tab(fig, clusters, init_unit_options, init_part_options, init_type_options, algorithm_options):
    return dcc.Tab(label='散点图', value='scatter', children=[
        html.Div([
            html.Div([
                html.Div([
                    html.Label('降维算法:'),
                    dcc.Dropdown(id='algorithm-selector', options=algorithm_options, value='umap'),
                ], style={'width': '200px'}),
                html.Div([
                    html.Label('可视化维度:'),
                    dcc.RadioItems(id='dimension-selector', options=[{'label': '2D', 'value': 2}, {'label': '3D', 'value': 3}], value=2),
                ], style={'width': '180px'}),
                html.Div([
                    html.Label('3D z轴:'),
                    dcc.Dropdown(
                        id='z-axis-selector',
                        options=[{'label': '降维结果', 'value': 'dimension'}, {'label': 'unit_C', 'value': 'unit_C'}],
                        value='dimension',
                    ),
                ], style={'width': '220px'}),
            ], style={'marginBottom': '8px', 'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px 12px', 'alignItems': 'flex-end'}),
            html.Div([
                html.Div([
                    html.Label('筛选簇:'),
                    dcc.Dropdown(id='cluster-filter', options=[{'label': str(c), 'value': c} for c in clusters], multi=True, placeholder='选择一个或多个簇（留空显示所有）'),
                ], style={'width': '240px'}),
                html.Div([
                    html.Label('筛选单位:'),
                    dcc.Dropdown(id='unit-filter', options=init_unit_options, multi=True, placeholder='选择单位（留空显示所有）'),
                ], style={'width': '200px'}),
                html.Div([
                    html.Label('筛选部位:'),
                    dcc.Dropdown(id='part-filter', options=init_part_options, multi=True, placeholder='选择部位（留空显示所有）'),
                ], style={'width': '200px'}),
                html.Div([
                    html.Label('筛选类型:'),
                    dcc.Dropdown(id='type-filter', options=init_type_options, multi=True, placeholder='选择类型（留空显示所有）'),
                ], style={'width': '200px'}),
            ], style={'marginBottom': '6px', 'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px 12px', 'alignItems': 'flex-end'}),
        ], style={'marginBottom': '16px'}),
        html.Div([
            html.Div([
                dcc.Loading(
                    id='plot-loading',
                    type='default',
                    children=[
                        dcc.Graph(
                            id='tsne-plot',
                            figure=fig,
                            style={'height': 'calc(100vh - 380px)', 'width': '100%'},
                            clear_on_unhover=True,
                        )
                    ],
                ),
                html.Div(id='plot-loading-status', style={'textAlign': 'center', 'marginTop': '8px', 'color': '#666'}),
            ], style={'flex': '3 1 0%'}),
            html.Div([
                html.Div([
                    html.Button('Prev', id='cluster-prev', n_clicks=0),
                    html.Button('Next', id='cluster-next', n_clicks=0),
                    html.Span(id='page-indicator', style={'marginLeft': '8px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '8px'}),
                html.Div(id='cluster-panel', style={'height': 'calc(100vh - 420px)', 'overflowY': 'auto'}),
                dcc.Store(id='cluster-images-store'),
                dcc.Store(id='cluster-page', data=1),
            ], style={'width': '320px', 'flex': '1 0 320px', 'borderLeft': '1px solid #ddd', 'padding': '8px', 'boxSizing': 'border-box'}),
        ], style={'display': 'flex', 'gap': '4px', 'height': 'calc(100vh - 260px)'}),
        html.Div([
            html.Button('添加到比较', id='compare-add', style={'width': '120px', 'backgroundColor': '#0066cc', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'padding': '6px 10px'}),
            html.Button('清空比较', id='compare-clear', style={'width': '120px'}),
            html.Span('（先点击散点图选中样本，再添加到比较）', style={'marginLeft': '10px', 'color': '#666'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginTop': '8px', 'marginBottom': '8px'}),
    ])
