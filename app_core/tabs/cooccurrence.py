"""簇共现分析标签页布局定义。"""

from dash import dcc, html


def build_cooccurrence_tab():
    """构建簇共现分析页面。

    展示在同一地层中同时出现的簇之间的共现矩阵热力图与聚类树状图。
    """
    return dcc.Tab(
        label='共现分析',
        value='cooccurrence',
        children=[
            # 控制区
            html.Div([
                html.Div([
                    html.Label('层位筛选'),
                    dcc.Dropdown(
                        id='cooc-unit-filter',
                        multi=True,
                        placeholder='默认使用全部层位',
                    ),
                ], style={'flex': '2', 'minWidth': '220px'}),
                html.Div([
                    html.Label('归一化方式'),
                    dcc.RadioItems(
                        id='cooc-norm-mode',
                        options=[
                            {'label': '原始计数', 'value': 'raw'},
                            {'label': 'Jaccard 相似度', 'value': 'jaccard'},
                            {'label': '条件概率 P(j|i)', 'value': 'conditional'},
                        ],
                        value='jaccard',
                        labelStyle={'display': 'block', 'marginBottom': '3px'},
                    ),
                ], style={'width': '180px'}),
                html.Div([
                    html.Label('最小共现层数'),
                    dcc.Slider(
                        id='cooc-min-count',
                        min=1,
                        max=10,
                        step=1,
                        value=1,
                        marks={1: '1', 3: '3', 5: '5', 10: '10'},
                        tooltip={'placement': 'bottom', 'always_visible': False},
                    ),
                ], style={'flex': '1', 'minWidth': '200px'}),
                html.Div([
                    html.Label('树状图联接方法'),
                    dcc.Dropdown(
                        id='cooc-linkage',
                        options=[
                            {'label': 'Average', 'value': 'average'},
                            {'label': 'Complete', 'value': 'complete'},
                            {'label': 'Ward', 'value': 'ward'},
                            {'label': 'Single', 'value': 'single'},
                        ],
                        value='average',
                        clearable=False,
                    ),
                ], style={'width': '160px'}),
            ], style={
                'display': 'flex',
                'alignItems': 'flex-start',
                'gap': '16px',
                'marginBottom': '12px',
                'padding': '8px',
                'flexWrap': 'wrap',
            }),

            # 树状图
            html.Div([
                html.Div('簇聚类树状图（按共现相似度）', style={
                    'fontWeight': '600', 'marginBottom': '6px', 'fontSize': '14px', 'color': '#333',
                }),
                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='cooc-dendrogram', style={'height': '260px'}),
                ),
            ], style={
                'padding': '12px',
                'border': '1px solid #e6e6e6',
                'borderRadius': '8px',
                'backgroundColor': '#fff',
                'marginBottom': '12px',
            }),

            # 热力图 + 统计并排
            html.Div([
                html.Div([
                    html.Div('共现矩阵热力图', style={
                        'fontWeight': '600', 'marginBottom': '6px', 'fontSize': '14px', 'color': '#333',
                    }),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(id='cooc-heatmap', style={'height': '480px'}),
                    ),
                ], style={
                    'flex': '3',
                    'minWidth': '360px',
                    'padding': '12px',
                    'border': '1px solid #e6e6e6',
                    'borderRadius': '8px',
                    'backgroundColor': '#fff',
                }),
                html.Div([
                    html.Div('共现统计', style={
                        'fontWeight': '600', 'marginBottom': '8px', 'fontSize': '14px', 'color': '#333',
                    }),
                    html.Div(id='cooc-stats'),
                ], style={
                    'flex': '1',
                    'minWidth': '200px',
                    'padding': '12px',
                    'border': '1px solid #e6e6e6',
                    'borderRadius': '8px',
                    'backgroundColor': '#fafafa',
                }),
            ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'flex-start'}),
        ],
    )
