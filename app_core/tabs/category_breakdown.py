"""类别构成标签页布局定义。"""

from dash import dcc, html


def build_category_breakdown_tab():
    """构建类别构成页面，用于按簇/单位查看类别分布。"""
    return dcc.Tab(
        label='类别构成',
        value='category-breakdown',
        children=[
            html.Div([
                html.Div([
                    html.Label('类别字段:'),
                    dcc.Dropdown(
                        id='category-field-selector',
                        options=[
                            {'label': '部位 (part_C)', 'value': 'part_C'},
                            {'label': '类型 (type_C)', 'value': 'type_C'},
                            {'label': '单位 (unit_C)', 'value': 'unit_C'},
                        ],
                        value='part_C',
                        clearable=False,
                        style={'width': '220px'},
                    ),
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Label('横轴'),
                    dcc.RadioItems(
                        id='category-x-axis',
                        options=[
                            {'label': '按簇', 'value': 'cluster'},
                            {'label': '按单位 (unit_C)', 'value': 'unit_C'},
                        ],
                        value='cluster',
                        labelStyle={'marginRight': '12px'},
                    ),
                ], style={'marginBottom': '8px'}),
                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='category-breakdown-graph', style={'height': 'calc(100vh - 320px)'}),
                ),
                html.Hr(style={'margin': '16px 0', 'borderColor': '#e4e8ef'}),
                html.Div([
                    html.Span('器类验证矩阵', style={
                        'fontSize': '13px', 'fontWeight': '700', 'color': '#2c3e50',
                    }),
                    html.Span('type_C × cluster 混淆热力图及 ARI/NMI 纯度指数', style={
                        'fontSize': '11px', 'color': '#888', 'marginLeft': '8px',
                    }),
                ], style={'marginBottom': '8px', 'display': 'flex', 'alignItems': 'baseline'}),
                html.Div([
                    html.Div([
                        html.Label('归一化'),
                        dcc.RadioItems(
                            id='type-val-norm',
                            options=[
                                {'label': '按器类', 'value': 'by_type'},
                                {'label': '按簇', 'value': 'by_cluster'},
                                {'label': '绝对数', 'value': 'count'},
                            ],
                            value='by_type',
                            labelStyle={'marginRight': '12px'},
                        ),
                    ], style={'marginRight': '24px'}),
                    html.Div([
                        html.Label('Top-N 器类'),
                        dcc.Slider(
                            id='type-val-topn',
                            min=5, max=30, step=1, value=15,
                            marks={5: '5', 10: '10', 15: '15', 20: '20', 25: '25', 30: '30'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'flex': '1', 'minWidth': '240px'}),
                ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '8px'}),
                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='type-val-heatmap', style={'height': '340px'}),
                ),
                html.Div(id='type-val-metrics', style={'marginTop': '8px'}),
                html.Div(id='type-val-detail', style={'marginTop': '8px'}),
            ], style={'marginTop': '12px', 'padding': '0 8px'}),
        ],
    )
