"""地层流动分析标签页布局定义。"""

from dash import dcc, html


def build_stratigraphy_tab():
    """构建地层流动分析页面。

    包含 Sankey 图（层位→簇）、跨层热力图和统计摘要。
    """
    return dcc.Tab(
        label='地层流动',
        value='stratigraphy',
        children=[
            # 控制区
            html.Div([
                html.Div([
                    html.Label('层位筛选'),
                    dcc.Dropdown(
                        id='strat-unit-filter',
                        multi=True,
                        placeholder='默认显示全部层位',
                    ),
                ], style={'flex': '2', 'minWidth': '220px'}),
                html.Div([
                    html.Label('簇筛选'),
                    dcc.Dropdown(
                        id='strat-cluster-filter',
                        multi=True,
                        placeholder='默认显示全部簇',
                    ),
                ], style={'flex': '2', 'minWidth': '180px'}),
                html.Div([
                    html.Label('热力图模式'),
                    dcc.RadioItems(
                        id='strat-heatmap-mode',
                        options=[
                            {'label': '绝对数', 'value': 'count'},
                            {'label': '按层归一化', 'value': 'by_layer'},
                            {'label': '按簇归一化', 'value': 'by_cluster'},
                        ],
                        value='count',
                        labelStyle={'display': 'block', 'marginBottom': '2px'},
                    ),
                ], style={'width': '150px'}),
                html.Div([
                    html.Label('Sankey 最小连线（片）'),
                    dcc.Slider(
                        id='strat-min-link',
                        min=1,
                        max=50,
                        step=1,
                        value=5,
                        marks={1: '1', 10: '10', 25: '25', 50: '50'},
                        tooltip={'placement': 'bottom', 'always_visible': False},
                    ),
                ], style={'flex': '2', 'minWidth': '200px'}),
            ], style={
                'display': 'flex',
                'alignItems': 'flex-start',
                'gap': '16px',
                'marginBottom': '12px',
                'padding': '8px',
                'flexWrap': 'wrap',
            }),

            # Sankey 图
            html.Div([
                html.Div('层位 → 簇 流向图', style={
                    'fontWeight': '600',
                    'marginBottom': '6px',
                    'fontSize': '14px',
                    'color': '#333',
                }),
                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='stratigraphy-sankey', style={'height': '520px'}),
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
                    html.Div('簇跨层分布热力图', style={
                        'fontWeight': '600',
                        'marginBottom': '6px',
                        'fontSize': '14px',
                        'color': '#333',
                    }),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(id='stratigraphy-heatmap', style={'height': '420px'}),
                    ),
                ], style={
                    'flex': '2',
                    'minWidth': '320px',
                    'padding': '12px',
                    'border': '1px solid #e6e6e6',
                    'borderRadius': '8px',
                    'backgroundColor': '#fff',
                }),
                html.Div([
                    html.Div('统计摘要', style={
                        'fontWeight': '600',
                        'marginBottom': '8px',
                        'fontSize': '14px',
                        'color': '#333',
                    }),
                    html.Div(id='stratigraphy-stats'),
                ], style={
                    'flex': '1',
                    'minWidth': '220px',
                    'padding': '12px',
                    'border': '1px solid #e6e6e6',
                    'borderRadius': '8px',
                    'backgroundColor': '#fafafa',
                }),
            ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'flex-start'}),
        ],
    )
