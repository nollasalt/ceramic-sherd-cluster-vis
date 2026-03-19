"""代表样本标签页布局定义。"""

from dash import dcc, html


def build_representatives_tab():
    """构建代表样本展示标签页。"""
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
                dcc.Loading(
                    id='rep-grid-loading',
                    type='default',
                    children=html.Div(id='representative-grid', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px'}),
                ),
                dcc.Store(id='rep-page-index', data=1),
                html.Div([
                    html.Button(
                        '上一页',
                        id='rep-page-prev',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#ffffff',
                            'color': '#333',
                            'border': '1px solid #ccc',
                            'padding': '6px 14px',
                            'cursor': 'pointer',
                            'borderRadius': '4px'
                        }
                    ),
                    html.Div(id='rep-page-status', style={'fontSize': '13px', 'color': '#555'}),
                    html.Button(
                        '下一页',
                        id='rep-page-next',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#ffffff',
                            'color': '#333',
                            'border': '1px solid #ccc',
                            'padding': '6px 14px',
                            'cursor': 'pointer',
                            'borderRadius': '4px'
                        }
                    ),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginTop': '8px'}),
            ], style={'marginTop': '12px', 'padding': '0 8px'}),
        ],
    )
