"""簇分析标签页布局定义。"""

from dash import dcc, html


def build_cluster_analysis_tab():
    """构建簇分析页面。

    包含簇质量表、特征差异图、模式洞察和 Unit 层差异分析区域。
    """
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
            html.Div(
                id='cluster-pattern-insights',
                style={
                    'marginTop': '12px',
                    'padding': '12px',
                    'border': '1px solid #e6e6e6',
                    'borderRadius': '8px',
                    'backgroundColor': '#fafafa'
                }
            ),
            html.Div([
                html.Div('Unit 层差异分析', style={'fontWeight': '600', 'marginBottom': '8px'}),
                html.Div([
                    html.Div([
                        html.Label('Unit A'),
                        dcc.Dropdown(id='unit-compare-a', placeholder='选择 Unit A')
                    ], style={'flex': '1', 'minWidth': '200px'}),
                    html.Div([
                        html.Label('Unit B'),
                        dcc.Dropdown(id='unit-compare-b', placeholder='选择 Unit B')
                    ], style={'flex': '1', 'minWidth': '200px'}),
                ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '8px', 'padding': '0 4px'}),
                html.Div(id='unit-compare-summary', style={'padding': '6px 4px', 'color': '#333'}),
                dcc.Loading(
                    id='unit-compare-loading',
                    type='default',
                    children=dcc.Graph(id='unit-compare-graph', style={'height': '360px'})
                )
            ], style={
                'marginTop': '12px',
                'padding': '12px',
                'border': '1px solid #e6e6e6',
                'borderRadius': '8px',
                'backgroundColor': '#ffffff'
            }),
        ],
    )
