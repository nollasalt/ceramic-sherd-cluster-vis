"""簇分析标签页布局定义。"""

from dash import dcc, html


def build_cluster_analysis_tab():
    """构建簇分析页面。

    包含簇质量表、特征差异图和模式洞察。
    """
    return dcc.Tab(
        label='簇分析',
        value='cluster-analysis',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面提供簇质量评估和特征差异分析。顶部展示各簇的规模、纯度、紧密度等指标，选择特定簇后可查看其与全局均值的特征差异，帮助理解该簇的独特性。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#e8f5e9', 'border': '1px solid #a5d6a7',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

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
                html.Div([
                    html.Div(id='cluster-quality-table', style={'flex': '1', 'minWidth': '320px'}),
                    dcc.Store(id='analysis-table-page-index', data=1),
                    html.Div([
                        html.Button('上一页', id='analysis-table-prev', n_clicks=0),
                        html.Div(id='analysis-table-page-status', style={'fontSize': '13px', 'color': '#555'}),
                        html.Button('下一页', id='analysis-table-next', n_clicks=0),
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'gap': '8px', 'marginTop': '8px'})
                ], style={'flex': '1', 'minWidth': '320px', 'padding': '0 8px'}),
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
        ],
    )
