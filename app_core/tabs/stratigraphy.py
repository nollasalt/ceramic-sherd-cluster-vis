"""地层流动分析标签页布局定义。"""

from dash import dcc, html

_CARD = {'padding': '14px 16px', 'border': '1px solid #e4e8ef',
         'borderRadius': '10px', 'backgroundColor': '#fff',
         'boxShadow': '0 1px 4px rgba(0,0,0,0.05)', 'marginBottom': '12px'}

_CARD_TITLE = {'fontSize': '13px', 'fontWeight': '700', 'color': '#2c3e50',
               'margin': '0 0 10px 0', 'letterSpacing': '0.02em'}


def build_stratigraphy_tab():
    """构建地层流动分析页面。

    包含 Sankey 图（层位→簇）、跨层热力图和统计摘要。
    """
    return dcc.Tab(
        label='地层流动',
        value='stratigraphy',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面展示器物群（簇）在不同地层中的流动与分布。Sankey图直观显示从地层到簇的流向，热力图展示簇的跨层分布模式，帮助识别贯穿多层的长期使用器物群和集中于单层的特定事件器物群。统计摘要提供各层多样性和簇持续性指标。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#e0f7fa', 'border': '1px solid #80deea',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                # ── 控制区 ─────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('层位筛选'),
                        dcc.Dropdown(
                            id='strat-unit-filter',
                            multi=True,
                            placeholder='默认显示全部层位',
                        ),
                    ], style={'flex': '2', 'minWidth': '200px'}),

                    html.Div([
                        html.Label('簇筛选'),
                        dcc.Dropdown(
                            id='strat-cluster-filter',
                            multi=True,
                            placeholder='默认显示全部簇',
                        ),
                    ], style={'flex': '2', 'minWidth': '160px'}),

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
                            labelStyle={'display': 'block', 'marginBottom': '3px'},
                        ),
                    ], style={'minWidth': '130px'}),

                    html.Div([
                        html.Label('Sankey 最小连线（片）'),
                        dcc.Slider(
                            id='strat-min-link',
                            min=1, max=50, step=1, value=5,
                            marks={1: '1', 10: '10', 25: '25', 50: '50'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'flex': '3', 'minWidth': '200px'}),
                ], className='analysis-control-bar'),

                # ── Sankey 图 ───────────────────────────────────────────────
                html.Div([
                    html.P('层位 → 簇 流向图', className='dash-card-title'),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(id='stratigraphy-sankey'),
                    ),
                ], style=_CARD),

                # ── 热力图 + 统计并排 ────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.P('簇跨层分布热力图', className='dash-card-title'),
                        dcc.Loading(
                            type='default',
                            children=dcc.Graph(id='stratigraphy-heatmap', style={'height': '400px'}),
                        ),
                    ], style={**_CARD, 'flex': '2', 'minWidth': '300px', 'marginBottom': '0'}),

                    html.Div([
                        html.P('统计摘要', className='dash-card-title'),
                        html.Div(id='stratigraphy-stats'),
                    ], style={**_CARD, 'flex': '1', 'minWidth': '200px',
                               'backgroundColor': '#f8fafc', 'marginBottom': '0'}),
                ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'flex-start'}),

            ], style={'padding': '14px'}),
        ],
    )
