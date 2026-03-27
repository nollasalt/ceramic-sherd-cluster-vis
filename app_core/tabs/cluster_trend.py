"""跨层簇趋势分析标签页布局（折线图 + 趋势线分类）。"""

from dash import dcc, html

_CARD = {'padding': '14px 16px', 'border': '1px solid #e4e8ef',
         'borderRadius': '10px', 'backgroundColor': '#fff',
         'boxShadow': '0 1px 4px rgba(0,0,0,0.05)', 'marginBottom': '12px'}
_CARD_TITLE = {'fontSize': '13px', 'fontWeight': '700', 'color': '#2c3e50',
               'margin': '0 0 10px 0', 'letterSpacing': '0.02em'}


def build_cluster_trend_tab():
    """构建跨层簇趋势折线图页面。

    X 轴 = 地层时序，Y 轴 = 该层内该簇占比，
    附趋势线拟合，自动识别兴起型/衰落型/瞬间型/稳定型器物群。
    """
    return dcc.Tab(
        label='跨层趋势',
        value='cluster-trend',
        children=[
            html.Div([

                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面展示各簇在不同地层中的分布趋势，自动识别兴起型（↑）、衰落型（↓）、瞬间型（⚡）、稳定型（─）四种器物群演变模式。使用层位筛选快捷按钮可快速选择分析范围，建议同时筛选不超过20个簇以保持图表清晰。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#fff8e1', 'border': '1px solid #ffe082',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                # ── 控制区 ─────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('层位筛选'),
                        dcc.Dropdown(
                            id='trend-unit-filter',
                            multi=True,
                            placeholder='默认全部层位',
                        ),
                        html.Div([
                            html.Button('全选', id='trend-unit-select-all', n_clicks=0,
                                       style={'fontSize': '11px', 'padding': '2px 8px', 'marginRight': '4px'}),
                            html.Button('清空', id='trend-unit-clear', n_clicks=0,
                                       style={'fontSize': '11px', 'padding': '2px 8px', 'marginRight': '4px'}),
                            html.Button('仅主要层', id='trend-unit-main', n_clicks=0,
                                       style={'fontSize': '11px', 'padding': '2px 8px'}),
                        ], style={'marginTop': '4px'}),
                    ], style={'flex': '2', 'minWidth': '180px'}),

                    html.Div([
                        html.Label('簇筛选'),
                        dcc.Dropdown(
                            id='trend-cluster-filter',
                            multi=True,
                            placeholder='默认全部（建议选 ≤20 个）',
                        ),
                    ], style={'flex': '2', 'minWidth': '200px'}),

                    html.Div([
                        html.Label('显示类型'),
                        dcc.Checklist(
                            id='trend-type-filter',
                            options=[
                                {'label': '兴起型 ↑', 'value': 'rising'},
                                {'label': '衰落型 ↓', 'value': 'declining'},
                                {'label': '瞬间型 ⚡', 'value': 'transient'},
                                {'label': '稳定型 ─', 'value': 'stable'},
                            ],
                            value=['rising', 'declining', 'transient', 'stable'],
                            labelStyle={'display': 'block', 'marginBottom': '3px'},
                        ),
                    ], style={'minWidth': '120px'}),

                    html.Div([
                        html.Div([
                            html.Label('Y 轴'),
                            dcc.RadioItems(
                                id='trend-y-mode',
                                options=[
                                    {'label': '层内占比', 'value': 'by_layer'},
                                    {'label': '绝对数', 'value': 'count'},
                                ],
                                value='by_layer',
                                labelStyle={'display': 'block', 'marginBottom': '3px'},
                            ),
                        ], style={'marginBottom': '10px'}),
                        html.Div([
                            html.Label('最少出现层数'),
                            dcc.Slider(
                                id='trend-min-layers',
                                min=1, max=8, step=1, value=2,
                                marks={1: '1', 2: '2', 4: '4', 6: '6', 8: '8'},
                                tooltip={'placement': 'bottom', 'always_visible': False},
                            ),
                        ]),
                    ], style={'flex': '2', 'minWidth': '180px'}),

                ], className='analysis-control-bar'),

                # ── 趋势折线图 ──────────────────────────────────────────────
                html.Div([
                    html.P('簇跨层占比趋势（附线性趋势线）', style=_CARD_TITLE),
                    html.P(
                        '实线 = 实际占比；虚线 = 线性趋势拟合。'
                        '地层顺序从左（最新/浅）到右（最早/深）。',
                        style={'fontSize': '11px', 'color': '#888', 'margin': '-4px 0 8px'},
                    ),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(id='cluster-trend-chart', style={'height': 'calc(100vh - 400px)'}),
                    ),
                ], style=_CARD),

                # ── 分类汇总 + 明细表 ────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.P('趋势分类汇总', style=_CARD_TITLE),
                        html.Div(id='cluster-trend-summary'),
                    ], style={**_CARD, 'flex': '1', 'minWidth': '220px',
                               'backgroundColor': '#f8fafc', 'marginBottom': '0'}),

                    html.Div([
                        html.P('逐簇趋势明细', style=_CARD_TITLE),
                        html.Div(id='cluster-trend-detail'),
                    ], style={**_CARD, 'flex': '3', 'minWidth': '320px', 'marginBottom': '0'}),

                ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'flex-start'}),

            ], style={'padding': '14px'}),
        ],
    )
