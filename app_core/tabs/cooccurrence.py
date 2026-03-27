"""簇共现分析标签页布局定义。"""

from dash import dcc, html

_CARD = {'padding': '14px 16px', 'border': '1px solid #e4e8ef',
         'borderRadius': '10px', 'backgroundColor': '#fff',
         'boxShadow': '0 1px 4px rgba(0,0,0,0.05)', 'marginBottom': '12px'}


def build_cooccurrence_tab():
    """构建簇共现分析页面。

    展示在同一地层中同时出现的簇之间的共现矩阵热力图与聚类树状图。
    """
    return dcc.Tab(
        label='共现分析',
        value='cooccurrence',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面分析不同簇在同一地层中共同出现的模式。共现矩阵热力图显示簇对在多少层位中同时出现，树状图按共现相似度对簇进行聚类。高共现度表明这些器物群可能属于同一使用场景或时期，有助于理解器物组合关系。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#fff9c4', 'border': '1px solid #fff176',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                # ── 控制区 ─────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('层位筛选'),
                        dcc.Dropdown(
                            id='cooc-unit-filter',
                            multi=True,
                            placeholder='默认使用全部层位',
                        ),
                    ], style={'flex': '2', 'minWidth': '200px'}),

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
                    ], style={'minWidth': '160px'}),

                    html.Div([
                        html.Label('最小共现层数'),
                        dcc.Slider(
                            id='cooc-min-count',
                            min=1, max=10, step=1, value=1,
                            marks={1: '1', 3: '3', 5: '5', 10: '10'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'flex': '2', 'minWidth': '180px'}),

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
                            style={'width': '140px'},
                        ),
                    ], style={'minWidth': '140px'}),
                ], className='analysis-control-bar'),

                # ── 树状图 ──────────────────────────────────────────────────
                html.Div([
                    html.P('簇聚类树状图（按共现相似度）', className='dash-card-title'),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(id='cooc-dendrogram', style={'height': '240px'}),
                    ),
                ], style=_CARD),

                # ── 热力图 + 统计并排 ────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.P('共现矩阵热力图', className='dash-card-title'),
                        dcc.Loading(
                            type='default',
                            children=dcc.Graph(id='cooc-heatmap', style={'height': '460px'}),
                        ),
                    ], style={**_CARD, 'flex': '3', 'minWidth': '340px', 'marginBottom': '0'}),

                    html.Div([
                        html.P('共现统计', className='dash-card-title'),
                        html.Div(id='cooc-stats'),
                    ], style={**_CARD, 'flex': '1', 'minWidth': '190px',
                               'backgroundColor': '#f8fafc', 'marginBottom': '0'}),
                ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'flex-start'}),

            ], style={'padding': '14px'}),
        ],
    )
