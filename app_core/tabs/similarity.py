"""簇相似度矩阵标签页布局定义。"""

from dash import dcc, html


def build_similarity_tab():
    """构建相似度矩阵页面。

    包含度量类型、重排/标注选项、最近邻数量与矩阵图输出区域。
    """
    return dcc.Tab(
        label='聚类相似度矩阵',
        value='similarity',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面展示簇与簇之间的相似度矩阵，帮助识别哪些簇在特征空间中彼此接近。可选择余弦相似度或欧氏距离度量，启用层次重排可将相似的簇聚集在一起。下方列出每个簇的最近邻簇，便于发现潜在的簇合并机会或理解簇间关系。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#ede7f6', 'border': '1px solid #b39ddb',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                html.Div([
                    html.Label('矩阵类型:'),
                    dcc.RadioItems(
                        id='similarity-metric',
                        options=[
                            {'label': '相似度 (余弦)', 'value': 'cosine'},
                            {'label': '距离 (欧氏)', 'value': 'euclidean'},
                        ],
                        value='cosine',
                        labelStyle={'marginRight': '12px'},
                    ),
                    dcc.Checklist(
                        id='similarity-options',
                        options=[
                            {'label': '层次重排', 'value': 'reorder'},
                            {'label': '显示数值', 'value': 'annotate'},
                        ],
                        value=[],
                        style={'marginTop': '4px'},
                    ),
                    html.Div([
                        html.Label('最近邻簇数量'),
                        dcc.Slider(
                            id='similarity-neighbor-k',
                            min=1,
                            max=10,
                            step=1,
                            value=3,
                            marks={1: '1', 3: '3', 5: '5', 7: '7', 10: '10'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'marginTop': '6px'}),
                ], style={'marginBottom': '8px'}),
                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='similarity-graph', style={'height': 'calc(100vh - 240px)', 'width': '100%'}),
                ),
                html.Div([
                    html.Span('最近邻簇', style={
                        'fontSize': '13px', 'fontWeight': '700', 'color': '#2c3e50',
                    }),
                    html.Span('每个簇与其最相似的 K 个簇，绿色表示高度相似', style={
                        'fontSize': '11px', 'color': '#888', 'marginLeft': '8px',
                    }),
                ], style={'marginTop': '10px', 'marginBottom': '6px', 'display': 'flex', 'alignItems': 'baseline'}),
                html.Div(id='nearest-cluster-list', style={
                    'padding': '10px 12px',
                    'border': '1px solid #e4e8ef',
                    'borderRadius': '10px',
                    'backgroundColor': '#f8fafc',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.04)',
                }),
            ], style={'marginTop': '12px'}),
        ],
    )
