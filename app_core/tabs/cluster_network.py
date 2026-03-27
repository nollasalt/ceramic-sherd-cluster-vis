"""簇间距离网络图标签页布局（质心 kNN 网络）。"""

from dash import dcc, html

_CARD = {'padding': '14px 16px', 'border': '1px solid #e4e8ef',
         'borderRadius': '10px', 'backgroundColor': '#fff',
         'boxShadow': '0 1px 4px rgba(0,0,0,0.05)', 'marginBottom': '12px'}
_CARD_TITLE = {'fontSize': '13px', 'fontWeight': '700', 'color': '#2c3e50',
               'margin': '0 0 10px 0', 'letterSpacing': '0.02em'}


def build_cluster_network_tab():
    """构建簇间距离网络图页面。

    节点 = 簇，边粗细 = 质心相似度，布局由质心 PCA/UMAP 2D 投影决定。
    相近的簇会视觉上聚集，揭示层次结构与"近亲"关系。
    """
    return dcc.Tab(
        label='簇距离网络',
        value='cluster-network',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面以网络图形式展示簇间关系，节点代表簇，边的粗细表示质心相似度。节点位置由PCA或UMAP降维决定，相似的簇会在视觉上聚集。节点大小反映簇的样本数，颜色可按主导器类、器部或簇编号着色，帮助发现簇的层次结构和"近亲"关系。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#e8eaf6', 'border': '1px solid #9fa8da',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                # ── 控制区 ─────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('节点颜色'),
                        dcc.RadioItems(
                            id='net-node-color',
                            options=[
                                {'label': '主导器类 (type_C)', 'value': 'type'},
                                {'label': '主导器部 (part_C)', 'value': 'part'},
                                {'label': '簇编号', 'value': 'cluster'},
                            ],
                            value='type',
                            labelStyle={'display': 'block', 'marginBottom': '3px'},
                        ),
                    ], style={'minWidth': '160px'}),

                    html.Div([
                        html.Label('距离度量'),
                        dcc.RadioItems(
                            id='net-metric',
                            options=[
                                {'label': '余弦距离', 'value': 'cosine'},
                                {'label': '欧氏距离', 'value': 'euclidean'},
                            ],
                            value='cosine',
                            labelStyle={'display': 'block', 'marginBottom': '3px'},
                        ),
                    ], style={'minWidth': '120px'}),

                    html.Div([
                        html.Label('布局算法'),
                        dcc.RadioItems(
                            id='net-layout',
                            options=[
                                {'label': 'PCA（快速）', 'value': 'pca'},
                                {'label': 'UMAP（慢）', 'value': 'umap'},
                            ],
                            value='pca',
                            labelStyle={'display': 'block', 'marginBottom': '3px'},
                        ),
                    ], style={'minWidth': '130px'}),

                    html.Div([
                        html.Div([
                            html.Label('每节点最近邻 K'),
                            dcc.Slider(
                                id='net-knn',
                                min=1, max=10, step=1, value=3,
                                marks={1: '1', 3: '3', 5: '5', 10: '10'},
                                tooltip={'placement': 'bottom', 'always_visible': False},
                            ),
                        ], style={'marginBottom': '10px'}),
                        html.Div([
                            html.Label('最大显示簇数'),
                            dcc.Slider(
                                id='net-max-clusters',
                                min=10, max=200, step=10, value=100,
                                marks={10: '10', 50: '50', 100: '100', 200: '全'},
                                tooltip={'placement': 'bottom', 'always_visible': False},
                            ),
                        ]),
                    ], style={'flex': '3', 'minWidth': '220px'}),

                ], className='analysis-control-bar'),

                # ── 网络图 ──────────────────────────────────────────────────
                html.Div([
                    html.P('簇质心 kNN 距离网络', style=_CARD_TITLE),
                    html.P(
                        '节点大小 = 簇样本数；边粗细 = 质心相似度；'
                        '节点颜色 = 主导标签。位置由质心 PCA/UMAP 投影决定。',
                        style={'fontSize': '11px', 'color': '#888', 'margin': '-4px 0 8px'},
                    ),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(
                            id='cluster-network-graph',
                            style={'height': 'calc(100vh - 360px)'},
                            config={'scrollZoom': True},
                        ),
                    ),
                ], style=_CARD),

                # ── 近邻列表 ────────────────────────────────────────────────
                html.Div([
                    html.P('点击节点查看近邻', style=_CARD_TITLE),
                    html.Div(id='cluster-network-detail',
                             style={'fontSize': '12px', 'color': '#555'}),
                ], style={**_CARD, 'backgroundColor': '#f8fafc'}),

            ], style={'padding': '14px'}),
        ],
    )
