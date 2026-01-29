import dash
from dash import dcc, html


def build_layout(
    fig,
    clusters,
    init_unit_options,
    init_part_options,
    init_type_options,
    algorithm_options,
    initial_cluster_mode,
    cluster_metadata,
    df,
    feature_cols,
    raw_feature_cols,
    cluster_col,
    image_col,
):
    """Construct the main Dash layout."""
    return html.Div([
        dcc.Location(id='url', refresh=True),
        html.Div([
            html.H3('陶片聚类交互可视化 v1.2', style={'display': 'inline-block', 'marginRight': '20px'}),
            html.Div()
        ]),
        html.Div([
            html.Div([
                html.Label('聚类数量 (K):'),
                dcc.Input(id='n-clusters-input', type='number', value=20, min=2, step=1,
                          style={'width': '80px', 'marginLeft': '10px', 'marginRight': '20px'}),
            ], style={'display': 'inline-block', 'marginRight': '20px'}),
            html.Div([
                html.Label('聚类算法:'),
                dcc.Dropdown(
                    id='cluster-algorithm-selector',
                    options=[
                        {'label': 'K-Means', 'value': 'kmeans'},
                        {'label': '层次聚类 (Ward)', 'value': 'agglomerative-ward'},
                        {'label': '谱聚类 (Spectral)', 'value': 'spectral-kmeans'},
                        {'label': 'Leiden (Mutual kNN)', 'value': 'leiden'}
                    ],
                    value='kmeans',
                    clearable=False,
                    style={'width': '180px', 'marginLeft': '10px'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '20px', 'verticalAlign': 'middle'}),
            html.Div([
                html.Label('聚类模式:'),
                dcc.Dropdown(
                    id='cluster-mode-selector',
                    options=[
                        {'label': '融合 (正反面)', 'value': 'merged'},
                        {'label': '仅外部 (exterior)', 'value': 'exterior'},
                        {'label': '仅内部 (interior)', 'value': 'interior'}
                    ],
                    value=initial_cluster_mode,
                    clearable=False,
                    style={'width': '150px', 'marginLeft': '10px'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '20px', 'verticalAlign': 'middle'}),
            html.Button('重新聚类', id='recluster-button', n_clicks=0,
                       style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
                              'padding': '8px 16px', 'cursor': 'pointer', 'borderRadius': '4px'}),
            dcc.Loading(
                id='recluster-loading',
                type='circle',
                children=[html.Span(id='recluster-status', style={'marginLeft': '15px', 'color': '#666'})],
                style={'display': 'inline-block', 'marginLeft': '15px'}
            ),
        ], style={'padding': '10px', 'backgroundColor': '#f5f5f5', 'borderRadius': '4px', 'marginBottom': '10px'}),
        dcc.Tabs(id='visualization-tabs', value='scatter', children=[
            dcc.Tab(label='散点图', value='scatter', children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Label('降维算法:'),
                            dcc.Dropdown(id='algorithm-selector', options=algorithm_options, value='umap'),
                        ], style={'width': '200px'}),
                        html.Div([
                            html.Label('可视化维度:'),
                            dcc.RadioItems(id='dimension-selector', options=[{'label': '2D', 'value': 2}, {'label': '3D', 'value': 3}], value=2),
                        ], style={'width': '180px'}),
                        html.Div([
                            html.Label('3D z轴:'),
                            dcc.Dropdown(id='z-axis-selector',
                                         options=[{'label': '降维结果', 'value': 'dimension'}, {'label': 'unit_C', 'value': 'unit_C'}],
                                         value='dimension'),
                        ], style={'width': '220px'}),
                    ], style={'marginBottom': '8px', 'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px 12px', 'alignItems': 'flex-end'}),
                    html.Div([
                        html.Div([
                            html.Label('筛选簇:'),
                            dcc.Dropdown(id='cluster-filter', options=[{'label': str(c), 'value': c} for c in clusters], multi=True, placeholder='选择一个或多个簇（留空显示所有）'),
                        ], style={'width': '240px'}),
                        html.Div([
                            html.Label('筛选单位:'),
                            dcc.Dropdown(id='unit-filter', options=init_unit_options, multi=True, placeholder='选择单位（留空显示所有）'),
                        ], style={'width': '200px'}),
                        html.Div([
                            html.Label('筛选部位:'),
                            dcc.Dropdown(id='part-filter', options=init_part_options, multi=True, placeholder='选择部位（留空显示所有）'),
                        ], style={'width': '200px'}),
                        html.Div([
                            html.Label('筛选类型:'),
                            dcc.Dropdown(id='type-filter', options=init_type_options, multi=True, placeholder='选择类型（留空显示所有）'),
                        ], style={'width': '200px'}),
                    ], style={'marginBottom': '6px', 'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px 12px', 'alignItems': 'flex-end'}),
                ], style={'marginBottom': '16px'}),
                html.Div([
                    html.Div([
                        dcc.Loading(
                            id='plot-loading',
                            type='default',
                            children=[
                                dcc.Graph(
                                    id='tsne-plot',
                                    figure=fig,
                                    style={'height': 'calc(100vh - 380px)', 'width': '100%'},
                                    clear_on_unhover=True
                                )
                            ]
                        ),
                        html.Div(id='plot-loading-status', style={'textAlign': 'center', 'marginTop': '8px', 'color': '#666'})
                    ], style={'flex': '3 1 0%'}),
                    html.Div([
                        html.Div([
                            html.Button('Prev', id='cluster-prev', n_clicks=0),
                            html.Button('Next', id='cluster-next', n_clicks=0),
                            html.Span(id='page-indicator', style={'marginLeft': '8px'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '8px'}),
                        html.Div(id='cluster-panel', style={'height': 'calc(100vh - 420px)', 'overflowY': 'auto'}),
                        dcc.Store(id='cluster-images-store'),
                        dcc.Store(id='cluster-page', data=1)
                    ], style={'width': '320px', 'flex': '1 0 320px', 'borderLeft': '1px solid #ddd', 'padding': '8px', 'boxSizing': 'border-box'})
                    ], style={'display': 'flex', 'gap': '4px', 'height': 'calc(100vh - 260px)'}),
                html.Div([
                    html.Button('添加到比较', id='compare-add', style={'width': '120px', 'backgroundColor': '#0066cc', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'padding': '6px 10px'}),
                    html.Button('清空比较', id='compare-clear', style={'width': '120px'}),
                    html.A('进入比较视图', href='#compare-section', target='_blank', style={'marginLeft': '10px', 'textDecoration': 'none', 'color': '#0066cc', 'fontWeight': '600'}),
                    html.Span('（先点击散点图选中样本，再添加到比较）', style={'marginLeft': '10px', 'color': '#666'})
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginTop': '8px', 'marginBottom': '8px'})
            ]),
            dcc.Tab(label='聚类特征热力图', value='heatmap', children=[
                html.Div([
                    html.Div(id='heatmap-container', style={'height': 'calc(100vh - 180px)', 'width': '100%'})
                ], style={'marginTop': '12px'})
            ]),
            dcc.Tab(label='聚类相似度矩阵', value='similarity', children=[
                html.Div([
                    html.Div([
                        html.Label('矩阵类型:'),
                        dcc.RadioItems(
                            id='similarity-metric',
                            options=[
                                {'label': '相似度 (余弦)', 'value': 'cosine'},
                                {'label': '距离 (欧氏)', 'value': 'euclidean'}
                            ],
                            value='cosine',
                            labelStyle={'marginRight': '12px'}
                        ),
                        dcc.Checklist(
                            id='similarity-options',
                            options=[
                                {'label': '层次重排', 'value': 'reorder'},
                                {'label': '显示数值', 'value': 'annotate'}
                            ],
                            value=[],
                            style={'marginTop': '4px'}
                        ),
                        html.Div([
                            html.Label('最近邻簇数量'),
                            dcc.Slider(
                                id='similarity-neighbor-k', min=1, max=10, step=1, value=3,
                                marks={1: '1', 3: '3', 5: '5', 7: '7', 10: '10'},
                                tooltip={'placement': 'bottom', 'always_visible': False}
                            )
                        ], style={'marginTop': '6px'})
                    ], style={'marginBottom': '8px'}),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(id='similarity-graph', style={'height': 'calc(100vh - 240px)', 'width': '100%'})
                    ),
                    html.Div(id='nearest-cluster-list', style={'marginTop': '8px', 'fontSize': '13px', 'color': '#333'})
                ], style={'marginTop': '12px'})
            ]),
            dcc.Tab(label='簇规模分布', value='cluster-size', children=[
                html.Div([
                    dcc.Graph(
                        id='cluster-size-graph',
                        style={'height': 'calc(100vh - 200px)'}
                    )
                ], style={'marginTop': '12px'})
            ]),
            dcc.Tab(label='聚类质量', value='cluster-quality', children=[
                html.Div([
                    html.Div(id='cluster-quality-cards', style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),
                    dcc.Graph(id='cluster-quality-bars', style={'height': '380px', 'width': '100%', 'marginBottom': '8px'}),
                    html.Div(id='cluster-quality-detail', style={'fontSize': '13px', 'color': '#333', 'padding': '0 4px'}),
                    html.Div('颜色指示: 绿=清晰，黄=需关注，红=混杂/易粘连', style={'fontSize': '12px', 'color': '#666', 'marginTop': '4px', 'padding': '0 4px'})
                ], style={'marginTop': '12px', 'padding': '0 8px'})
            ]),
            dcc.Tab(label='类别构成', value='category-breakdown', children=[
                html.Div([
                    html.Div([
                        html.Label('类别字段:'),
                        dcc.Dropdown(
                            id='category-field-selector',
                            options=[
                                {'label': '部位 (part_C)', 'value': 'part_C'},
                                {'label': '类型 (type_C)', 'value': 'type_C'},
                                {'label': '单位 (unit_C)', 'value': 'unit_C'}
                            ],
                            value='part_C',
                            clearable=False,
                            style={'width': '220px'}
                        )
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.Label('横轴'),
                        dcc.RadioItems(
                            id='category-x-axis',
                            options=[
                                {'label': '按簇', 'value': 'cluster'},
                                {'label': '按单位 (unit_C)', 'value': 'unit_C'}
                            ],
                            value='cluster',
                            labelStyle={'marginRight': '12px'}
                        )
                    ], style={'marginBottom': '8px'}),
                    dcc.Graph(
                        id='category-breakdown-graph',
                        style={'height': 'calc(100vh - 230px)'}
                    )
                ], style={'marginTop': '12px', 'padding': '0 8px'})
            ]),
            dcc.Tab(label='簇分析', value='cluster-analysis', children=[
                html.Div([
                    html.Div([
                        html.Label('选择簇'),
                        dcc.Dropdown(id='analysis-cluster-selector', placeholder='选择一个簇查看特征差异'),
                    ], style={'width': '200px', 'marginRight': '12px'}),
                    html.Div([
                        html.Label('Top-K 特征'),
                        dcc.Slider(
                            id='feature-topk-slider', min=3, max=20, step=1, value=8,
                            marks={3: '3', 5: '5', 8: '8', 12: '12', 16: '16', 20: '20'},
                            tooltip={'placement': 'bottom', 'always_visible': False}
                        )
                    ], style={'flex': '1', 'minWidth': '240px'}),
                    html.Div([
                        html.Label('差异度量'),
                        dcc.RadioItems(
                            id='feature-diff-mode',
                            options=[
                                {'label': '均值差', 'value': 'mean'},
                                {'label': 'z-score', 'value': 'zscore'}
                            ],
                            value='mean',
                            labelStyle={'marginRight': '12px'}
                        )
                    ], style={'width': '200px'})
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '12px', 'padding': '0 8px'}),
                html.Div([
                    html.Div(id='cluster-quality-table', style={'flex': '1', 'minWidth': '320px', 'padding': '0 8px'}),
                    html.Div(
                        dcc.Loading(id='feature-diff-loading', type='default', children=dcc.Graph(id='feature-diff-graph', style={'height': '420px'})),
                        style={'flex': '1', 'minWidth': '320px', 'padding': '0 8px'}
                    )
                ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'flex-start'})
            ]),
            dcc.Tab(label='代表样本', value='representatives', children=[
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
                        )
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.Label('代表样本选择'),
                        dcc.RadioItems(
                            id='rep-strategy',
                            options=[
                                {'label': '最近中心', 'value': 'center'},
                                {'label': '随机', 'value': 'random'}
                            ],
                            value='center',
                            labelStyle={'marginRight': '12px'}
                        )
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.Label('每簇离群样本数'),
                        dcc.Slider(
                            id='outlier-count',
                            min=1,
                            max=5,
                            step=1,
                            value=2,
                            marks={1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
                            tooltip={'placement': 'bottom', 'always_visible': False}
                        )
                    ], style={'marginBottom': '8px'}),
                    dcc.Loading(
                        id='rep-grid-loading',
                        type='default',
                        children=html.Div(id='representative-grid', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px'})
                    ),
                    html.Div(id='outlier-list', style={'marginTop': '12px', 'fontSize': '13px', 'color': '#333'})
                ], style={'marginTop': '12px', 'padding': '0 8px'})
            ]),
        ]),
        html.Div(id='sample-panel', style={'marginTop': '12px', 'minHeight': '220px', 'borderTop': '1px solid #ddd', 'paddingTop': '8px'}),
        html.Div(id='selected-meta'),
        html.Div([
            html.Div([
                html.H4('手动比较视图', style={'margin': 0}),
                html.Div([
                    html.Button('清空比较', id='compare-clear-bottom', style={'width': '120px'})
                ], style={'display': 'flex', 'gap': '8px', 'marginTop': '6px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '4px', 'marginBottom': '8px'}),
            html.Div([
                html.Div([
                    html.Label('卡片尺寸'),
                    dcc.Slider(id='compare-size', min=140, max=320, step=20, value=220,
                               marks={140:'140', 200:'200', 260:'260', 320:'320'}, tooltip={'placement':'bottom','always_visible':False}),
                ], style={'flex': '1', 'minWidth': '200px', 'marginRight': '12px'}),
                html.Div([
                    html.Label('布局模式'),
                    dcc.RadioItems(
                        id='compare-layout',
                        options=[{'label': '网格换行', 'value': 'grid'}, {'label': '横向滚动', 'value': 'row'}],
                        value='grid',
                        labelStyle={'marginRight': '12px'}
                    )
                ], style={'width': '260px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px', 'marginBottom': '8px'}),
            html.Div(id='compare-panel', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px', 'padding': '8px', 'border': '1px dashed #ddd', 'minHeight': '120px'})
        ], id='compare-section', style={'borderTop': '1px solid #eee', 'paddingTop': '8px', 'marginTop': '8px'}),
        dcc.Store(id='data-store', data={
            'df': df.to_json(orient='split'),
            'feature_cols': feature_cols,
            'raw_feature_cols': raw_feature_cols,
            'cluster_col': cluster_col,
            'image_col': image_col,
            'cluster_mode': initial_cluster_mode,
            'params': {},
            'version': 0
        }),
        dcc.Store(id='reload-trigger', data=0),
        dcc.Store(id='cluster-metadata-store', data=cluster_metadata),
        dcc.Store(id='compare-selected-store', data=[]),
        dcc.Store(id='last-selected-store', data={}),
        dcc.Store(id='hover-state', data={'hovered_cluster': None}),
        dcc.Store(id='sample-cluster-mapping', data=df.set_index('sample_id')[cluster_col].to_dict()),
        html.Div(id='image-modal', style={
            'display': 'none',
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0, 0, 0, 0.9)',
            'zIndex': 9999,
            'cursor': 'pointer'
        }, children=[
            html.Div(style={
                'position': 'absolute',
                'top': '20px',
                'right': '20px',
                'zIndex': 10000,
                'display': 'flex',
                'gap': '10px'
            }, children=[
                html.Button('放大', id='zoom-in-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('缩小', id='zoom-out-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('左转', id='rotate-left-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('右转', id='rotate-right-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('重置', id='reset-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('关闭', id='close-modal-btn', style={
                    'backgroundColor': 'rgba(255, 0, 0, 0.8)',
                    'color': 'white',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                })
            ]),
            html.Div(style={
                'position': 'absolute',
                'top': '0',
                'left': '0',
                'width': '100%',
                'height': '100%',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'overflow': 'visible'
            }, children=[
                html.Img(id='modal-image', style={
                    'maxWidth': '80vw',
                    'maxHeight': '80vh',
                    'border': '2px solid white',
                    'borderRadius': '4px',
                    'cursor': 'move',
                    'transition': 'transform 0.2s ease',
                    'transformOrigin': 'center center',
                    'position': 'relative',
                    'zIndex': 1
                }),
            ]),
            html.Div(style={
                'position': 'absolute',
                'bottom': '20px',
                'left': '50%',
                'transform': 'translateX(-50%)',
                'color': 'white',
                'fontSize': '14px',
                'textAlign': 'center'
            }, children=[
                html.Div('拖拽移动图片 | 滚轮缩放 | ESC键关闭', style={'opacity': '0.8'})
            ])
        ]),
        html.Div(id='image-click-trigger', style={'display': 'none'}),
        html.Div(id='modal-close-trigger', style={'display': 'none'}),
        dcc.Input(id='image-path-input', type='text', value='', style={'display': 'none'})
    ], style={'margin': '8px', 'padding': '0'})
