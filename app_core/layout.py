import dash
from dash import dcc, html
from app_core.components.modal import build_modal
from app_core.tabs.scatter import build_scatter_tab
from app_core.tabs.heatmap import build_heatmap_tab
from app_core.tabs.similarity import build_similarity_tab
from app_core.tabs.cluster_size import build_cluster_size_tab
from app_core.tabs.cluster_quality import build_cluster_quality_tab
from app_core.tabs.category_breakdown import build_category_breakdown_tab
from app_core.tabs.cluster_analysis import build_cluster_analysis_tab
from app_core.tabs.representatives import build_representatives_tab


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
        dcc.Tabs(
            id='visualization-tabs',
            value='representatives',
            children=[
                build_representatives_tab(),
                build_scatter_tab(
                    fig=fig,
                    clusters=clusters,
                    init_unit_options=init_unit_options,
                    init_part_options=init_part_options,
                    init_type_options=init_type_options,
                    algorithm_options=algorithm_options,
                ),
                build_heatmap_tab(),
                build_similarity_tab(),
                build_cluster_size_tab(),
                build_cluster_quality_tab(),
                build_category_breakdown_tab(),
                build_cluster_analysis_tab(),
            ],
        ),
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
        # Keep only lightweight metadata in the client store; full df is cached server-side
        dcc.Store(id='data-store', data={
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
        dcc.Store(id='rep-last-view-click', data={'cluster': None, 'count': 0}),
        dcc.Store(id='sample-cluster-mapping', data=df.set_index('sample_id')[cluster_col].to_dict()),
        build_modal()
    ], style={'margin': '8px', 'padding': '0'})

