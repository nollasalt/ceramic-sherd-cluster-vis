"""应用主布局构建模块。"""

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
from app_core.tabs.help import build_help_tab
from app_core.tabs.stratigraphy import build_stratigraphy_tab
from app_core.tabs.cooccurrence import build_cooccurrence_tab
from app_core.tabs.borderline import build_borderline_tab


def _group_tab(value, label):
    """创建不可点击的侧边栏分组标题（通过 CSS pointer-events: none 禁用）。"""
    return dcc.Tab(label=label, value=value, className='nav-group-label', children=[])


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
    """构建 Dash 主界面布局。

    参数包含初始图形、筛选选项、聚类元数据及缓存字段，返回完整页面组件树。
    """
    return html.Div([
        dcc.Location(id='url', refresh=True),

        # ── 页面标题 ───────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span('🏺', style={'fontSize': '22px', 'marginRight': '8px'}),
                html.Span('陶片聚类交互可视化', style={
                    'fontSize': '18px', 'fontWeight': '700', 'color': '#1a1a2e', 'verticalAlign': 'middle',
                }),
                html.Span('v1.2', style={
                    'fontSize': '11px', 'color': '#aaa', 'marginLeft': '8px',
                    'verticalAlign': 'middle', 'fontWeight': '400',
                }),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], style={'padding': '10px 4px 6px 4px'}),

        # ── 顶部控制栏 ─────────────────────────────────────────────────────────
        html.Div([
            # 组 1：聚类数量
            html.Div([
                html.Span('聚类数量 K', className='control-label-inline'),
                dcc.Input(
                    id='n-clusters-input', type='number', value=20, min=2, step=1,
                    style={
                        'width': '68px', 'padding': '5px 8px',
                        'border': '1px solid #d0d8e8', 'borderRadius': '6px',
                        'fontSize': '13px', 'color': '#222',
                    },
                ),
            ], className='top-control-group'),

            # 组 2：聚类算法
            html.Div([
                html.Span('算法', className='control-label-inline'),
                dcc.Dropdown(
                    id='cluster-algorithm-selector',
                    options=[
                        {'label': 'K-Means', 'value': 'kmeans'},
                        {'label': '层次聚类 (Ward)', 'value': 'agglomerative-ward'},
                        {'label': '谱聚类 (Spectral)', 'value': 'spectral-kmeans'},
                        {'label': 'Leiden (kNN)', 'value': 'leiden'},
                    ],
                    value='kmeans',
                    clearable=False,
                    style={'width': '170px'},
                ),
            ], className='top-control-group'),

            # 组 3：聚类模式
            html.Div([
                html.Span('模式', className='control-label-inline'),
                dcc.Dropdown(
                    id='cluster-mode-selector',
                    options=[
                        {'label': '融合 (正反面)', 'value': 'merged'},
                        {'label': '仅外部 (exterior)', 'value': 'exterior'},
                        {'label': '仅内部 (interior)', 'value': 'interior'},
                    ],
                    value=initial_cluster_mode,
                    clearable=False,
                    style={'width': '148px'},
                ),
            ], className='top-control-group'),

            # 组 4：执行按钮 + 状态
            html.Div([
                html.Button('重新聚类', id='recluster-button', n_clicks=0, className='btn-primary'),
                dcc.Loading(
                    id='recluster-loading',
                    type='circle',
                    children=[html.Span(id='recluster-status', style={'fontSize': '12px', 'color': '#666'})],
                    style={'display': 'inline-flex', 'alignItems': 'center'},
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'paddingLeft': '4px'}),
        ], className='top-control-bar'),

        dcc.Tabs(
            id='visualization-tabs',
            value='representatives',
            vertical=True,
            parent_className='sidebar-nav',
            children=[
                # ── 总览 ──────────────────────────────────────
                _group_tab('group-overview', '总览'),
                build_help_tab(),
                build_representatives_tab(),
                build_borderline_tab(),
                build_scatter_tab(
                    fig=fig,
                    clusters=clusters,
                    init_unit_options=init_unit_options,
                    init_part_options=init_part_options,
                    init_type_options=init_type_options,
                    algorithm_options=algorithm_options,
                ),
                # ── 质量评估 ───────────────────────────────────
                _group_tab('group-quality', '质量评估'),
                build_cluster_size_tab(),
                build_cluster_quality_tab(),
                build_heatmap_tab(),
                build_similarity_tab(),
                # ── 构成分析 ───────────────────────────────────
                _group_tab('group-analysis', '构成分析'),
                build_category_breakdown_tab(),
                build_cluster_analysis_tab(),
                # ── 地层分析 ───────────────────────────────────
                _group_tab('group-stratigraphy', '地层分析'),
                build_stratigraphy_tab(),
                build_cooccurrence_tab(),
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

