"""
陶片聚类交互可视化应用
使用 Dash 构建的 Web 应用，支持降维可视化、聚类浏览等功能
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State

# 从数据处理模块导入
from data_processing import (
    load_cluster_metadata,
    detect_columns,
    ensure_sample_ids,
    build_fused_samples,
    ensure_dimensionality_reduction,
    create_cluster_pattern_heatmap,
    create_cluster_similarity_matrix,
    img_to_base64,
    DEFAULT_CSV,
    DEFAULT_IMAGE_ROOT,
)

# 配置常量
CSV = DEFAULT_CSV
IMAGE_ROOT = DEFAULT_IMAGE_ROOT

# 生成足够多的不同颜色用于聚类显示（支持最多50个聚类）
def generate_distinct_colors(n_colors):
    """生成 n 个视觉上不同的颜色"""
    # 组合多个 Plotly 调色板以获得足够多的颜色
    base_colors = (
        px.colors.qualitative.Plotly +      # 10 colors
        px.colors.qualitative.D3 +          # 10 colors
        px.colors.qualitative.G10 +         # 10 colors
        px.colors.qualitative.T10 +         # 10 colors
        px.colors.qualitative.Alphabet      # 26 colors
    )
    # 去重并返回所需数量
    seen = set()
    unique_colors = []
    for c in base_colors:
        if c not in seen:
            seen.add(c)
            unique_colors.append(c)
    return unique_colors[:n_colors]

# 预生成50种不同的颜色
CLUSTER_COLORS = generate_distinct_colors(50)


def create_app(csv=CSV, image_root=IMAGE_ROOT):
    """创建并配置 Dash 应用"""
    image_root = Path(image_root)
    
    df = pd.read_csv(csv)
    cluster_col, image_col = detect_columns(df)
    if cluster_col is None or image_col is None:
        raise RuntimeError('无法识别聚类列或图片列，请检查 CSV')

    df = ensure_sample_ids(df, image_col)

    # 检查是否包含 jd_sherds_info 的字段（在 build_table.py 中已合并）
    has_info = 'sherd_id' in df.columns or 'unit_C' in df.columns
    if has_info:
        matched_count = df['sherd_id'].notna().sum() if 'sherd_id' in df.columns else 0
        print(f"已加载包含 jd_sherds_info 的表格，匹配了 {matched_count}/{len(df)} 条记录")
    else:
        print(f"表格中未包含 jd_sherds_info 字段，请运行 build_table.py 进行合并")

    # detect feature columns (numeric excluding metadata)
    exclude = {cluster_col, image_col, 'image_name', 'sample_id', 'side', 'image_id', 'sherd_id', 
               'unit', 'part', 'type', 'image_side', 'image_id_original', 'unit_C', 'part_C', 'type_C'}
    raw_feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if len(raw_feature_cols) == 0:
        raise RuntimeError('未找到数值特征列')

    # fuse front/back so each sample is represented once (matching k-means input)
    df, feature_cols, dropped_samples = build_fused_samples(df, raw_feature_cols, cluster_col, image_col)
    if len(df) == 0:
        raise RuntimeError('融合正反面后没有可用的数据，请检查输入')
    if dropped_samples:
        print(f"有 {len(dropped_samples)} 个样本因缺少正反面被跳过: {dropped_samples[:10]} ...")

    # 初始使用UMAP降维
    df, initial_reduction_key = ensure_dimensionality_reduction(df, feature_cols, algorithm='umap', n_components=2)

    app = dash.Dash(__name__)

    clusters = sorted(df[cluster_col].unique())

    # 初始化筛选器选项的辅助函数
    def get_filter_options(selected_clusters=None):
        dff = df
        if selected_clusters and len(selected_clusters) > 0:
            dff = df[df[cluster_col].isin(selected_clusters)]
        
        unit_options = [{'label': str(v), 'value': v} for v in sorted(dff['unit_C'].dropna().unique())] if 'unit_C' in dff.columns else []
        part_options = [{'label': str(v), 'value': v} for v in sorted(dff['part_C'].dropna().unique())] if 'part_C' in dff.columns else []
        type_options = [{'label': str(v), 'value': v} for v in sorted(dff['type_C'].dropna().unique())] if 'type_C' in dff.columns else []
        
        return unit_options, part_options, type_options
    
    # 初始化筛选器的默认选项
    init_unit_options, init_part_options, init_type_options = get_filter_options()

    # 准备 hover_data 和 custom_data
    hover_cols = ['sample_id', 'image_name']
    if 'sherd_id' in df.columns:
        hover_cols.append('sherd_id')
    if 'unit_C' in df.columns:
        hover_cols.append('unit_C')
    if 'part_C' in df.columns:
        hover_cols.append('part_C')
    if 'type_C' in df.columns:
        hover_cols.append('type_C')
    
    # include sample_id first for reliable lookup
    custom = ['sample_id', 'image_name']
    if 'paired_images' in df.columns:
        custom.append('paired_images')
    if 'sherd_id' in df.columns:
        custom.append('sherd_id')
    # 初始使用UMAP降维的结果
    initial_algorithm = 'umap'
    fig = px.scatter(df, x=f'{initial_reduction_key}_0', y=f'{initial_reduction_key}_1', 
                     color=df[cluster_col].astype(str), hover_data=hover_cols, custom_data=custom,
                     color_discrete_sequence=CLUSTER_COLORS)

    # 准备降维算法选项
    algorithm_options = [
        {'label': 'PCA', 'value': 'pca'},
        {'label': 't-SNE', 'value': 'tsne'}
    ]
    algorithm_options.append({'label': 'UMAP', 'value': 'umap'})
    
    # 加载聚类元数据
    cluster_metadata = load_cluster_metadata()
    
    # 准备可视化类型选项
    visualization_types = [
        {'label': '降维散点图', 'value': 'scatter'},
        {'label': '聚类特征热力图', 'value': 'heatmap'},
        {'label': '聚类相似度矩阵', 'value': 'similarity'}
    ]
    
    app.layout = html.Div([
        html.H3('陶片聚类交互可视化'),
        
        # 选项卡容器
        dcc.Tabs(id='visualization-tabs', value='scatter', children=[
            # 散点图选项卡
            dcc.Tab(label='散点图', value='scatter', children=[
                html.Div([
                    # 第一行：主要降维参数
                    html.Div([
                        html.Div([
                            html.Label('降维算法:'),
                            dcc.Dropdown(id='algorithm-selector', options=algorithm_options, value='umap'),
                        ], style={'width': '20%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '15px'}),
                        html.Div([
                            html.Label('可视化维度:'),
                            dcc.RadioItems(id='dimension-selector', options=[{'label': '2D', 'value': 2}, {'label': '3D', 'value': 3}], value=2),
                        ], style={'width': '15%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '15px'}),
                        # z轴选择（仅在3D模式下生效）
                        html.Div([
                            html.Label('3D z轴:'),
                            dcc.Dropdown(id='z-axis-selector', 
                                         options=[{'label': '降维结果', 'value': 'dimension'}, {'label': 'unit_C', 'value': 'unit_C'}], 
                                         value='dimension'),
                        ], style={'width': '20%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '15px'}),
                    ], style={'marginBottom': '8px'}),
                    
                    # 第二行：算法参数
                    html.Div([
                        # t-SNE参数
                        html.Div([
                            html.Label('t-SNE困惑度 (perplexity):'),
                            dcc.Slider(id='tsne-perplexity', min=5, max=50, step=1, value=30, marks={5: '5', 25: '25', 50: '50'}),
                        ], style={'width': '32%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '15px'}),
                        # UMAP参数
                        html.Div([
                            html.Label('UMAP邻居数 (n_neighbors):'),
                            dcc.Slider(id='umap-n-neighbors', min=5, max=50, step=1, value=15, marks={5: '5', 25: '25', 50: '50'}),
                        ], style={'width': '32%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '15px'}),
                        html.Div([
                            html.Label('UMAP最小距离 (min_dist):'),
                            dcc.Slider(id='umap-min-dist', min=0.01, max=1.0, step=0.01, value=0.1, marks={0.01: '0.01', 0.5: '0.5', 1.0: '1.0'}),
                        ], style={'width': '32%', 'display': 'inline-block', 'marginBottom': '8px'}),
                    ], style={'marginBottom': '8px'}),
                    
                    # 第三行：筛选条件
                    html.Div([
                        html.Div([
                            html.Label('筛选簇:'),
                            dcc.Dropdown(id='cluster-filter', options=[{'label': str(c), 'value': c} for c in clusters], multi=True, placeholder='选择一个或多个簇（留空显示所有）'),
                        ], style={'width': '32%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '15px'}),
                        html.Div([
                            html.Label('筛选单位:'),
                            dcc.Dropdown(id='unit-filter', options=init_unit_options, multi=True, placeholder='选择单位（留空显示所有）'),
                        ], style={'width': '22%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '15px'}),
                        html.Div([
                            html.Label('筛选部位:'),
                            dcc.Dropdown(id='part-filter', options=init_part_options, multi=True, placeholder='选择部位（留空显示所有）'),
                        ], style={'width': '22%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '15px'}),
                        html.Div([
                            html.Label('筛选类型:'),
                            dcc.Dropdown(id='type-filter', options=init_type_options, multi=True, placeholder='选择类型（留空显示所有）'),
                        ], style={'width': '22%', 'display': 'inline-block', 'marginBottom': '8px'}),
                    ], style={'marginBottom': '8px'}),
                ], style={'marginBottom': '16px'}),
                
                # main area: left = plot (large), right = controls + scrollable cluster thumbnails
            html.Div([
                html.Div([
                    dcc.Loading(
                        id='plot-loading',
                        type='default',
                        children=[
                            dcc.Graph(
                                id='tsne-plot', 
                                figure=fig, 
                                style={'height': 'calc(100vh - 240px)', 'width': '100%'},
                                clear_on_unhover=True
                            )
                        ]
                    ),
                    html.Div(id='plot-loading-status', style={'textAlign': 'center', 'marginTop': '8px', 'color': '#666'})
                ], style={'flex': '1 1 auto'}),
                html.Div([
                    html.Div([
                        html.Button('Prev', id='cluster-prev', n_clicks=0),
                        html.Button('Next', id='cluster-next', n_clicks=0),
                        html.Span(id='page-indicator', style={'marginLeft': '8px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '8px'}),
                    html.Div(id='cluster-panel', style={'height': 'calc(100vh - 300px)', 'overflowY': 'auto'}) ,
                    dcc.Store(id='cluster-images-store'),
                    dcc.Store(id='cluster-page', data=1)
                ], style={'width': '320px', 'borderLeft': '1px solid #ddd', 'padding': '8px', 'boxSizing': 'border-box'})
            ], style={'display': 'flex', 'gap': '12px', 'height': 'calc(100vh - 180px)'})
            ]),
            
            # 聚类特征热力图选项卡
            dcc.Tab(label='聚类特征热力图', value='heatmap', children=[
                html.Div([
                    html.Div(id='heatmap-container', style={'height': 'calc(100vh - 180px)', 'width': '100%'})
                ], style={'marginTop': '12px'})
            ]),
            
            # 聚类相似度矩阵选项卡
            dcc.Tab(label='聚类相似度矩阵', value='similarity', children=[
                html.Div([
                    html.Div(id='similarity-container', style={'height': 'calc(100vh - 180px)', 'width': '100%'})
                ], style={'marginTop': '12px'})
            ]),
        ]),
        
        # bottom area: sample (front/back) gallery
        html.Div(id='sample-panel', style={'marginTop': '12px', 'minHeight': '220px', 'borderTop': '1px solid #ddd', 'paddingTop': '8px'}),
        html.Div(id='selected-meta'),
        
        # Store components for data sharing between callbacks
        dcc.Store(id='data-store', data={
            'df': df.to_json(orient='split'),
            'feature_cols': feature_cols,
            'cluster_col': cluster_col,
            'image_col': image_col
        }),
        # Store for cluster metadata
        dcc.Store(id='cluster-metadata-store', data=cluster_metadata),
        # Store for hover state
        dcc.Store(id='hover-state', data={'hovered_cluster': None}),
        # Store for sample_id to cluster_id mapping (for fast hover lookup)
        dcc.Store(id='sample-cluster-mapping', data=df.set_index('sample_id')[cluster_col].to_dict())
    ], style={'margin': '8px', 'padding': '0'})


    # 动态更新筛选器选项的回调
    @app.callback(
        [Output('unit-filter', 'options'),
         Output('part-filter', 'options'),
         Output('type-filter', 'options')],
        Input('cluster-filter', 'value')
    )
    def update_filter_options(selected_clusters):
        return get_filter_options(selected_clusters)

    # 悬停状态回调（优化版，使用缓存映射）
    @app.callback(
        Output('hover-state', 'data'),
        [Input('tsne-plot', 'hoverData')],
        [State('sample-cluster-mapping', 'data')]
    )
    def update_hover_state(hoverData, sample_cluster_mapping):
        """更新悬停状态，使用缓存映射快速查找聚类ID"""
        # 如果没有悬停数据，清除悬停状态
        if not hoverData or not sample_cluster_mapping:
            return {'hovered_cluster': None}
        
        try:
            # 从 hoverData 获取样本ID
            hover_point = hoverData['points'][0]
            
            if 'customdata' in hover_point and hover_point['customdata']:
                sample_id = hover_point['customdata'][0]
                # 使用缓存映射快速查找聚类ID
                cluster_id = sample_cluster_mapping.get(sample_id)
                return {'hovered_cluster': cluster_id}
            
            return {'hovered_cluster': None}
        except Exception:
            return {'hovered_cluster': None}

    # 客户端回调：快速处理悬停透明度效果
    app.clientside_callback(
        """
        function(hover_state, figure) {
            if (!figure) return figure;
            
            const hovered_cluster = hover_state ? hover_state.hovered_cluster : null;
            const new_figure = JSON.parse(JSON.stringify(figure));
            
            if (new_figure.data) {
                new_figure.data.forEach(trace => {
                    if (hovered_cluster !== null && hovered_cluster !== undefined) {
                        // 如果有悬停聚类，突出显示该聚类
                        const cluster_match = trace.name === String(hovered_cluster);
                        trace.opacity = cluster_match ? 1.0 : 0.2;
                    } else {
                        // 如果没有悬停，恢复默认透明度
                        trace.opacity = 0.85;
                    }
                });
            }
            
            return new_figure;
        }
        """,
        Output('tsne-plot', 'figure', allow_duplicate=True),
        [Input('hover-state', 'data')],
        [State('tsne-plot', 'figure')],
        prevent_initial_call=True
    )

    @app.callback(
        [Output('tsne-plot', 'figure'),
         Output('data-store', 'data')],
        [Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value'),
         Input('algorithm-selector', 'value'),
         Input('dimension-selector', 'value'),
         Input('z-axis-selector', 'value'),
         Input('tsne-perplexity', 'value'),
         Input('umap-n-neighbors', 'value'),
         Input('umap-min-dist', 'value')],
        State('data-store', 'data')
    )
    def update_plot(selected_clusters, selected_units, selected_parts, selected_types, 
                   selected_algorithm, selected_dimension, 
                   z_axis='dimension',
                   tsne_perplexity=30, umap_n_neighbors=15, umap_min_dist=0.1,
                   data_store=None):
        # 从data-store获取数据
        if data_store is None:
            raise ValueError("Data store is empty")
        
        # 确保selected_algorithm不是None
        if selected_algorithm is None:
            selected_algorithm = 'umap'
        
        # 解析数据
        df = pd.read_json(StringIO(data_store['df']), orient='split')
        feature_cols = data_store['feature_cols']
        cluster_col = data_store['cluster_col']
        image_col = data_store['image_col']
        
        # 使用新的缓存机制计算降维结果
        if selected_algorithm == 'tsne':
            df, reduction_key = ensure_dimensionality_reduction(df, feature_cols, 
                                                           algorithm=selected_algorithm, 
                                                           n_components=selected_dimension,
                                                           perplexity=tsne_perplexity)
        elif selected_algorithm == 'umap':
            df, reduction_key = ensure_dimensionality_reduction(df, feature_cols, 
                                                           algorithm=selected_algorithm, 
                                                           n_components=selected_dimension,
                                                           n_neighbors=umap_n_neighbors,
                                                           min_dist=umap_min_dist)
        else:
            df, reduction_key = ensure_dimensionality_reduction(df, feature_cols, 
                                                           algorithm=selected_algorithm, 
                                                           n_components=selected_dimension)
        
        dff = df.copy()
        
        # 应用所有筛选条件
        if selected_clusters and len(selected_clusters) > 0:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        
        if selected_units and len(selected_units) > 0 and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        
        if selected_parts and len(selected_parts) > 0 and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        
        if selected_types and len(selected_types) > 0 and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]
        
        hover_cols = ['sample_id', 'image_name']
        if 'sherd_id' in dff.columns:
            hover_cols.append('sherd_id')
        if 'unit_C' in dff.columns:
            hover_cols.append('unit_C')
        if 'part_C' in dff.columns:
            hover_cols.append('part_C')
        if 'type_C' in dff.columns:
            hover_cols.append('type_C')
        
        custom = ['sample_id', 'image_name']
        if 'paired_images' in dff.columns:
            custom.append('paired_images')
        if 'sherd_id' in dff.columns:
            custom.append('sherd_id')
        
        # 生成散点图
        if selected_dimension == 2:
            fig = px.scatter(dff, x=f'{reduction_key}_0', y=f'{reduction_key}_1', 
                           color=dff[cluster_col].astype(str), 
                           hover_data=hover_cols, custom_data=custom,
                           color_discrete_sequence=CLUSTER_COLORS)
        else:  # 3D
            # 根据z轴选择生成不同的3D图
            if z_axis == 'unit_C' and 'unit_C' in dff.columns:
                # 创建一个字典将带圆圈的数字映射到普通数字
                circle_to_num = {
                    '①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5,
                    '⑥': 6, '⑦': 7, '⑧': 8, '⑨': 9, '⑩': 10,
                    '⑪': 11, '⑫': 12, '⑬': 13, '⑭': 14, '⑮': 15,
                    '⑯': 16, '⑰': 17, '⑱': 18, '⑲': 19, '⑳': 20,
                    '㉑': 21, '㉒': 22, '㉓': 23, '㉔': 24, '㉕': 25,
                    '㉖': 26, '㉗': 27, '㉘': 28, '㉙': 29, '㉚': 30
                }
                
                # 为H690项目创建z轴使用的数字值
                dff = dff.copy()
                dff['h690_num'] = dff['unit_C'].apply(
                    lambda x: circle_to_num.get(x[4:], 0) 
                    if isinstance(x, str) and x.startswith('H690') and len(x) > 4 
                    else 0
                )
                
                # 使用数字作为z轴值，这样Plotly会自动按正确顺序排列
                fig = px.scatter_3d(dff, x=f'{reduction_key}_0', y=f'{reduction_key}_1', z='h690_num',
                                  color=dff[cluster_col].astype(str), 
                                  hover_data=hover_cols + ['unit_C'],  # 在hover中显示原始的unit_C值
                                  custom_data=custom,
                                  title=f'{selected_algorithm} + unit_C三维图',
                                  color_discrete_sequence=CLUSTER_COLORS)
                
                # 更新z轴标题
                fig.update_layout(
                    scene=dict(
                        zaxis=dict(
                            title='H690序号'
                        )
                    )
                )
                
                # 直接设置点大小
                fig.update_traces(marker={'size': 1})
            else:
                # 确保已经计算了3维降维结果
                df, reduction_key_3d = ensure_dimensionality_reduction(df, feature_cols, algorithm=selected_algorithm, n_components=3,
                                                                      perplexity=tsne_perplexity if selected_algorithm == 'tsne' else 30,
                                                                      n_neighbors=umap_n_neighbors if selected_algorithm == 'umap' else 15,
                                                                      min_dist=umap_min_dist if selected_algorithm == 'umap' else 0.1)
                dff = df.copy()
                
                # 重新应用筛选条件
                if selected_clusters and len(selected_clusters) > 0:
                    dff = dff[dff[cluster_col].isin(selected_clusters)]
                if selected_units and len(selected_units) > 0 and 'unit_C' in dff.columns:
                    dff = dff[dff['unit_C'].isin(selected_units)]
                if selected_parts and len(selected_parts) > 0 and 'part_C' in dff.columns:
                    dff = dff[dff['part_C'].isin(selected_parts)]
                if selected_types and len(selected_types) > 0 and 'type_C' in dff.columns:
                    dff = dff[dff['type_C'].isin(selected_types)]
                    
                fig = px.scatter_3d(dff, x=f'{reduction_key_3d}_0', y=f'{reduction_key_3d}_1', z=f'{reduction_key_3d}_2',
                                  color=dff[cluster_col].astype(str), 
                                  hover_data=hover_cols, custom_data=custom,
                                  title=f'{selected_algorithm}三维降维图',
                                  color_discrete_sequence=CLUSTER_COLORS)
                # 直接设置点大小
                fig.update_traces(marker={'size': 1})
        
        # 根据维度设置不同的点大小（保持2D的设置）
        if selected_dimension == 2:
            fig.update_traces(marker={'size': 8})
        
        # 保存当前使用的降维参数
        params = data_store.get('params', {})
        
        # 为当前算法和参数组合保存参数
        if selected_algorithm == 'tsne':
            params[reduction_key] = {'perplexity': tsne_perplexity}
        elif selected_algorithm == 'umap':
            params[reduction_key] = {'n_neighbors': umap_n_neighbors, 'min_dist': umap_min_dist}
        elif selected_algorithm == 'pca':
            params[reduction_key] = {}
        
        # 更新data-store
        updated_data_store = {
            'df': df.to_json(orient='split'),
            'feature_cols': feature_cols,
            'cluster_col': cluster_col,
            'image_col': image_col,
            'params': params
        }
        
        return fig, updated_data_store


    @app.callback(
        Output('cluster-images-store', 'data'),
        Output('sample-panel', 'children'),
        Output('selected-meta', 'children'),
        Input('tsne-plot', 'clickData'),
        State('data-store', 'data')
    )
    def show_selected(clickData, data_store=None):
        # when no click, clear store
        if not clickData or data_store is None:
            return [], html.Div('点击一个点以查看图片'), ''

        pts = clickData.get('points', [])
        if len(pts) == 0:
            return [], html.Div('点击一个点以查看图片'), ''

        p = pts[0]
        cd = p.get('customdata') or []
        sample_id = cd[0] if len(cd) >= 1 else None
        img_name = cd[1] if len(cd) >= 2 else p.get('hovertext')
        paired_images = cd[2] if len(cd) >= 3 else None
        
        # 从data-store获取数据
        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']
        image_col = data_store['image_col']

        df = ensure_sample_ids(df, image_col)

        # find the primary row
        row = None
        if sample_id is not None:
            row_candidates = df[df.get('sample_id') == sample_id]
            if len(row_candidates) > 0:
                row = row_candidates.iloc[0]

        if row is None and img_name is not None:
            row_candidates = df[df.get('image_name') == img_name]
            if len(row_candidates) > 0:
                row = row_candidates.iloc[0]

        if row is None:
            x = p.get('x')
            y = p.get('y')
            for algo in ['tsne', 'umap', 'pca']:
                if f'{algo}_0' in df.columns and f'{algo}_1' in df.columns:
                    row_candidate = df[(np.isclose(df[f'{algo}_0'], x)) & (np.isclose(df[f'{algo}_1'], y))]
                    if len(row_candidate) > 0:
                        row = row_candidate.iloc[0]
                        break
        if row is None:
            return [], html.Div('未找到对应记录'), ''

        sample_id = sample_id or row.get('sample_id')
        paired_images = paired_images or row.get('paired_images')

        # determine sample id (main id) to find both sides
        if sample_id is None:
            name = str(row['image_name'])
            sample_id = Path(name).stem.replace('_exterior', '').replace('_interior', '')

        # collect images for this sample (both sides)
        paired_names = []
        if paired_images and str(paired_images) != 'nan':
            paired_names = [n for n in str(paired_images).split(';') if n]
        elif img_name:
            paired_names = [str(img_name)]

        base_dir = Path(row[image_col]).parent
        sample_imgs = []
        for nm in paired_names:
            ipath = base_dir / nm
            if not ipath.exists():
                cand = image_root / Path(nm).name
                if cand.exists():
                    ipath = cand
            b64 = img_to_base64(ipath)
            if b64:
                sample_imgs.append(html.Img(src=b64, style={'height': '200px', 'border': '1px solid #ccc', 'margin-right': '6px'}))

        if len(sample_imgs) == 0:
            sample_imgs = [html.Div('未找到正反面图片')]

        left_col = html.Div(sample_imgs)
        sample_panel_children = html.Div([html.Div(left_col, style={'display': 'flex', 'gap': '8px', 'justifyContent': 'center'})])

        # collect all images in same cluster
        cluster_val = row[cluster_col]
        same_cluster = df[df[cluster_col] == cluster_val]
        image_paths = []
        for _, r in same_cluster.iterrows():
            base_dir = Path(r[image_col]).parent
            names = []
            if 'paired_images' in r and pd.notna(r['paired_images']):
                names = [n for n in str(r['paired_images']).split(';') if n]
            elif 'image_name' in r:
                names = [str(r['image_name'])]
            for nm in names:
                ipath = base_dir / nm
                if not ipath.exists():
                    cand = image_root / Path(nm).name
                    if cand.exists():
                        ipath = cand
                image_paths.append(str(ipath))

        # 构建详细的元数据信息
        meta_parts = [
            html.B(row['image_name']),
            html.Br(),
            f"聚类: {cluster_val}",
            html.Br(),
            f"样本ID: {sample_id}"
        ]
        
        # 添加 jd_sherds_info 的字段
        if 'sherd_id' in row.index and pd.notna(row['sherd_id']):
            meta_parts.extend([
                html.Br(),
                f"陶片ID: {row['sherd_id']}"
            ])
        if 'unit_C' in row.index and pd.notna(row['unit_C']):
            meta_parts.extend([
                html.Br(),
                f"单位: {row['unit_C']}"
            ])
        if 'part_C' in row.index and pd.notna(row['part_C']):
            meta_parts.extend([
                html.Br(),
                f"部位: {row['part_C']}"
            ])
        if 'type_C' in row.index and pd.notna(row['type_C']):
            meta_parts.extend([
                html.Br(),
                f"类型: {row['type_C']}"
            ])
        if 'part' in row.index and pd.notna(row['part']):
            meta_parts.extend([
                html.Br(),
                f"Part: {row['part']}"
            ])
        if 'type' in row.index and pd.notna(row['type']):
            meta_parts.extend([
                html.Br(),
                f"Type: {row['type']}"
            ])
        
        meta = html.Div(meta_parts)

        return image_paths, sample_panel_children, meta


    PAGE_SIZE = 20

    @app.callback(
        Output('cluster-panel', 'children'),
        Output('page-indicator', 'children'),
        Input('cluster-images-store', 'data'),
        Input('cluster-page', 'data')
    )
    def update_cluster_panel(image_paths, page):
        if not image_paths:
            return html.Div('点击点以加载该簇的图片'), ''
        total = len(image_paths)
        max_page = (total + PAGE_SIZE - 1) // PAGE_SIZE
        page = max(1, min(page or 1, max_page))
        start = (page - 1) * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        thumbs = []
        for pth in image_paths[start:end]:
            b64 = img_to_base64(Path(pth), max_size=180)
            if b64:
                thumbs.append(html.Img(src=b64, style={'height': '120px', 'margin': '4px', 'border': '1px solid #ccc'}))
            else:
                thumbs.append(html.Div(str(Path(pth).name)))
        grid = html.Div(thumbs, style={'display': 'flex', 'flexWrap': 'wrap'})
        indicator = f'Page {page} / {max_page} ({total} images)'
        return grid, indicator


    @app.callback(
        Output('cluster-page', 'data'),
        Input('cluster-prev', 'n_clicks'),
        Input('cluster-next', 'n_clicks'),
        State('cluster-page', 'data'),
        State('cluster-images-store', 'data')
    )
    def change_page(prev_clicks, next_clicks, current_page, image_paths):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_page or 1
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        total = len(image_paths) if image_paths else 0
        max_page = (total + PAGE_SIZE - 1) // PAGE_SIZE if total > 0 else 1
        current = current_page or 1
        if triggered == 'cluster-prev':
            new = max(1, current - 1)
        elif triggered == 'cluster-next':
            new = min(max_page, current + 1)
        else:
            new = current
        return new





    # 聚类特征热力图生成
    @app.callback(
        Output('heatmap-container', 'children'),
        Input('visualization-tabs', 'value'),
        State('cluster-metadata-store', 'data')
    )
    def update_heatmap(tab_value, cluster_metadata):
        if tab_value != 'heatmap' or cluster_metadata is None:
            return html.Div('请选择"聚类特征热力图"选项卡')
        
        # 显示加载状态
        loading_div = html.Div([
            html.H4('正在生成热力图...'),
            html.Div(style={'margin': '20px'}),
            dcc.Loading(type='default', children=html.Div(id='heatmap-loading-spinner'))
        ])
        
        try:
            # 从元数据获取聚类中心
            cluster_centers = np.array(cluster_metadata.get('cluster_centers', []))
            if cluster_centers.shape[0] == 0:
                return html.Div('未找到聚类中心数据')
            
            # 优化：只使用前50个特征来生成热力图，减少计算量
            if cluster_centers.shape[1] > 50:
                cluster_centers = cluster_centers[:, :50]
            
            # 生成热力图
            fig = create_cluster_pattern_heatmap(cluster_centers)
            
            return dcc.Graph(figure=fig)
        except Exception as e:
            return html.Div(f'生成热力图时出错: {str(e)}')


    # 聚类相似度矩阵生成
    @app.callback(
        Output('similarity-container', 'children'),
        Input('visualization-tabs', 'value'),
        State('cluster-metadata-store', 'data')
    )
    def update_similarity(tab_value, cluster_metadata):
        if tab_value != 'similarity' or cluster_metadata is None:
            return html.Div('请选择"聚类相似度矩阵"选项卡')
        
        # 显示加载状态
        loading_div = html.Div([
            html.H4('正在生成相似度矩阵...'),
            html.Div(style={'margin': '20px'}),
            dcc.Loading(type='default', children=html.Div(id='similarity-loading-spinner'))
        ])
        
        try:
            # 从元数据获取聚类中心
            cluster_centers = np.array(cluster_metadata.get('cluster_centers', []))
            if cluster_centers.shape[0] == 0:
                return html.Div('未找到聚类中心数据')
            
            # 优化：只使用前100个特征来计算相似度，减少计算量
            if cluster_centers.shape[1] > 100:
                cluster_centers = cluster_centers[:, :100]
            
            # 生成相似度矩阵
            fig = create_cluster_similarity_matrix(cluster_centers)
            
            return dcc.Graph(figure=fig)
        except Exception as e:
            return html.Div(f'生成相似度矩阵时出错: {str(e)}')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False, port=9000, host='127.0.0.1')
