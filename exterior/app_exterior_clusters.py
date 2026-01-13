import base64
import io
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State
import dash

from umap import UMAP

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# 配置 - 仅正面图像
CSV = (BASE_DIR / 'exterior_cluster_table.csv').resolve()
IMAGE_ROOT = (ROOT_DIR / 'all_cutouts').resolve()  # 图像根目录

# 加载聚类元数据
def load_cluster_metadata():
    """加载正面图像聚类元数据"""
    meta_path = (BASE_DIR / 'exterior_kmeans_results' / 'cluster_metadata.json').resolve()
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# 生成聚类特征热力图
def create_cluster_pattern_heatmap(cluster_centers, feature_names=None):
    """创建聚类中心特征热力图"""
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(cluster_centers.shape[1])]
    
    # 对特征进行降维以便可视化
    if cluster_centers.shape[1] > 100:
        # 只选择前100个特征或使用PCA降维
        cluster_centers_vis = cluster_centers[:, :100]
        feature_names = feature_names[:100]
    else:
        cluster_centers_vis = cluster_centers
    
    # 创建热力图
    fig = px.imshow(
        cluster_centers_vis,
        x=feature_names,
        y=[f'Cluster {i}' for i in range(cluster_centers_vis.shape[0])],
        color_continuous_scale='Viridis',
        title='正面图像聚类中心特征热力图',
        labels={'x': '特征', 'y': '聚类', 'color': '特征值'}
    )
    
    # 优化布局
    fig.update_layout(
        xaxis={'tickangle': -45, 'title_text': '特征索引'},
        yaxis={'title_text': '聚类ID'},
        coloraxis_colorbar={'title': '特征值'}
    )
    
    return fig

# 创建聚类相似度矩阵
def create_cluster_similarity_matrix(cluster_centers):
    """创建聚类相似度矩阵"""
    n_clusters = cluster_centers.shape[0]
    similarity_matrix = np.zeros((n_clusters, n_clusters))
    
    # 计算聚类间的余弦相似度
    for i in range(n_clusters):
        for j in range(n_clusters):
            a = cluster_centers[i]
            b = cluster_centers[j]
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            similarity_matrix[i, j] = similarity
    
    # 创建热力图
    fig = px.imshow(
        similarity_matrix,
        x=[f'Cluster {i}' for i in range(n_clusters)],
        y=[f'Cluster {i}' for i in range(n_clusters)],
        color_continuous_scale='RdBu_r',
        title='正面图像聚类相似度矩阵',
        labels={'x': '聚类', 'y': '聚类', 'color': '余弦相似度'},
        zmin=-1, zmax=1
    )
    
    # 优化布局
    fig.update_layout(
        xaxis={'tickangle': -45},
        coloraxis_colorbar={'title': '相似度'}
    )
    
    return fig

def detect_columns(df):
    cluster_col = None
    image_col = None
    for c in df.columns:
        if 'cluster' in c.lower() and df[c].nunique() > 1:
            cluster_col = c
        if 'image' in c.lower() and ('path' in c.lower() or 'name' in c.lower() or 'file' in c.lower()):
            image_col = c
    return cluster_col, image_col

def generate_reduction_key(algorithm, n_components, **params):
    """生成降维结果的唯一标识符"""
    if algorithm == 'pca':
        return f'{algorithm}_{n_components}'
    elif algorithm == 'tsne':
        perplexity = params.get('perplexity', 30.0)
        return f'{algorithm}_{n_components}_perp{int(perplexity)}'
    elif algorithm == 'umap':
        n_neighbors = params.get('n_neighbors', 15)
        min_dist = params.get('min_dist', 0.1)
        return f'{algorithm}_{n_components}_nn{n_neighbors}_md{int(min_dist*100)}'
    else:
        return f'{algorithm}_{n_components}'

def ensure_dimensionality_reduction(df, feature_cols, algorithm='umap', n_components=2, perplexity=30.0, n_neighbors=15, min_dist=0.1):
    """Ensure dimensionality reduction has been performed for the given algorithm and parameters."""
    # 生成当前参数组合的唯一键
    reduction_key = generate_reduction_key(algorithm, n_components, perplexity=perplexity, n_neighbors=n_neighbors, min_dist=min_dist)
    cols = [f'{reduction_key}_{i}' for i in range(n_components)]
    
    # 检查是否已经有了该参数组合的结果
    if all(c in df.columns for c in cols):
        return df, reduction_key
    
    X = df[feature_cols].values
    
    # 标准化特征（对降维很重要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algorithm == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca')
    elif algorithm == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif algorithm == 'umap':
        reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    else:
        # Fallback to t-SNE for any other case
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca')
        algorithm = 'tsne'
    
    Xt = reducer.fit_transform(X_scaled)  # 使用标准化后的特征
    for i in range(n_components):
        df[f'{reduction_key}_{i}'] = Xt[:, i]
    return df, reduction_key

def img_to_base64(path, max_size=400):
    try:
        im = Image.open(path).convert('RGBA')
        # naive white->transparent: make near-white pixels transparent
        try:
            import numpy as _np
            arr = _np.array(im)
            if arr.shape[2] == 4:
                r, g, b, a = _np.split(arr, 4, axis=2)
                mask = (_np.squeeze(r) > 245) & (_np.squeeze(g) > 245) & (_np.squeeze(b) > 245)
                arr[mask, 3] = 0
                im = Image.fromarray(arr)
        except Exception:
            # if numpy not available, skip transparency step
            pass

        # resize for faster transfer
        im.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        data = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{data}'
    except Exception:
        return None

def create_app(csv=CSV, image_root=IMAGE_ROOT):
    csv_path = Path(csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}。请先运行 build_exterior_table.py")
    
    df = pd.read_csv(csv_path)
    cluster_col, image_col = detect_columns(df)
    if cluster_col is None or image_col is None:
        raise RuntimeError('无法识别聚类列或图片列，请检查 CSV')

    # 验证是否为正面图像数据
    exterior_count = sum(1 for name in df[image_col] if 'exterior' in name.lower())
    print(f" 数据验证: {exterior_count}/{len(df)} 为正面图像")
    
    # 检查是否包含 jd_sherds_info 的字段
    has_info = 'sherd_id' in df.columns or 'unit_C' in df.columns
    if has_info:
        matched_count = df['sherd_id'].notna().sum() if 'sherd_id' in df.columns else 0
        print(f" 已加载包含 jd_sherds_info 的表格，匹配了 {matched_count}/{len(df)} 条记录")
    else:
        print(f" 表格中未包含 jd_sherds_info 字段")

    # detect feature columns (numeric excluding metadata)
    exclude = {cluster_col, image_col, 'image_name', 'sample_id', 'side', 'image_id', 'sherd_id', 
               'unit', 'part', 'type', 'image_side', 'image_id_original', 'unit_C', 'part_C', 'type_C', 'image_path'}
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if len(feature_cols) == 0:
        raise RuntimeError('未找到数值特征列')

    print(f" 特征列数量: {len(feature_cols)}")

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
    hover_cols = ['image_name']
    if 'sherd_id' in df.columns:
        hover_cols.append('sherd_id')
    if 'unit_C' in df.columns:
        hover_cols.append('unit_C')
    if 'part_C' in df.columns:
        hover_cols.append('part_C')
    if 'type_C' in df.columns:
        hover_cols.append('type_C')
    
    # include image_name and sample_id in customdata for reliable lookup on click
    custom = ['image_name']
    if 'sample_id' in df.columns:
        custom.append('sample_id')
    if 'sherd_id' in df.columns:
        custom.append('sherd_id')
        
    # 初始使用UMAP降维的结果
    initial_algorithm = 'umap'
    fig = px.scatter(df, x=f'{initial_reduction_key}_0', y=f'{initial_reduction_key}_1', 
                     color=df[cluster_col].astype(str), hover_data=hover_cols, custom_data=custom)

    # 准备降维算法选项
    algorithm_options = [
        {'label': 'PCA', 'value': 'pca'},
        {'label': 't-SNE', 'value': 'tsne'},
        {'label': 'UMAP', 'value': 'umap'}
    ]
    
    # 加载聚类元数据
    cluster_metadata = load_cluster_metadata()
    
    app.layout = html.Div([
        html.H3('陶片正面图像聚类交互可视化'),
        html.P(f" 数据概览: {len(df)} 张正面图像, {len(clusters)} 个聚类, {len(feature_cols)} 维特征", 
               style={'color': '#666', 'marginBottom': '16px'}),
        
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
                                dcc.Graph(id='tsne-plot', figure=fig, style={'height': 'calc(100vh - 240px)', 'width': '100%'})
                            ]
                        ),
                        html.Div(id='plot-loading-status', style={'textAlign': 'center', 'marginTop': '8px', 'color': '#666'})
                    ], style={'flex': '1 1 auto'}),
                    html.Div([
                        html.H4('聚类图像预览', style={'marginBottom': '8px'}),
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
        
        # bottom area: sample detail panel
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
        dcc.Store(id='cluster-metadata-store', data=cluster_metadata)
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

    @app.callback(
        [Output('tsne-plot', 'figure'),
         Output('data-store', 'data')],
        [Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value'),
         Input('algorithm-selector', 'value'),
         Input('dimension-selector', 'value'),
         Input('tsne-perplexity', 'value'),
         Input('umap-n-neighbors', 'value'),
         Input('umap-min-dist', 'value')],
        State('data-store', 'data')
    )
    def update_plot(selected_clusters, selected_units, selected_parts, selected_types, 
                   selected_algorithm, selected_dimension, 
                   tsne_perplexity=30, umap_n_neighbors=15, umap_min_dist=0.1,
                   data_store=None):
        # 从data-store获取数据
        if data_store is None:
            raise ValueError("Data store is empty")
        
        # 确保selected_algorithm不是None
        if selected_algorithm is None:
            selected_algorithm = 'umap'
        
        # 解析数据
        from io import StringIO
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
        
        # 准备 hover_data 和 custom_data
        hover_cols = ['image_name']
        if 'sherd_id' in dff.columns:
            hover_cols.append('sherd_id')
        if 'unit_C' in dff.columns:
            hover_cols.append('unit_C')
        if 'part_C' in dff.columns:
            hover_cols.append('part_C')
        if 'type_C' in dff.columns:
            hover_cols.append('type_C')
        
        custom = ['image_name']
        if 'sample_id' in dff.columns:
            custom.append('sample_id')
        if 'sherd_id' in dff.columns:
            custom.append('sherd_id')
        
        # 创建图表
        if selected_dimension == 2:
            fig = px.scatter(dff, 
                           x=f'{reduction_key}_0', 
                           y=f'{reduction_key}_1',
                           color=dff[cluster_col].astype(str), 
                           hover_data=hover_cols,
                           custom_data=custom,
                           title=f'正面图像聚类可视化 ({selected_algorithm.upper()}) - {len(dff)} 张图像')
        else:
            fig = px.scatter_3d(dff, 
                              x=f'{reduction_key}_0', 
                              y=f'{reduction_key}_1',
                              z=f'{reduction_key}_2',
                              color=dff[cluster_col].astype(str), 
                              hover_data=hover_cols,
                              custom_data=custom,
                              title=f'正面图像聚类可视化 ({selected_algorithm.upper()} 3D) - {len(dff)} 张图像')
        
        fig.update_traces(marker_size=8)
        fig.update_layout(height=600, showlegend=True)
        
        # 更新data-store
        updated_data = {
            'df': df.to_json(orient='split'),
            'feature_cols': feature_cols,
            'cluster_col': cluster_col,
            'image_col': image_col
        }
        
        return fig, updated_data

    # 点击散点图显示聚类图像
    @app.callback(
        Output('cluster-images-store', 'data'),
        Input('tsne-plot', 'clickData'),
        State('data-store', 'data')
    )
    def update_cluster_images(clickData, data_store):
        if clickData is None or data_store is None:
            return []
        
        # 解析数据
        from io import StringIO
        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']
        
        # 获取点击的聚类ID
        clicked_point = clickData['points'][0]
        cluster_id = int(clicked_point['curveNumber'])
        
        # 如果有颜色信息，使用颜色对应的聚类
        if 'customdata' in clicked_point and clicked_point['customdata']:
            # 从customdata中获取图像名称，然后找到对应的聚类
            image_name = clicked_point['customdata'][0]
            cluster_id = df[df['image_name'] == image_name][cluster_col].iloc[0]
        else:
            # 从散点图的颜色获取聚类ID
            cluster_id = int(clicked_point['curveNumber'])
        
        # 获取该聚类的所有图像
        cluster_images = df[df[cluster_col] == cluster_id]['image_name'].tolist()
        
        return cluster_images

    # 显示聚类图像预览
    @app.callback(
        [Output('cluster-panel', 'children'),
         Output('page-indicator', 'children')],
        [Input('cluster-images-store', 'data'),
         Input('cluster-prev', 'n_clicks'),
         Input('cluster-next', 'n_clicks')],
        State('cluster-page', 'data')
    )
    def display_cluster_images(cluster_images, prev_clicks, next_clicks, current_page):
        if not cluster_images:
            return [], ""
        
        images_per_page = 6
        total_pages = (len(cluster_images) - 1) // images_per_page + 1
        
        # 处理翻页
        ctx = dash.callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'cluster-prev' and current_page > 1:
                current_page -= 1
            elif button_id == 'cluster-next' and current_page < total_pages:
                current_page += 1
        
        # 获取当前页的图像
        start_idx = (current_page - 1) * images_per_page
        end_idx = min(start_idx + images_per_page, len(cluster_images))
        page_images = cluster_images[start_idx:end_idx]
        
        # 创建图像元素
        image_elements = []
        for img_name in page_images:
            img_path = IMAGE_ROOT / img_name
            if img_path.exists():
                b64_img = img_to_base64(img_path, max_size=150)
                if b64_img:
                    image_elements.append(
                        html.Div([
                            html.Img(src=b64_img, style={'width': '100%', 'marginBottom': '4px'}),
                            html.P(img_name, style={'fontSize': '10px', 'textAlign': 'center', 'margin': '0'})
                        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
                    )
        
        page_info = f"第 {current_page}/{total_pages} 页 (共 {len(cluster_images)} 张图像)"
        
        return image_elements, page_info

    # 更新页码状态
    @app.callback(
        Output('cluster-page', 'data'),
        [Input('cluster-prev', 'n_clicks'),
         Input('cluster-next', 'n_clicks')],
        [State('cluster-page', 'data'),
         State('cluster-images-store', 'data')]
    )
    def update_page(prev_clicks, next_clicks, current_page, cluster_images):
        if not cluster_images:
            return 1
        
        images_per_page = 6
        total_pages = (len(cluster_images) - 1) // images_per_page + 1
        
        ctx = dash.callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'cluster-prev' and current_page > 1:
                return current_page - 1
            elif button_id == 'cluster-next' and current_page < total_pages:
                return current_page + 1
        
        return current_page

    # 显示聚类特征热力图
    @app.callback(
        Output('heatmap-container', 'children'),
        Input('visualization-tabs', 'value'),
        State('cluster-metadata-store', 'data')
    )
    def update_heatmap(active_tab, cluster_metadata):
        if active_tab != 'heatmap' or not cluster_metadata:
            return []
        
        cluster_centers = np.array(cluster_metadata['cluster_centers'])
        fig = create_cluster_pattern_heatmap(cluster_centers)
        
        return [dcc.Graph(figure=fig, style={'height': '100%'})]

    # 显示聚类相似度矩阵
    @app.callback(
        Output('similarity-container', 'children'),
        Input('visualization-tabs', 'value'),
        State('cluster-metadata-store', 'data')
    )
    def update_similarity(active_tab, cluster_metadata):
        if active_tab != 'similarity' or not cluster_metadata:
            return []
        
        cluster_centers = np.array(cluster_metadata['cluster_centers'])
        fig = create_cluster_similarity_matrix(cluster_centers)
        
        return [dcc.Graph(figure=fig, style={'height': '100%'})]
    
    return app

if __name__ == '__main__':
    try:
        app = create_app()
        print(f"\n 启动正面图像聚类可视化应用")
        print(f" 访问地址: http://127.0.0.1:9001/")
        print(f" 按 Ctrl+C 停止应用")
        app.run(debug=True, host='127.0.0.1', port=9001)
    except Exception as e:
        print(f" 启动应用失败: {e}")
        print("请确保已运行完整的处理流程：")
        print("1. DINOV3_features_exterior.py")
        print("2. kmeans_exterior_only.py") 
        print("3. build_exterior_table.py")