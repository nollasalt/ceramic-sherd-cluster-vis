import base64
import io
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State

from umap import UMAP

# 加载聚类元数据
def load_cluster_metadata():
    """加载聚类元数据"""
    meta_path = Path('all_kmeans_new/cluster_metadata.json')
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
        title='聚类中心特征热力图',
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
        title='聚类相似度矩阵',
        labels={'x': '聚类', 'y': '聚类', 'color': '余弦相似度'},
        zmin=-1, zmax=1
    )
    
    # 优化布局
    fig.update_layout(
        xaxis={'tickangle': -45},
        coloraxis_colorbar={'title': '相似度'}
    )
    
    return fig


CSV = 'sherd_cluster_table_clustered_only.csv'
IMAGE_ROOT = Path('all_cutouts')


def detect_columns(df):
    cluster_col = None
    image_col = None
    for c in df.columns:
        if 'cluster' in c.lower() and df[c].nunique() > 1:
            cluster_col = c
        if 'image' in c.lower() and ('path' in c.lower() or 'name' in c.lower() or 'file' in c.lower()):
            image_col = c
    return cluster_col, image_col


def ensure_dimensionality_reduction(df, feature_cols, algorithm='umap', n_components=2, perplexity=30.0, n_neighbors=15, min_dist=0.1):
    """Ensure dimensionality reduction has been performed for the given algorithm."""
    cols = [f'{algorithm}_{i}' for i in range(n_components)]
    if all(c in df.columns for c in cols):
        return df
    
    X = df[feature_cols].values
    
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
    
    Xt = reducer.fit_transform(X)
    for i in range(n_components):
        df[f'{algorithm}_{i}'] = Xt[:, i]
    return df


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
    df = pd.read_csv(csv)
    cluster_col, image_col = detect_columns(df)
    if cluster_col is None or image_col is None:
        raise RuntimeError('无法识别聚类列或图片列，请检查 CSV')

    # 检查是否包含 jd_sherds_info 的字段（在 build_table.py 中已合并）
    has_info = 'sherd_id' in df.columns or 'unit_C' in df.columns
    if has_info:
        matched_count = df['sherd_id'].notna().sum() if 'sherd_id' in df.columns else 0
        print(f"✅ 已加载包含 jd_sherds_info 的表格，匹配了 {matched_count}/{len(df)} 条记录")
    else:
        print(f"ℹ️  表格中未包含 jd_sherds_info 字段，请运行 build_table.py 进行合并")

    # detect feature columns (numeric excluding metadata)
    exclude = {cluster_col, image_col, 'image_name', 'sample_id', 'side', 'image_id', 'sherd_id', 
               'unit', 'part', 'type', 'image_side', 'image_id_original', 'unit_C', 'part_C', 'type_C'}
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if len(feature_cols) == 0:
        raise RuntimeError('未找到数值特征列')

    # 初始使用UMAP降维
    df = ensure_dimensionality_reduction(df, feature_cols, algorithm='umap', n_components=2)

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
    fig = px.scatter(df, x=f'{initial_algorithm}_0', y=f'{initial_algorithm}_1', 
                     color=df[cluster_col].astype(str), hover_data=hover_cols, custom_data=custom)

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
        
        # 可视化类型选择
        html.Div([
            html.Label('可视化类型:'),
            dcc.RadioItems(id='visualization-type', options=visualization_types, value='scatter', inline=True),
        ], style={'marginBottom': '12px'}),
        
        # 选项卡容器
        dcc.Tabs(id='visualization-tabs', value='scatter', children=[
            # 散点图选项卡
            dcc.Tab(label='散点图', value='scatter', children=[
                html.Div([
                    html.Div([
                        html.Label('降维算法:'),
                        dcc.Dropdown(id='algorithm-selector', options=algorithm_options, value='umap'),
                    ], style={'width': '15%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '10px'}),
                    html.Div([
                        html.Label('可视化维度:'),
                        dcc.RadioItems(id='dimension-selector', options=[{'label': '2D', 'value': 2}, {'label': '3D', 'value': 3}], value=2),
                    ], style={'width': '15%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '10px'}),
                    # t-SNE参数
                    html.Div([
                        html.Label('t-SNE困惑度 (perplexity):'),
                        dcc.Slider(id='tsne-perplexity', min=5, max=50, step=1, value=30, marks={5: '5', 25: '25', 50: '50'}),
                    ], style={'width': '25%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '10px'}),
                    # UMAP参数
                    html.Div([
                        html.Label('UMAP邻居数 (n_neighbors):'),
                        dcc.Slider(id='umap-n-neighbors', min=5, max=50, step=1, value=15, marks={5: '5', 25: '25', 50: '50'}),
                    ], style={'width': '25%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '10px'}),
                    html.Div([
                        html.Label('UMAP最小距离 (min_dist):'),
                        dcc.Slider(id='umap-min-dist', min=0.01, max=1.0, step=0.01, value=0.1, marks={0.01: '0.01', 0.5: '0.5', 1.0: '1.0'}),
                    ], style={'width': '25%', 'display': 'inline-block', 'marginBottom': '8px'}),
                    html.Div([
                        html.Label('筛选簇:'),
                        dcc.Dropdown(id='cluster-filter', options=[{'label': str(c), 'value': c} for c in clusters], multi=True, placeholder='选择一个或多个簇（留空显示所有）'),
                    ], style={'width': '25%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '10px'}),
                    html.Div([
                        html.Label('筛选单位:'),
                        dcc.Dropdown(id='unit-filter', options=init_unit_options, multi=True, placeholder='选择单位（留空显示所有）'),
                    ], style={'width': '15%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '10px'}),
                    html.Div([
                        html.Label('筛选部位:'),
                        dcc.Dropdown(id='part-filter', options=init_part_options, multi=True, placeholder='选择部位（留空显示所有）'),
                    ], style={'width': '15%', 'display': 'inline-block', 'marginBottom': '8px', 'marginRight': '10px'}),
                    html.Div([
                        html.Label('筛选类型:'),
                        dcc.Dropdown(id='type-filter', options=init_type_options, multi=True, placeholder='选择类型（留空显示所有）'),
                    ], style={'width': '15%', 'display': 'inline-block', 'marginBottom': '8px'}),
                ], style={'marginBottom': '8px'}),
                
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
        
        # 解析数据
        from io import StringIO
        df = pd.read_json(StringIO(data_store['df']), orient='split')
        feature_cols = data_store['feature_cols']
        cluster_col = data_store['cluster_col']
        image_col = data_store['image_col']
        
        # 检查是否已经计算过该算法和维度的结果
        required_cols = [f'{selected_algorithm}_{i}' for i in range(selected_dimension)]
        params_key = f'{selected_algorithm}_{selected_dimension}'
        
        # 对于t-SNE和UMAP，如果参数变化则需要重新计算
        needs_recalculation = False
        
        if selected_algorithm == 'tsne':
            # 检查t-SNE参数是否变化
            if params_key not in data_store.get('params', {}):
                needs_recalculation = True
            else:
                old_perplexity = data_store['params'][params_key].get('perplexity', 30)
                if old_perplexity != tsne_perplexity:
                    needs_recalculation = True
                    # 删除旧的t-SNE结果
                    for col in [col for col in df.columns if col.startswith('tsne_')]:
                        df = df.drop(columns=[col])
        elif selected_algorithm == 'umap':
            # 检查UMAP参数是否变化
            if params_key not in data_store.get('params', {}):
                needs_recalculation = True
            else:
                old_n_neighbors = data_store['params'][params_key].get('n_neighbors', 15)
                old_min_dist = data_store['params'][params_key].get('min_dist', 0.1)
                if old_n_neighbors != umap_n_neighbors or old_min_dist != umap_min_dist:
                    needs_recalculation = True
                    # 删除旧的UMAP结果
                    for col in [col for col in df.columns if col.startswith('umap_')]:
                        df = df.drop(columns=[col])
        
        # 如果需要重新计算或结果不存在，则计算降维
        if needs_recalculation or not all(c in df.columns for c in required_cols):
            if selected_algorithm == 'tsne':
                df = ensure_dimensionality_reduction(df, feature_cols, 
                                                    algorithm=selected_algorithm, 
                                                    n_components=selected_dimension,
                                                    perplexity=tsne_perplexity)
            elif selected_algorithm == 'umap':
                df = ensure_dimensionality_reduction(df, feature_cols, 
                                                    algorithm=selected_algorithm, 
                                                    n_components=selected_dimension,
                                                    n_neighbors=umap_n_neighbors,
                                                    min_dist=umap_min_dist)
            else:
                # PCA不需要考虑参数变化
                df = ensure_dimensionality_reduction(df, feature_cols, 
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
        
        # 根据选择的维度创建图表
        if selected_dimension == 2:
            fig = px.scatter(dff, x=f'{selected_algorithm}_0', y=f'{selected_algorithm}_1', 
                           color=dff[cluster_col].astype(str), 
                           hover_data=hover_cols, custom_data=custom)
        else:  # 3D
            fig = px.scatter_3d(dff, x=f'{selected_algorithm}_0', y=f'{selected_algorithm}_1', z=f'{selected_algorithm}_2',
                              color=dff[cluster_col].astype(str), 
                              hover_data=hover_cols, custom_data=custom)
        
        fig.update_traces(marker={'size': 8})
        
        # 保存当前使用的降维参数
        params_key = f'{selected_algorithm}_{selected_dimension}'
        params = data_store.get('params', {})
        
        if selected_algorithm == 'tsne':
            params[params_key] = {'perplexity': tsne_perplexity}
        elif selected_algorithm == 'umap':
            params[params_key] = {'n_neighbors': umap_n_neighbors, 'min_dist': umap_min_dist}
        elif selected_algorithm == 'pca':
            params[params_key] = {}
        
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
        img_name = cd[0] if len(cd) >= 1 else p.get('hovertext')
        sample_id = cd[1] if len(cd) >= 2 else None
        
        # 从data-store获取数据
        from io import StringIO
        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']
        image_col = data_store['image_col']

        # find the primary row
        if img_name is not None:
            row = df[df['image_name'] == img_name]
        else:
            x = p.get('x')
            y = p.get('y')
            # 查找所有可能的降维算法结果
            row = None
            for algo in ['tsne', 'umap', 'pca']:
                if f'{algo}_0' in df.columns and f'{algo}_1' in df.columns:
                    row_candidate = df[(np.isclose(df[f'{algo}_0'], x)) & (np.isclose(df[f'{algo}_1'], y))]
                    if len(row_candidate) > 0:
                        row = row_candidate
                        break
            if row is None:
                return [], html.Div('未找到对应记录'), ''
        if len(row) == 0:
            return [], html.Div('未找到对应记录'), ''
        row = row.iloc[0]

        # determine sample id (main id) to find both sides
        if sample_id is None and 'sample_id' in df.columns:
            sample_id = row.get('sample_id')
        if sample_id is None:
            name = str(row['image_name'])
            sample_id = Path(name).stem.replace('_exterior', '').replace('_interior', '')

        # collect images for this sample (both sides)
        sample_rows = df[(df.get('sample_id') == sample_id) | (df['image_name'].str.contains(sample_id, na=False))]
        sample_imgs = []
        for _, r in sample_rows.iterrows():
            ipath = Path(r[image_col])
            if not ipath.exists():
                cand = image_root / ipath.name
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
            ipath = Path(r[image_col])
            if not ipath.exists():
                cand = image_root / ipath.name
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


    # 可视化类型选择与选项卡同步
    @app.callback(
        Output('visualization-tabs', 'value'),
        Input('visualization-type', 'value')
    )
    def sync_visualization_type(selected_type):
        return selected_type


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
