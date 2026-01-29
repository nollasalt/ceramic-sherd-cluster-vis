"""
陶片聚类交互可视化应用
使用 Dash 构建的 Web 应用，支持降维可视化、聚类浏览等功能
版本: 1.2.0 (优化版)
"""

from io import StringIO
from pathlib import Path
import json
import os
import time

import pandas as pd
import numpy as np
import plotly.express as px
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import dash
from dash import dcc, html, Input, Output, State, ALL

# 从数据处理模块导入
from data_processing import (
    load_cluster_metadata,
    detect_columns,
    ensure_sample_ids,
    build_fused_samples,
    build_samples_for_mode,
    ensure_dimensionality_reduction,
    create_cluster_pattern_heatmap,
    create_cluster_similarity_matrix,
    img_to_base64,
    img_to_base64_full,  # 导入原图函数
    perform_kmeans_clustering,
    perform_agglomerative_clustering,
    perform_spectral_clustering,
    perform_leiden_clustering,
    DEFAULT_CSV,
    DEFAULT_IMAGE_ROOT,
)

# 导入优化模块
try:
    from performance_utils import (
        timing_decorator, cache_plot_result, optimize_dataframe,
        batch_process_images, plot_cache, image_cache
    )
    OPTIMIZATIONS_ENABLED = True
except ImportError:
    OPTIMIZATIONS_ENABLED = False
    # 定义占位装饰器
    def timing_decorator(func): return func
    def cache_plot_result(func): return func
    def optimize_dataframe(df): return df
    image_cache = None

# 配置常量
CSV = DEFAULT_CSV
IMAGE_ROOT = DEFAULT_IMAGE_ROOT
FEATURES_CSV = Path(__file__).parent / 'all_features_dinov3.csv'
TABLE_CSV = Path(__file__).parent / 'sherd_cluster_table_clustered_only.csv'

# 应用配置
APP_CONFIG = {
    'title': '陶片聚类交互可视化',
    'port': 15687,
    'host': '127.0.0.1',
    'debug': False,
    'max_clusters': 50,
    'default_thumbnail_size': 80
}

# UI文本常量
UI_TEXT = {
    'loading': '加载中...',
    'error_no_data': '❌ 没有可用数据',
    'error_invalid_clusters': '❌ 请输入有效的聚类数量 (2-50)',
    'success_reclustering': '✅ 重新聚类完成',
    'click_to_view': '点击散点图中的点来查看聚类详情',
    'cluster_info': '聚类 {}: {} 个样本',
    'sample_details': '样本详情: {}'
}

from app_core.utils import (
    generate_distinct_colors,
    get_part_symbol_settings,
    CLUSTER_COLORS,
    PART_SYMBOL_SEQUENCE,
)
from app_core.layout import build_layout
from app_core.callbacks import (
    register_scatter_callbacks,
    register_compare_callbacks,
    register_cluster_panel_callbacks,
)


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

    # 从元数据读取聚类模式
    cluster_metadata = load_cluster_metadata()
    initial_cluster_mode = cluster_metadata.get('cluster_mode', None) if cluster_metadata else None
    
    # 如果元数据中没有聚类模式，则根据数据自动检测
    if initial_cluster_mode is None:
        img_name_col = 'image_name' if 'image_name' in df.columns else image_col
        has_exterior = df[img_name_col].str.lower().str.contains('exterior').any()
        has_interior = df[img_name_col].str.lower().str.contains('interior').any()
        
        if has_exterior and has_interior:
            initial_cluster_mode = 'merged'
        elif has_exterior:
            initial_cluster_mode = 'exterior'
        elif has_interior:
            initial_cluster_mode = 'interior'
        else:
            initial_cluster_mode = 'merged'  # 默认
        print(f"自动检测聚类模式: {initial_cluster_mode} (exterior={has_exterior}, interior={has_interior})")
    else:
        print(f"从元数据读取聚类模式: {initial_cluster_mode}")

    # 根据聚类模式构建样本数据
    df, feature_cols, dropped_samples = build_samples_for_mode(df, raw_feature_cols, cluster_col, image_col, initial_cluster_mode)
    if len(df) == 0:
        raise RuntimeError('构建样本后没有可用的数据，请检查输入')
    if dropped_samples:
        print(f"有 {len(dropped_samples)} 个样本被跳过: {dropped_samples[:10]} ...")

    # 初始使用UMAP降维，同时计算2D和3D版本
    df, initial_reduction_key = ensure_dimensionality_reduction(df, feature_cols, algorithm='umap', n_components=2)
    # 预计算3D降维结果，确保首次选择3D时有数据可用
    df, _ = ensure_dimensionality_reduction(df, feature_cols, algorithm='umap', n_components=3)

    app = dash.Dash(__name__)

    # 添加Flask路由来提供原图
    @app.server.route('/get_full_image/<path:filename>')
    def get_full_image(filename):
        """动态提供原图"""
        from flask import jsonify
        try:
            print(f"正在查找图片: {filename}")  # 调试日志
            
            # 在所有可能的目录中查找图片
            search_paths = [
                IMAGE_ROOT,  # 使用已定义的IMAGE_ROOT常量
                Path(__file__).parent / "all_cutouts",
                Path(__file__).parent / "all_kmeans_new"
            ]
            
            for search_path in search_paths:
                if Path(search_path).exists():
                    print(f"搜索路径: {search_path}")  # 调试日志
                    # 在该目录及其子目录中查找文件
                    for root, dirs, files in os.walk(search_path):
                        if filename in files:
                            filepath = Path(root) / filename
                            print(f"找到图片: {filepath}")  # 调试日志
                            # 生成高分辨率图片
                            full_image_b64 = img_to_base64_full(filepath)
                            if full_image_b64:
                                print(f"图片编码成功")  # 调试日志
                                return jsonify({'status': 'success', 'image': full_image_b64})
                            else:
                                print(f"图片编码失败")  # 调试日志
            
            print(f"图片未找到: {filename}")  # 调试日志
            return jsonify({'status': 'error', 'message': 'Image not found'}), 404
        except Exception as e:
            print(f"Flask路由错误: {e}")  # 调试日志
            return jsonify({'status': 'error', 'message': str(e)}), 500

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
    part_symbol_col, part_symbol_map = get_part_symbol_settings(df)
    scatter_kwargs = {
        'x': f'{initial_reduction_key}_0',
        'y': f'{initial_reduction_key}_1',
        'color': df[cluster_col].astype(str),
        'hover_data': hover_cols,
        'custom_data': custom,
        'color_discrete_sequence': CLUSTER_COLORS
    }
    if part_symbol_col:
        scatter_kwargs.update({
            'symbol': part_symbol_col,
            'symbol_map': part_symbol_map,
            'symbol_sequence': PART_SYMBOL_SEQUENCE
        })

    fig = px.scatter(df, **scatter_kwargs)

    # 准备降维算法选项
    algorithm_options = [
        {'label': 't-SNE', 'value': 'tsne'},
        {'label': 'UMAP', 'value': 'umap'}
    ]
    
    # cluster_metadata 已在上面加载
    
    # 准备可视化类型选项
    visualization_types = [
        {'label': '降维散点图', 'value': 'scatter'},
        {'label': '聚类特征热力图', 'value': 'heatmap'},
        {'label': '聚类相似度矩阵', 'value': 'similarity'}
    ]
    
    app.layout = build_layout(
        fig=fig,
        clusters=clusters,
        init_unit_options=init_unit_options,
        init_part_options=init_part_options,
        init_type_options=init_type_options,
        algorithm_options=algorithm_options,
        initial_cluster_mode=initial_cluster_mode,
        cluster_metadata=cluster_metadata,
        df=df,
        feature_cols=feature_cols,
        raw_feature_cols=raw_feature_cols,
        cluster_col=cluster_col,
        image_col=image_col,
    )

    # Register modularized callbacks
    register_compare_callbacks(app)
    register_cluster_panel_callbacks(app)

    # 客户端回调：处理图片点击显示模态框
    app.clientside_callback(
        """
        function(n1, n2) {
            console.log('=== CLIENTSIDE CALLBACK LOADED ===');
            
            // 图片变换状态
            let imageTransform = {
                scale: 1,
                rotation: 0,
                translateX: 0,
                translateY: 0
            };
            
            let isDragging = false;
            let lastX = 0;
            let lastY = 0;
            
            function updateImageTransform() {
                const modalImg = document.getElementById('modal-image');
                if (modalImg) {
                    // 只使用 transform 属性进行变换，不改变图片的基础尺寸
                    modalImg.style.transform = `scale(${imageTransform.scale}) rotate(${imageTransform.rotation}deg) translate(${imageTransform.translateX}px, ${imageTransform.translateY}px)`;
                }
            }
            
            function resetImageTransform() {
                imageTransform = {scale: 1, rotation: 0, translateX: 0, translateY: 0};
                updateImageTransform();
            }
            
            // 添加全局事件监听器
            if (!window.imageModalInitialized) {
                console.log('=== INITIALIZING IMAGE MODAL ===');
                
                document.addEventListener('click', function(e) {
                    console.log('=== CLICK DETECTED ===', e.target.tagName, e.target.id);
                    
                    // 阻止按钮点击事件冒泡
                    if (e.target.id && (e.target.id.includes('btn') || e.target.id === 'close-modal-btn')) {
                        e.stopPropagation();
                        
                        if (e.target.id === 'zoom-in-btn') {
                            imageTransform.scale *= 1.2;
                            updateImageTransform();
                        } else if (e.target.id === 'zoom-out-btn') {
                            imageTransform.scale /= 1.2;
                            if (imageTransform.scale < 0.1) imageTransform.scale = 0.1;
                            updateImageTransform();
                        } else if (e.target.id === 'rotate-left-btn') {
                            imageTransform.rotation -= 90;
                            updateImageTransform();
                        } else if (e.target.id === 'rotate-right-btn') {
                            imageTransform.rotation += 90;
                            updateImageTransform();
                        } else if (e.target.id === 'reset-btn') {
                            resetImageTransform();
                        } else if (e.target.id === 'close-modal-btn') {
                            document.getElementById('image-modal').style.display = 'none';
                        }
                        return;
                    }
                    
                    // 处理模态框关闭（点击背景）
                    if (e.target.id === 'image-modal') {
                        console.log('=== CLOSING MODAL (BACKDROP) ===');
                        document.getElementById('image-modal').style.display = 'none';
                        resetImageTransform();
                        return;
                    }
                    
                    // 处理图片点击打开模态框
                    if (e.target.tagName === 'IMG' && e.target.src && e.target.src.startsWith('data:image/')) {
                        // 排除模态框内的图片
                        if (e.target.id === 'modal-image') return;
                        
                        console.log('=== IMAGE CLICKED ===');
                        e.preventDefault();
                        e.stopPropagation();
                        
                        const modal = document.getElementById('image-modal');
                        const modalImg = document.getElementById('modal-image');
                        
                        if (modal && modalImg) {
                            // 获取原始图片路径和原图数据
                            const imagePath = e.target.getAttribute('data-image-path');
                            const fullImageData = e.target.getAttribute('data-full-src');
                            console.log('=== IMAGE PATH ===', imagePath);
                            console.log('=== HAS FULL IMAGE DATA ===', fullImageData ? 'YES' : 'NO');
                            
                            if (fullImageData) {
                                // 使用嵌入的原图数据
                                modalImg.src = fullImageData;
                                console.log('=== USING EMBEDDED FULL IMAGE ===', fullImageData.substring(0, 50));
                            } else if (imagePath) {
                                // 对于侧栏缩略图，动态请求原图
                                console.log('=== REQUESTING FULL IMAGE FOR ===', imagePath);
                                
                                // 先显示缩略图
                                modalImg.src = e.target.src;
                                
                                // 异步请求原图
                                fetch(`/get_full_image/${encodeURIComponent(imagePath)}`)
                                    .then(response => response.json())
                                    .then(data => {
                                        if (data.status === 'success' && data.image) {
                                            modalImg.src = data.image;
                                            console.log('=== FULL IMAGE LOADED ===');
                                        } else {
                                            console.log('=== FULL IMAGE REQUEST FAILED ===', data.message);
                                        }
                                    })
                                    .catch(error => {
                                        console.log('=== FULL IMAGE REQUEST ERROR ===', error);
                                    });
                            } else {
                                // 如果没有路径信息，直接使用缩略图
                                modalImg.src = e.target.src;
                            }
                            
                            modal.style.display = 'block';
                            resetImageTransform(); // 重置变换状态
                            console.log('=== MODAL OPENED ===');
                        }
                    }
                });
                
                // 图片拖拽功能
                document.addEventListener('mousedown', function(e) {
                    if (e.target.id === 'modal-image') {
                        isDragging = true;
                        lastX = e.clientX;
                        lastY = e.clientY;
                        e.preventDefault();
                    }
                });
                
                document.addEventListener('mousemove', function(e) {
                    if (isDragging) {
                        const deltaX = e.clientX - lastX;
                        const deltaY = e.clientY - lastY;
                        
                        imageTransform.translateX += deltaX / imageTransform.scale;
                        imageTransform.translateY += deltaY / imageTransform.scale;
                        
                        updateImageTransform();
                        
                        lastX = e.clientX;
                        lastY = e.clientY;
                    }
                });
                
                document.addEventListener('mouseup', function(e) {
                    isDragging = false;
                });
                
                // 滚轮缩放功能
                document.addEventListener('wheel', function(e) {
                    if (e.target.id === 'modal-image' || e.target.closest('#image-modal')) {
                        e.preventDefault();
                        
                        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
                        imageTransform.scale *= zoomFactor;
                        
                        if (imageTransform.scale < 0.1) imageTransform.scale = 0.1;
                        if (imageTransform.scale > 10) imageTransform.scale = 10;
                        
                        updateImageTransform();
                    }
                });
                
                // ESC键关闭模态框
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') {
                        console.log('=== CLOSING MODAL (ESC) ===');
                        const modal = document.getElementById('image-modal');
                        if (modal && modal.style.display === 'block') {
                            modal.style.display = 'none';
                            resetImageTransform();
                        }
                    }
                });
                
                window.imageModalInitialized = true;
                console.log('=== IMAGE MODAL INITIALIZED ===');
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('image-click-trigger', 'children'),
        [Input('sample-panel', 'children'),
         Input('cluster-panel', 'children')]
    )

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

            const has3d = figure.data && figure.data.some(t => t.type === 'scatter3d');
            // 3D 下不做透明度变更，直接保持现有 figure，避免触发相机重置
            if (has3d) {
                return window.dash_clientside.no_update;
            }

            const hovered_cluster = hover_state ? hover_state.hovered_cluster : null;
            const new_figure = JSON.parse(JSON.stringify(figure));

            if (new_figure.data) {
                new_figure.data.forEach(trace => {
                    if (hovered_cluster !== null && hovered_cluster !== undefined) {
                        // Plotly 在 color+symbol 情况下 trace.name 可能为 "cluster,part"，因此用前缀匹配
                        const name = trace.name || '';
                        const clusterMatch = name === String(hovered_cluster) || name.startsWith(String(hovered_cluster) + ',');
                        trace.opacity = clusterMatch ? 1.0 : 0.2;
                    } else {
                        // 没有悬停时保持默认透明度
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
         Output('data-store', 'data'),
         Output('sample-cluster-mapping', 'data'),
         Output('cluster-filter', 'options')],
        [Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value'),
         Input('algorithm-selector', 'value'),
         Input('dimension-selector', 'value'),
         Input('z-axis-selector', 'value'),

         Input('reload-trigger', 'data')],
        State('data-store', 'data')
    )
    def update_plot(selected_clusters, selected_units, selected_parts, selected_types, 
                   selected_algorithm, selected_dimension, 
                   z_axis='dimension',
                   reload_trigger=0,
                   data_store=None):
        # 从data-store获取数据
        if data_store is None:
            raise ValueError("Data store is empty")
        
        # 先解析当前的data_store以获取信息
        current_feature_cols = data_store['feature_cols']
        raw_feature_cols = data_store.get('raw_feature_cols', current_feature_cols)
        old_cluster_mode = data_store.get('cluster_mode', 'merged')
        
        # 检查是否需要重新加载数据
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reload-trigger.data' and reload_trigger > 0:
            # 重新加载CSV数据
            print(f"触发数据重新加载 (trigger={reload_trigger})")
            
            # 读取新的聚类元数据以获取聚类模式
            new_metadata = load_cluster_metadata()
            new_cluster_mode = new_metadata.get('cluster_mode', 'merged') if new_metadata else 'merged'
            print(f"新的聚类模式: {new_cluster_mode}")
            
            # 读取新的聚类表格
            df_new = pd.read_csv(csv)
            
            # 重新检测列名
            new_cluster_col, new_image_col = detect_columns(df_new)
            if new_cluster_col is None or new_image_col is None:
                raise RuntimeError('无法识别聚类列或图片列，请检查 CSV')
            
            # 添加sample_id如果不存在
            df_new = ensure_sample_ids(df_new, new_image_col)
            
            # 检测原始特征列（128维）
            exclude = {new_cluster_col, new_image_col, 'image_name', 'sample_id', 'side', 'image_id', 'sherd_id', 
                      'unit', 'part', 'type', 'image_side', 'image_id_original', 'unit_C', 'part_C', 'type_C', 'image_path'}
            new_raw_feature_cols = [c for c in df_new.columns if c not in exclude and np.issubdtype(df_new[c].dtype, np.number)]
            
            # 根据新的聚类模式重新构建样本数据
            df_processed, new_feature_cols, _ = build_samples_for_mode(
                df_new, new_raw_feature_cols, new_cluster_col, new_image_col, new_cluster_mode
            )
            
            print(f"重新加载数据: {len(df_processed)} 条记录")
            print(f"新的聚类数量: {df_processed[new_cluster_col].nunique()}")
            print(f"特征维度: {len(new_feature_cols)}")
            
            # 重新构建data_store
            data_store = {
                'df': df_processed.to_json(orient='split'),
                'feature_cols': new_feature_cols,
                'raw_feature_cols': new_raw_feature_cols,
                'cluster_col': new_cluster_col,
                'image_col': new_image_col,
                'cluster_mode': new_cluster_mode,
                'version': reload_trigger  # 添加版本号
            }
        
        # 确保selected_algorithm不是None
        if selected_algorithm is None:
            selected_algorithm = 'umap'
        
        # 解析数据（暂时去除缓存机制以确保功能正常）
        df = pd.read_json(StringIO(data_store['df']), orient='split')
        feature_cols = data_store['feature_cols']
        cluster_col = data_store['cluster_col']
        image_col = data_store['image_col']
        
        # 使用高效的降维计算（具有内置缓存）
        if selected_algorithm == 'tsne':
            df, reduction_key = ensure_dimensionality_reduction(df, feature_cols, 
                                                           algorithm=selected_algorithm, 
                                                           n_components=selected_dimension,
                                                           perplexity=30)
        elif selected_algorithm == 'umap':
            df, reduction_key = ensure_dimensionality_reduction(df, feature_cols, 
                                                           algorithm=selected_algorithm, 
                                                           n_components=selected_dimension,
                                                           n_neighbors=15,
                                                           min_dist=0.1)
        else:
            df, reduction_key = ensure_dimensionality_reduction(df, feature_cols, 
                                                           algorithm=selected_algorithm, 
                                                           n_components=selected_dimension)
        
        part_symbol_col, part_symbol_map = get_part_symbol_settings(df)
        symbol_kwargs = {}
        if part_symbol_col:
            symbol_kwargs = {
                'symbol': part_symbol_col,
                'symbol_map': part_symbol_map,
                'symbol_sequence': PART_SYMBOL_SEQUENCE
            }

        # 3D图在部分 Plotly 版本对 symbol_sequence 支持不稳定，单独分支
        symbol_kwargs_3d = {}  # 先禁用3D形状，确保渲染稳定

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
                           color_discrete_sequence=CLUSTER_COLORS,
                           **symbol_kwargs)
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
                                  color_discrete_sequence=CLUSTER_COLORS,
                                  **symbol_kwargs_3d)
                
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
                                                                      perplexity=30 if selected_algorithm == 'tsne' else None,
                                                                      n_neighbors=15 if selected_algorithm == 'umap' else None,
                                                                      min_dist=0.1 if selected_algorithm == 'umap' else None)
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
                                  color_discrete_sequence=CLUSTER_COLORS,
                                  **symbol_kwargs_3d)
                # 直接设置点大小
                fig.update_traces(marker={'size': 1})
        
        # 根据维度设置不同的点大小（保持2D的设置）
        if selected_dimension == 2:
            fig.update_traces(marker={'size': 8})

        # 保持用户交互状态（如3D相机角度），避免每次回调重置
        fig.update_layout(uirevision='tsne-plot')
        
        # 保存当前使用的降维参数
        params = data_store.get('params', {})
        
        # 为当前算法和参数组合保存参数
        if selected_algorithm == 'tsne':
            params[reduction_key] = {'perplexity': 30}
        elif selected_algorithm == 'umap':
            params[reduction_key] = {'n_neighbors': 15, 'min_dist': 0.1}
        
        # 更新data-store
        updated_data_store = {
            'df': df.to_json(orient='split'),
            'feature_cols': feature_cols,
            'cluster_col': cluster_col,
            'image_col': image_col,
            'params': params
        }
        
        # 更新sample-cluster-mapping
        sample_cluster_mapping = df.set_index('sample_id')[cluster_col].to_dict()
        
        # 更新cluster-filter的选项
        clusters = sorted(df[cluster_col].dropna().unique())
        cluster_options = [{'label': str(int(c)), 'value': int(c)} for c in clusters]
        
        return fig, updated_data_store, sample_cluster_mapping, cluster_options


    @app.callback(
        Output('cluster-images-store', 'data'),
        Output('last-selected-store', 'data'),
        Output('sample-panel', 'children'),
        Output('selected-meta', 'children'),
        Input('tsne-plot', 'clickData'),
        State('data-store', 'data')
    )
    def show_selected(clickData, data_store=None):
        # when no click, clear cluster panel
        if not clickData or data_store is None:
            return [], {}, html.Div('点击一个点以查看图片'), ''

        pts = clickData.get('points', [])
        if len(pts) == 0:
            return [], {}, html.Div('点击一个点以查看图片'), ''

        base_root = Path(__file__).parent
        image_root = (base_root / IMAGE_ROOT) if not Path(IMAGE_ROOT).is_absolute() else Path(IMAGE_ROOT)

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
            for algo in ['tsne', 'umap']:
                if f'{algo}_0' in df.columns and f'{algo}_1' in df.columns:
                    row_candidate = df[(np.isclose(df[f'{algo}_0'], x)) & (np.isclose(df[f'{algo}_1'], y))]
                    if len(row_candidate) > 0:
                        row = row_candidate.iloc[0]
                        break
        if row is None:
            return [], {}, html.Div('未找到对应记录'), ''

        sample_id = sample_id or row.get('sample_id')
        paired_images = paired_images or row.get('paired_images')

        # determine sample id (main id) to find both sides
        if sample_id is None:
            name = str(row['image_name'])
            sample_id = Path(name).stem.replace('_exterior', '').replace('_interior', '')

        # collect images for this sample (both sides) — always include当前点击图像优先
        paired_names = []
        def side_label(name: str):
            low = str(name).lower()
            if 'interior' in low:
                return '内侧'
            if 'exterior' in low:
                return '外侧'
            return '未知侧'

        current_img = None
        if img_name:
            current_img = str(img_name)
            paired_names.append(current_img)
        elif row.get(image_col):
            current_img = str(row.get(image_col))
            paired_names.append(current_img)

        if len(paired_names) < 2:
            base_key = None
            if sample_id:
                base_key = str(sample_id).replace('_exterior', '').replace('_interior', '')
            elif img_name:
                base_key = Path(str(img_name)).stem
                base_key = base_key.replace('_exterior', '').replace('_interior', '')

            if base_key:
                candidates = []
                # 优先使用 image_name 列匹配
                if 'image_name' in df.columns:
                    candidates = df[df['image_name'].str.contains(base_key, na=False)]['image_name'].tolist()
                # 退回 sample_id 列匹配
                if len(candidates) < 2 and 'sample_id' in df.columns:
                    candidates = candidates + df[df['sample_id'].str.contains(base_key, na=False)]['image_name'].tolist()

                candidates = list({Path(c).name for c in candidates})  # 去重保留文件名
                if len(candidates) > 1:
                    paired_names.extend(candidates)

        base_ext = Path(str(img_name)).suffix or '.png'

        def resolve_image_path(name, fallback_dir):
            """尽量找到图片路径：优先给定目录，其次 all_kmeans_new 下的同名文件，再退回 all_cutouts。"""
            name = Path(str(name)).name

            fb_dir = Path(fallback_dir)
            if not fb_dir.is_absolute():
                fb_dir = base_root / fb_dir

            def search_with_name(candidate_name: str):
                candidate = fb_dir / candidate_name
                if candidate.exists():
                    return candidate
                cluster_root = base_root / 'all_kmeans_new'
                for root, _, files in os.walk(cluster_root):
                    if candidate_name in files:
                        return Path(root) / candidate_name
                cutout_path = base_root / 'all_cutouts' / candidate_name
                if cutout_path.exists():
                    return cutout_path
                return None

            # 1) 尝试原名
            primary = search_with_name(name)
            if primary:
                return primary

            # 2) 若无扩展名，尝试常见图片扩展
            if Path(name).suffix == '':
                # 尝试直接加扩展
                for ext in [base_ext, '.png', '.jpg', '.jpeg']:
                    alt = search_with_name(f"{name}{ext}")
                    if alt:
                        return alt
                # 若名字中无侧面标记，则尝试补 _exterior/_interior
                if '_exterior' not in name and '_interior' not in name:
                    for side in ['_exterior', '_interior']:
                        for ext in [base_ext, '.png', '.jpg', '.jpeg']:
                            alt = search_with_name(f"{name}{side}{ext}")
                            if alt:
                                return alt
            return fb_dir / name  # 返回一个可预测路径以便后续检测

        # 优先使用表里的 image_path（build_table 生成），否则用 image_col；若仍是文件名则退回图像根目录
        base_dir = image_root
        if 'image_path' in row and pd.notna(row['image_path']):
            base_dir = Path(row['image_path']).parent
        else:
            candidate = Path(str(row[image_col]))
            if candidate.parent != candidate:
                base_dir = candidate.parent
        print(f"[INFO] sample base_dir={base_dir}, image_root={image_root}, img_name={img_name}")

        # 如果仍然不足两张，尝试补另一侧
        if len(paired_names) < 2 and current_img:
            stem = Path(current_img).stem
            suffix = Path(current_img).suffix or base_ext
            alt_candidates = []
            if 'interior' in stem:
                alt_candidates.append(stem.replace('interior', 'exterior') + suffix)
            elif 'exterior' in stem:
                alt_candidates.append(stem.replace('exterior', 'interior') + suffix)
            else:
                alt_candidates.append(f"{stem}_interior{suffix}")
                alt_candidates.append(f"{stem}_exterior{suffix}")

            for alt in alt_candidates:
                alt_name = Path(alt).name
                if alt_name in paired_names:
                    continue
                ipath = resolve_image_path(alt_name, base_dir)
                if ipath.exists():
                    paired_names.append(alt_name)

        # 去重但保持当前点击图像优先
        seen = set()
        ordered = []
        for nm in paired_names:
            key = Path(str(nm)).name
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        paired_names = ordered
        sample_imgs = []
        for i, nm in enumerate(paired_names):
            ipath = resolve_image_path(nm, base_dir)
            if not ipath.exists():
                print(f"[WARN] resolved path missing: {ipath} for nm={nm}")
            b64 = img_to_base64(ipath)
            b64_full = img_to_base64_full(ipath)  # 同时生成原图
            if b64:
                side_txt = side_label(nm)
                sample_imgs.append(html.Div([
                    html.Img(
                        src=b64, 
                        id=f'sample-img-{i}',
                        **{'data-image-path': str(Path(nm).name)},  # 添加原始图片路径
                        **{'data-full-src': b64_full if b64_full else b64},  # 添加原图数据
                        style={
                            'height': '200px', 
                            'border': '1px solid #ccc', 
                            'marginRight': '6px',
                            'cursor': 'pointer'
                        },
                        title='点击放大查看'
                    ),
                    html.Div(side_txt, style={'textAlign': 'center', 'fontSize': '12px', 'color': '#555', 'marginTop': '4px'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}) )
            else:
                print(f"[WARN] failed to encode image: {ipath}")
                sample_imgs.append(html.Div(f"缺少图片或无法读取: {ipath.name}"))

        if len(sample_imgs) == 0:
            sample_imgs = [html.Div('未找到正反面图片')]

        left_col = html.Div(sample_imgs, style={'display': 'flex', 'gap': '8px', 'alignItems': 'flex-start', 'flexWrap': 'wrap'})

        # collect all images in same cluster
        cluster_val = row[cluster_col]
        same_cluster = df[df[cluster_col] == cluster_val]
        image_paths = []
        for _, r in same_cluster.iterrows():
            base_dir = image_root
            if 'image_path' in r and pd.notna(r['image_path']):
                base_dir = Path(r['image_path']).parent
            else:
                candidate = Path(str(r[image_col]))
                if candidate.parent != candidate:
                    base_dir = candidate.parent
            names = []
            if 'paired_images' in r and pd.notna(r['paired_images']):
                names = [n for n in str(r['paired_images']).split(';') if n]
            elif 'image_name' in r:
                names = [str(r['image_name'])]
            for nm in names:
                ipath = resolve_image_path(nm, base_dir)
                if not ipath.exists():
                    print(f"[WARN] cluster thumb path missing: {ipath} for nm={nm} base_dir={base_dir}")
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
        
        meta = html.Div(meta_parts, style={'fontSize': '13px', 'lineHeight': '1.6', 'color': '#333'})

        # 准备用于对比的记录（单次点击存储，不直接加入列表）
        rep_name = None
        if 'image_name' in row:
            rep_name = row['image_name']
        elif paired_images:
            rep_name = paired_images[0]
        rep_path = None
        if rep_name:
            base_dir = image_root
            if 'image_path' in row and pd.notna(row['image_path']):
                base_dir = Path(row['image_path']).parent
            else:
                candidate = Path(str(row[image_col]))
                if candidate.parent != candidate:
                    base_dir = candidate.parent
            pth = resolve_image_path(rep_name, base_dir)
            if not pth.exists():
                print(f"[WARN] compare path missing: {pth} for rep_name={rep_name}")
            rep_path = str(pth) if pth else None
        last_selected = {
            'sample_id': str(sample_id),
            'cluster': str(cluster_val),
            'name': str(rep_name) if rep_name else '未知图像',
            'path': rep_path or ''
        }

        sample_panel_children = html.Div([
            html.Div(left_col, style={'flex': '1 1 60%', 'minWidth': '280px'}),
            html.Div(meta, style={'flex': '1 1 40%', 'minWidth': '220px', 'paddingLeft': '8px'})
        ], style={'display': 'flex', 'gap': '12px', 'alignItems': 'flex-start'})

        return image_paths, last_selected, sample_panel_children, ''



    # 重新聚类回调
    @app.callback(
        [Output('recluster-status', 'children'),
         Output('reload-trigger', 'data')],
        Input('recluster-button', 'n_clicks'),
        [State('n-clusters-input', 'value'),
         State('cluster-mode-selector', 'value'),
         State('cluster-algorithm-selector', 'value'),
         State('reload-trigger', 'data')]
    )
    def perform_reclustering(n_clicks, n_clusters, cluster_mode, cluster_algorithm, current_trigger):
        if n_clicks == 0 or n_clicks is None:
            return '', dash.no_update
        
        try:
            cluster_algorithm = cluster_algorithm or 'kmeans'

            # 调用聚类函数
            if cluster_algorithm == 'kmeans':
                clustering_result = perform_kmeans_clustering(
                    features_csv_path=FEATURES_CSV,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode
                )
            elif cluster_algorithm.startswith('agglomerative'):
                _, _, linkage = cluster_algorithm.partition('-')
                linkage = linkage or 'ward'
                clustering_result = perform_agglomerative_clustering(
                    features_csv_path=FEATURES_CSV,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode,
                    linkage=linkage
                )
            elif cluster_algorithm.startswith('spectral'):
                _, _, assign_labels = cluster_algorithm.partition('-')
                assign_labels = assign_labels or 'kmeans'
                clustering_result = perform_spectral_clustering(
                    features_csv_path=FEATURES_CSV,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode,
                    assign_labels=assign_labels
                )
            elif cluster_algorithm == 'leiden':
                # Leiden 基于分辨率参数决定簇数，此处使用默认参数
                clustering_result = perform_leiden_clustering(
                    features_csv_path=FEATURES_CSV,
                    cluster_mode=cluster_mode
                )
            else:
                raise ValueError(f"不支持的聚类算法: {cluster_algorithm}")
            
            # 提取结果
            labels = clustering_result['labels']
            cluster_centers = clustering_result['cluster_centers']
            piece_ids = clustering_result['piece_ids']
            silhouette_avg = clustering_result['silhouette_score']
            selected_df = clustering_result['selected_df']  # 包含 filename 和 main_id 的 DataFrame
            algo_name = clustering_result.get('algorithm', cluster_algorithm)
            
            # 保存聚类结果到文件（类似 kmeans_DINO.py 的输出）
            import json
            from pathlib import Path
            import shutil
            
            output_dir = Path(__file__).parent / 'all_kmeans_new'
            
            # 清空旧的聚类目录
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # 保存元数据（包含聚类模式）
            metadata = {
                'n_clusters': int(clustering_result['n_clusters']),
                'cluster_centers': cluster_centers.tolist(),
                'silhouette_score': float(silhouette_avg),
                'cluster_mode': cluster_mode,  # 保存聚类模式
                'algorithm': algo_name
            }
            
            with open(output_dir / 'cluster_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 创建 piece_id -> cluster 映射
            piece_to_cluster = {pid: int(label) for pid, label in zip(piece_ids, labels)}
            
            # 为每个聚类创建目录并复制图片
            image_root = Path(__file__).parent / 'all_cutouts'
            cluster_file_map = {}  # filename -> cluster_id
            
            for idx, row in selected_df.iterrows():
                main_id = row['main_id']
                filename = row['filename']
                
                if main_id not in piece_to_cluster:
                    continue
                
                cluster_id = piece_to_cluster[main_id]
                cluster_dir = output_dir / f'cluster_{cluster_id}'
                cluster_dir.mkdir(exist_ok=True)
                
                # 复制图片文件
                src_path = image_root / filename
                if src_path.exists():
                    dst_path = cluster_dir / filename
                    shutil.copy2(src_path, dst_path)
                    cluster_file_map[filename] = cluster_id
            
            print(f"已复制 {len(cluster_file_map)} 个图片文件到聚类目录")
            
            # 重新运行 build_table.py 的逻辑来生成表格
            import subprocess
            result = subprocess.run(
                [str(Path(__file__).parent / 'venv' / 'Scripts' / 'python.exe'), 
                 str(Path(__file__).parent / 'build_table.py')],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent)
            )
            
            if result.returncode != 0:
                print(f"build_table.py 执行失败: {result.stderr}")
                raise RuntimeError(f"重新生成表格失败: {result.stderr}")
            
            print(f"build_table.py 执行成功: {result.stdout}")
            
            # 聚类模式中文名称
            mode_names = {'merged': '融合', 'exterior': '仅外部', 'interior': '仅内部'}
            mode_display = mode_names.get(cluster_mode, cluster_mode)

            algo_display = {
                'kmeans': 'K-Means',
                'agglomerative-ward': '层次(ward)',
                'spectral-kmeans': '谱聚类',
                'leiden': 'Leiden (kNN 图)'
            }
            status = f'✓ 聚类完成! 算法={algo_display.get(cluster_algorithm, algo_name)}, 模式={mode_display}, K={clustering_result["n_clusters"]}, 轮廓系数={silhouette_avg:.3f}'
            
            success_msg = html.Div([
                html.Span(status, style={'color': 'green', 'fontWeight': 'bold'}),
                html.Br(),
                html.Span('数据已自动重新加载，新的聚类结果现在可见。', style={'marginTop': '10px', 'color': '#28a745'})
            ])
            
            # 触发数据重新加载
            new_trigger = (current_trigger or 0) + 1
            return success_msg, new_trigger
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"聚类错误: {error_details}")
            error_msg = html.Div(f'✗ 聚类失败: {str(e)}', style={'color': 'red', 'fontWeight': 'bold'})
            return error_msg, dash.no_update


    # 簇规模分布图
    @app.callback(
        Output('cluster-size-graph', 'figure'),
        [Input('visualization-tabs', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_cluster_size(tab_value, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'cluster-size' or data_store is None:
            return dash.no_update

        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        if len(dff) == 0 or cluster_col not in dff.columns:
            empty_fig = px.bar(title='暂无数据')
            empty_fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))
            return empty_fig

        counts = dff[cluster_col].value_counts().sort_index()
        plot_df = counts.reset_index()
        plot_df.columns = ['cluster', 'count']
        plot_df['cluster_label'] = plot_df['cluster'].astype(str)

        def to_int_or_index(lbl, fallback_idx):
            try:
                return int(float(lbl))
            except Exception:
                return fallback_idx

        color_map = {}
        for i, lbl in enumerate(plot_df['cluster_label']):
            color_idx = to_int_or_index(lbl, i) % len(CLUSTER_COLORS)
            color_map[lbl] = CLUSTER_COLORS[color_idx]

        total = int(counts.sum())
        max_count = int(counts.max()) if len(counts) > 0 else 0
        max_ratio = max_count / total if total > 0 else 0
        sorted_counts = counts.sort_values()
        half = max(1, len(sorted_counts) // 2)
        tail_share = sorted_counts.head(half).sum() / total if total > 0 else 0

        fig = px.bar(
            plot_df,
            x='cluster_label',
            y='count',
            text='count',
            color='cluster_label',
            color_discrete_map=color_map
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            title=f"簇规模分布｜样本 {len(dff)}，簇 {len(counts)}｜最大簇占比 {max_ratio:.2%}｜长尾占比 {tail_share:.2%}",
            xaxis_title='簇 ID',
            yaxis_title='样本数',
            bargap=0.3,
            showlegend=False,
            margin=dict(l=40, r=30, t=60, b=80)
        )
        return fig


    # 聚类质量指标卡
    @app.callback(
        Output('cluster-quality-cards', 'children'),
        Output('cluster-quality-bars', 'figure'),
        Output('cluster-quality-detail', 'children'),
        [Input('visualization-tabs', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_cluster_quality(tab_value, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'cluster-quality' or data_store is None:
            return dash.no_update, dash.no_update, dash.no_update

        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']
        feature_cols = data_store.get('feature_cols', [])

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        if not feature_cols or cluster_col not in dff.columns or len(dff) < 3:
            empty = html.Div('暂无足够数据计算指标', style={'color': '#666', 'padding': '8px'})
            return empty, dash.no_update, dash.no_update

        dff = dff.dropna(subset=feature_cols)
        if len(dff) < 3:
            empty = html.Div('样本过少，无法计算指标', style={'color': '#666', 'padding': '8px'})
            return empty, dash.no_update, dash.no_update

        X = dff[feature_cols].values
        labels = dff[cluster_col].values

        if len(np.unique(labels)) < 2:
            empty = html.Div('簇数不足 2，无法计算指标', style={'color': '#666', 'padding': '8px'})
            return empty, dash.no_update, dash.no_update

        max_samples = 3000
        if len(X) > max_samples:
            sample_idx = np.random.default_rng(42).choice(len(X), size=max_samples, replace=False)
            X = X[sample_idx]
            labels = labels[sample_idx]

        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        def safe_metric(fn, default=np.nan):
            try:
                return float(fn(X, labels))
            except Exception:
                return default

        sil = safe_metric(silhouette_score)
        ch = safe_metric(calinski_harabasz_score)
        db = safe_metric(davies_bouldin_score)

        def card(title, value, hint):
            txt = '无法计算' if np.isnan(value) else f"{value:.4f}"
            return html.Div([
                html.Div(title, style={'fontSize': '13px', 'color': '#666', 'marginBottom': '6px'}),
                html.Div(txt, style={'fontSize': '22px', 'fontWeight': '600'}),
                html.Div(hint, style={'fontSize': '12px', 'color': '#888', 'marginTop': '4px'})
            ], style={
                'padding': '12px 14px',
                'border': '1px solid #e0e0e0',
                'borderRadius': '8px',
                'minWidth': '180px',
                'backgroundColor': '#fafafa'
            })

        cards = [
            card('Silhouette', sil, '越接近 1 越好'),
            card('Calinski-Harabasz', ch, '越大越好'),
            card('Davies-Bouldin', db, '越低越好')
        ]

        summary = html.Div(
            f"样本 {len(X)}｜簇 {len(np.unique(labels))}",
            style={'fontSize': '13px', 'color': '#555', 'marginBottom': '8px'}
        )

        # 细分：每簇轮廓、簇内平均距、簇间最近距
        compact_df = pd.DataFrame({
            'cluster': dff[cluster_col],
            'size': dff.groupby(cluster_col)[cluster_col].transform('count')
        })

        # silhouette per sample -> per cluster mean
        from sklearn.metrics import silhouette_samples
        sil_per_cluster = {}
        try:
            max_samples_detail = 4000
            X_detail = X
            labels_detail = labels
            if len(X_detail) > max_samples_detail:
                idx = np.random.default_rng(42).choice(len(X_detail), size=max_samples_detail, replace=False)
                X_detail = X_detail[idx]
                labels_detail = labels_detail[idx]
            sil_samples = silhouette_samples(X_detail, labels_detail, metric='euclidean')
            for cid in np.unique(labels_detail):
                mask = labels_detail == cid
                if np.any(mask):
                    sil_per_cluster[cid] = float(np.mean(sil_samples[mask]))
        except Exception:
            sil_per_cluster = {}

        # cluster centers and inter distances
        centers_df = dff.groupby(cluster_col)[feature_cols].mean()
        centers = centers_df.values
        center_ids = centers_df.index.to_numpy()
        inter_min = {}
        if len(center_ids) > 1:
            diff = centers[:, None, :] - centers[None, :, :]
            dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
            for i, cid in enumerate(center_ids):
                mask = np.ones(len(center_ids), dtype=bool)
                mask[i] = False
                inter_min[cid] = float(np.min(dist_mat[i][mask])) if np.any(mask) else np.nan

        intra_mean = {}
        for cid, group in dff.groupby(cluster_col):
            if len(group) == 0:
                intra_mean[cid] = np.nan
                continue
            center_vec = group[feature_cols].mean().values
            distances = np.linalg.norm(group[feature_cols].values - center_vec, axis=1)
            intra_mean[cid] = float(np.mean(distances))

        records = []
        for cid in sorted(dff[cluster_col].unique()):
            records.append({
                'cluster': cid,
                'size': int((dff[cluster_col] == cid).sum()),
                'silhouette': sil_per_cluster.get(cid, np.nan),
                'intra_mean': intra_mean.get(cid, np.nan),
                'inter_min': inter_min.get(cid, np.nan)
            })

        detail_df = pd.DataFrame(records)
        detail_df['cluster_label'] = detail_df['cluster'].astype(str)
        detail_df['looseness'] = detail_df['intra_mean'] / (detail_df['inter_min'] + 1e-8)

        # 视觉状态：松散/紧凑
        def status_color(looseness, sil):
            if pd.isna(looseness):
                return '#cccccc'
            if looseness < 0.3 and (pd.isna(sil) or sil >= 0.2):
                return '#4caf50'  # green, clear
            if looseness < 0.6 or (not pd.isna(sil) and sil >= 0.0):
                return '#ffb300'  # amber, watch
            return '#e53935'      # red, mixed

        detail_df['status_color'] = detail_df.apply(lambda r: status_color(r['looseness'], r['silhouette']), axis=1)

        plot_df = detail_df.sort_values('looseness', ascending=False)
        bar_fig = px.bar(
            plot_df,
            x='cluster_label',
            y='looseness',
            text='looseness',
            color='status_color',
            color_discrete_map='identity',
            labels={'cluster_label': '簇', 'looseness': '松散度（越低越紧凑）'},
            title='簇紧凑度/黏连风险（颜色：绿=清晰，黄=需关注，红=混杂）'
        )
        bar_fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        bar_fig.update_layout(margin=dict(l=40, r=30, t=60, b=80), showlegend=False)

        # detail table
        def fmt_val(v, digits=3):
            return '-' if pd.isna(v) else f"{v:.{digits}f}"

        table_rows = []
        header = html.Tr([
            html.Th('簇'), html.Th('规模'), html.Th('簇内均距'), html.Th('最近簇距'), html.Th('轮廓系数'), html.Th('松散度比')
        ])
        for _, row in detail_df.sort_values('looseness', ascending=False).iterrows():
            color = row['status_color'] if pd.notna(row['status_color']) else '#cccccc'
            table_rows.append(html.Tr([
                html.Td(str(row['cluster'])),
                html.Td(str(int(row['size']))),
                html.Td(fmt_val(row['intra_mean'])),
                html.Td(fmt_val(row['inter_min'])),
                html.Td(fmt_val(row['silhouette'])),
                html.Td(fmt_val(row['looseness']))
            ], style={'backgroundColor': '#fdfdfd', 'borderLeft': f'6px solid {color}'}))

        detail_table = html.Table([
            html.Thead(header),
            html.Tbody(table_rows)
        ], style={'borderCollapse': 'collapse', 'width': '100%', 'marginTop': '6px'})

        detail_hint = html.Div(
            '颜色含义：绿=簇紧凑且与邻簇分开；黄=轻微分散或稍粘连；红=分散或与邻簇混杂。松散度比 = 簇内平均距离 / 最近簇中心距离，越低越清晰。',
            style={'color': '#666', 'marginTop': '4px'}
        )

        return [summary] + cards, bar_fig, html.Div([detail_hint, detail_table])


    # 类别构成图（按簇堆叠）
    @app.callback(
        Output('category-breakdown-graph', 'figure'),
        [Input('visualization-tabs', 'value'),
         Input('category-field-selector', 'value'),
         Input('category-x-axis', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_category_breakdown(tab_value, category_field, x_axis_mode, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'category-breakdown' or data_store is None:
            return dash.no_update

        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']

        if category_field not in df.columns:
            fig = px.bar(title='所选类别字段不存在')
            fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))
            return fig

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        dff = dff[pd.notna(dff[category_field])]

        if len(dff) == 0 or cluster_col not in dff.columns:
            empty_fig = px.bar(title='暂无数据')
            empty_fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))
            return empty_fig

        x_axis_mode = x_axis_mode or 'cluster'
        x_field = cluster_col if x_axis_mode == 'cluster' else 'unit_C'

        if x_field not in dff.columns:
            fig = px.bar(title='所选横轴字段不存在')
            fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))
            return fig

        grouped = (
            dff
            .groupby([x_field, category_field])
            .size()
            .reset_index(name='count')
        )

        grouped['x_label'] = grouped[x_field].astype(str)
        grouped = grouped.sort_values([x_field, category_field])

        fig = px.bar(
            grouped,
            x='x_label',
            y='count',
            color=category_field,
            text='count',
            barmode='stack'
        )
        fig.update_traces(textposition='outside', cliponaxis=False)
        fig.update_layout(
            title=f"类别构成（{category_field}）｜样本 {len(dff)}",
            xaxis_title='簇' if x_axis_mode == 'cluster' else '单位 (unit_C)',
            yaxis_title='样本数',
            bargap=0.25,
            margin=dict(l=40, r=30, t=60, b=80),
            legend_title=category_field
        )
        return fig


    # 簇质量与纯度表 + 特征差异 Top-K
    @app.callback(
        [Output('cluster-quality-table', 'children'),
         Output('feature-diff-graph', 'figure'),
         Output('analysis-cluster-selector', 'options'),
         Output('analysis-cluster-selector', 'value')],
        [Input('visualization-tabs', 'value'),
         Input('analysis-cluster-selector', 'value'),
         Input('feature-diff-mode', 'value'),
         Input('feature-topk-slider', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def render_cluster_analysis(tab_value, selected_cluster, diff_mode, topk, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'cluster-analysis' or data_store is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']
        feature_cols = data_store.get('feature_cols', [])

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        if cluster_col not in dff.columns or len(dff) == 0:
            empty_fig = px.bar(title='暂无数据')
            return html.Div('暂无数据'), empty_fig, [], None

        clusters = sorted(dff[cluster_col].dropna().unique())
        options = [{'label': str(c), 'value': c} for c in clusters]
        if selected_cluster not in clusters:
            selected_cluster = clusters[0] if clusters else None

        # 纯度字段：优先 part_C, 其次 type_C, unit_C
        purity_field = None
        for cand in ['part_C', 'type_C', 'unit_C']:
            if cand in dff.columns and dff[cand].notna().any():
                purity_field = cand
                break

        # 计算簇规模
        size_series = dff[cluster_col].value_counts().sort_index()

        # 计算簇内纯度
        purity_data = {}
        if purity_field:
            grp = dff[[cluster_col, purity_field]].dropna().groupby(cluster_col)[purity_field]
            for cid, series in grp:
                vc = series.value_counts(normalize=True)
                purity = float(vc.iloc[0]) if len(vc) > 0 else np.nan
                top_label = str(vc.index[0]) if len(vc) > 0 else ''
                purity_data[cid] = (purity, top_label)
        else:
            purity_data = {cid: (np.nan, '') for cid in clusters}

        # 计算每个样本的轮廓系数，再聚合到簇
        sil_means = {cid: np.nan for cid in clusters}
        if feature_cols and len(feature_cols) > 1 and len(dff) >= 3 and len(clusters) >= 2:
            try:
                work = dff.dropna(subset=feature_cols)
                X = work[feature_cols].values
                labels = work[cluster_col].values
                if len(np.unique(labels)) >= 2 and len(X) >= 3:
                    from sklearn.metrics import silhouette_samples
                    # 控制规模
                    max_samples = 4000
                    if len(X) > max_samples:
                        idx = np.random.default_rng(42).choice(len(X), size=max_samples, replace=False)
                        X = X[idx]
                        labels = labels[idx]
                    sil_samples = silhouette_samples(X, labels, metric='euclidean')
                    for cid in np.unique(labels):
                        mask = labels == cid
                        if np.any(mask):
                            sil_means[cid] = float(np.mean(sil_samples[mask]))
            except Exception:
                pass

        # 生成表格数据
        rows = []
        for cid in clusters:
            size = int(size_series.get(cid, 0))
            purity, top_lbl = purity_data.get(cid, (np.nan, ''))
            sil = sil_means.get(cid, np.nan)
            rows.append((cid, size, purity, top_lbl, sil))

        rows.sort(key=lambda x: -x[1])

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.3f}" if not np.isnan(v) else '-'
            return str(v)

        table = html.Table([
            html.Thead(html.Tr([
                html.Th('簇'), html.Th('规模'), html.Th('纯度'), html.Th('主类别'), html.Th('簇内轮廓')
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(str(cid)),
                    html.Td(size),
                    html.Td(fmt(purity)),
                    html.Td(top_lbl),
                    html.Td(fmt(sil))
                ]) for cid, size, purity, top_lbl, sil in rows
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse'})

        # 特征差异 top-k（簇中心 vs 全局均值）
        topk = int(topk or 5)
        topk = max(3, min(30, topk))
        feat_fig = px.bar(title='特征差异')
        if feature_cols and selected_cluster is not None:
            try:
                cluster_center = dff[dff[cluster_col] == selected_cluster][feature_cols].mean().values
                global_center = dff[feature_cols].mean().values
                if diff_mode == 'zscore':
                    global_std = dff[feature_cols].std(ddof=0).replace(0, np.nan).values
                    diff = (cluster_center - global_center) / (global_std + 1e-8)
                    title_mode = 'z-score'
                else:
                    diff = cluster_center - global_center
                    title_mode = '均值差'
                abs_diff = np.abs(diff)
                idx = np.argsort(abs_diff)[-topk:][::-1]
                data = {
                    'feature': [feature_cols[i] for i in idx],
                    'delta': [float(diff[i]) for i in idx]
                }
                feat_fig = px.bar(data, x='feature', y='delta', title=f"簇 {selected_cluster} 特征差异 Top-{topk}（{title_mode}）")
                feat_fig.update_layout(margin=dict(l=40, r=30, t=60, b=120))
                feat_fig.update_traces(marker_color='#3366cc')
            except Exception:
                feat_fig = px.bar(title='特征差异计算失败')

        return table, feat_fig, options, selected_cluster


    # 代表样本网格（按簇显示）
    @app.callback(
        Output('representative-grid', 'children'),
        Output('outlier-list', 'children'),
        [Input('visualization-tabs', 'value'),
         Input('rep-samples-per-cluster', 'value'),
         Input('rep-strategy', 'value'),
         Input('outlier-count', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    def render_representatives(tab_value, samples_per_cluster, strategy, outlier_count, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'representatives' or data_store is None:
            return dash.no_update, dash.no_update

        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']
        image_col = data_store['image_col']
        feature_cols = data_store.get('feature_cols', [])

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        if cluster_col not in dff.columns or len(dff) == 0:
            empty_div = html.Div('暂无数据', style={'color': '#666', 'padding': '8px'})
            return empty_div, empty_div

        clusters = sorted(dff[cluster_col].dropna().unique())
        if len(clusters) == 0:
            empty_div = html.Div('暂无数据', style={'color': '#666', 'padding': '8px'})
            return empty_div, empty_div

        n_per = int(samples_per_cluster or 1)
        n_per = max(1, min(12, n_per))
        outlier_k = int(outlier_count or 1)
        outlier_k = max(1, min(5, outlier_k))

        max_total = 200
        if len(clusters) * n_per > max_total:
            n_per = max(1, max_total // len(clusters))

        base_root = Path(__file__).parent
        image_root = base_root / IMAGE_ROOT if not Path(IMAGE_ROOT).is_absolute() else Path(IMAGE_ROOT)

        def resolve_path(val: str):
            p = Path(str(val))
            if not p.is_absolute():
                p = image_root / p
            if p.exists():
                return p
            alt = base_root / 'all_cutouts' / p.name
            if alt.exists():
                return alt
            alt2 = base_root / 'all_kmeans_new' / p.name
            if alt2.exists():
                return alt2
            return p

        cards = []
        outlier_blocks = []
        thumb_size = 120
        for c in clusters:
            subset_all = dff[dff[cluster_col] == c]
            subset_feat = subset_all.dropna(subset=feature_cols) if feature_cols else subset_all

            # Representative selection
            chosen = subset_all
            if strategy == 'center' and feature_cols and len(subset_feat) > 0:
                center_vec = subset_feat[feature_cols].mean().values
                distances = np.linalg.norm(subset_feat[feature_cols].values - center_vec, axis=1)
                subset_feat = subset_feat.assign(_dist=distances)
                chosen = subset_feat.nsmallest(n_per, '_dist')
            elif strategy == 'random':
                chosen = subset_all.sample(n=min(n_per, len(subset_all)), random_state=42) if len(subset_all) > 0 else subset_all
            else:
                chosen = subset_all.head(n_per)

            if len(chosen) < n_per and len(subset_all) > len(chosen):
                extra = subset_all.drop(chosen.index, errors='ignore').head(n_per - len(chosen))
                chosen = pd.concat([chosen, extra])

            thumbs = []
            for _, row in chosen.head(n_per).iterrows():
                img_val = row.get('image_name') if 'image_name' in row else row.get(image_col)
                path = resolve_path(img_val)
                cache_key = f"rep_thumb_{Path(path).name}_{thumb_size}"
                b64 = image_cache.get(cache_key) if image_cache else None
                if b64 is None:
                    b64 = img_to_base64(path, max_size=thumb_size)
                    if image_cache and b64:
                        image_cache.set(cache_key, b64)
                if b64:
                    thumbs.append(html.Img(
                        src=b64,
                        style={'height': f'{thumb_size}px', 'border': '1px solid #ddd', 'borderRadius': '4px', 'backgroundColor': '#fafafa'},
                        **{'data-image-path': Path(path).name},
                        title=str(img_val)
                    ))
                else:
                    thumbs.append(html.Div(str(Path(path).name), style={'fontSize': '12px', 'color': '#999'}))

            if len(thumbs) == 0:
                thumbs.append(html.Div('无可用图片', style={'fontSize': '12px', 'color': '#999'}))

            cards.append(html.Div([
                html.Div(f"簇 {c}", style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '6px'}),
                html.Div(thumbs, style={'display': 'flex', 'gap': '6px', 'flexWrap': 'wrap'})
            ], style={
                'padding': '10px',
                'border': '1px solid #e0e0e0',
                'borderRadius': '8px',
                'minWidth': '180px',
                'backgroundColor': '#fff'
            }))

            # Outlier list (farthest from center)
            if feature_cols and len(subset_feat) > 0:
                center_vec = subset_feat[feature_cols].mean().values
                distances = np.linalg.norm(subset_feat[feature_cols].values - center_vec, axis=1)
                subset_feat = subset_feat.assign(_dist=distances)
                outliers = subset_feat.nlargest(outlier_k, '_dist')
                items = []
                for _, r in outliers.iterrows():
                    img_val = r.get('image_name') if 'image_name' in r else r.get(image_col)
                    path = resolve_path(img_val)
                    cache_key = f"outlier_thumb_{Path(path).name}_{thumb_size}"
                    b64 = image_cache.get(cache_key) if image_cache else None
                    if b64 is None:
                        b64 = img_to_base64(path, max_size=thumb_size)
                        if image_cache and b64:
                            image_cache.set(cache_key, b64)
                    label_text = f"样本 {r.get('sample_id', img_val)}｜距离 {r['_dist']:.3f}"
                    thumb = html.Img(src=b64, style={'height': '60px', 'border': '1px solid #ddd', 'borderRadius': '4px', 'marginRight': '6px'}) if b64 else None
                    items.append(html.Li([
                        thumb if thumb else html.Span(str(Path(path).name), style={'marginRight': '6px'}),
                        html.Span(label_text)
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '4px'}))
                outlier_blocks.append(html.Div([
                    html.Div(f"簇 {c} 离群样本", style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '4px'}),
                    html.Ul(items, style={'paddingLeft': '16px', 'marginTop': '0', 'marginBottom': '8px'})
                ], style={'marginBottom': '8px'}))

        if len(outlier_blocks) == 0:
            outlier_blocks = html.Div('缺少特征列，无法计算离群样本', style={'color': '#666', 'padding': '4px'})

        return cards, outlier_blocks


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


    # 聚类相似度/距离矩阵（基于当前过滤后的簇中心）
    @app.callback(
        Output('similarity-graph', 'figure'),
        Output('nearest-cluster-list', 'children'),
        [Input('visualization-tabs', 'value'),
         Input('similarity-metric', 'value'),
         Input('similarity-options', 'value'),
         Input('similarity-neighbor-k', 'value'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('data-store', 'data')
    )
    @cache_plot_result
    def update_similarity_matrix(tab_value, metric, options, neighbor_k, selected_clusters, selected_units, selected_parts, selected_types, data_store):
        if tab_value != 'similarity' or data_store is None:
            return dash.no_update, dash.no_update

        metric = metric or 'cosine'
        options = options or []
        annotate = 'annotate' in options
        reorder_requested = 'reorder' in options
        neighbor_k = int(neighbor_k or 3)

        df = pd.read_json(StringIO(data_store['df']), orient='split')
        cluster_col = data_store['cluster_col']
        feature_cols = data_store.get('feature_cols', [])

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        if cluster_col not in dff.columns or not feature_cols:
            fig = px.imshow([[0]], title='缺少簇列或特征列')
            return fig, ""

        # 仅保留特征完整的样本
        dff = dff.dropna(subset=feature_cols)
        if len(dff) == 0:
            fig = px.imshow([[0]], title='暂无数据')
            return fig, ""

        # 计算每簇中心
        centers_df = dff.groupby(cluster_col)[feature_cols].mean()
        clusters = centers_df.index.to_numpy()
        centers = centers_df.values

        if centers.shape[0] == 0:
            fig = px.imshow([[0]], title='暂无簇')
            return fig, ""

        # 计算矩阵
        if metric == 'euclidean':
            diff = centers[:, None, :] - centers[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            mat = dist
            neighbor_matrix = dist
            neighbor_is_distance = True
            title = f"簇中心距离矩阵｜簇 {len(clusters)}"
            color_scale = 'Viridis'
            zmin = None
            zmax = None
        else:
            norm = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
            normed = centers / norm
            sim = normed @ normed.T
            mat = sim
            neighbor_matrix = sim
            neighbor_is_distance = False
            title = f"簇中心相似度矩阵｜簇 {len(clusters)}"
            color_scale = 'RdBu'
            zmin = -1
            zmax = 1

        labels = np.array([str(c) for c in clusters])

        # 层次重排（仅在 SciPy 可用时启用）
        reordered = False
        if reorder_requested and SCIPY_AVAILABLE and len(labels) > 1:
            try:
                if neighbor_is_distance:
                    dist_mat = neighbor_matrix
                else:
                    sim01 = (neighbor_matrix + 1) / 2
                    dist_mat = 1 - sim01
                condensed = squareform(dist_mat, checks=False)
                order = leaves_list(linkage(condensed, method='average'))
                mat = mat[np.ix_(order, order)]
                neighbor_matrix = neighbor_matrix[np.ix_(order, order)]
                labels = labels[order]
                reordered = True
            except Exception:
                reordered = False

        fig = px.imshow(
            mat,
            x=labels,
            y=labels,
            color_continuous_scale=color_scale,
            zmin=zmin,
            zmax=zmax,
            labels={'x': '簇', 'y': '簇', 'color': '值'}
        )
        title_suffix = '（已重排）' if reordered else ''
        fig.update_layout(
            title=f"{title}{title_suffix}",
            margin=dict(l=40, r=30, t=60, b=60)
        )
        fig.update_xaxes(side='top')
        fig.update_yaxes(autorange='reversed')

        if annotate:
            text = np.round(mat, 3)
            fig.update_traces(text=text, texttemplate="%{text}")

        # 最近邻簇列表
        if neighbor_matrix.shape[0] > 1:
            k = max(1, min(neighbor_k, neighbor_matrix.shape[0] - 1))
            nearest_children = []
            for i, cid in enumerate(labels):
                if neighbor_is_distance:
                    order = np.argsort(neighbor_matrix[i])
                    nearest_idx = [idx for idx in order if idx != i][:k]
                    neighbors = [f"{labels[j]}（距离 {neighbor_matrix[i][j]:.3f}）" for j in nearest_idx]
                else:
                    order = np.argsort(-neighbor_matrix[i])
                    nearest_idx = [idx for idx in order if idx != i][:k]
                    neighbors = [f"{labels[j]}（相似度 {neighbor_matrix[i][j]:.3f}）" for j in nearest_idx]
                nearest_children.append(html.Li(f"簇 {cid}: " + ", ".join(neighbors)))
            nearest_list = html.Ul(nearest_children)
        else:
            nearest_list = ""

        return fig, nearest_list

    # 添加原图加载回调
    @app.callback(
        Output('modal-image', 'src'),
        [Input('image-path-input', 'value')],
        prevent_initial_call=True
    )
    def load_full_image(image_path):
        """加载原图用于模态框显示"""
        if not image_path or image_path == '':
            return dash.no_update
        
        try:
            # 构建完整的图片路径
            full_path = Path(IMAGE_ROOT) / image_path
            if full_path.exists():
                full_res_image = img_to_base64_full(str(full_path))
                print(f"✅ 加载原图成功: {image_path}")
                return full_res_image
            else:
                print(f"❌ 图片文件不存在: {full_path}")
            return dash.no_update
        except Exception as e:
            print(f"❌ 加载原图失败: {e}")
            return dash.no_update

    # 性能监控回调已移除
    
    return app


def main():
    """主函数 - 应用入口点"""
    try:
        # 从环境变量获取配置
        port = int(os.environ.get('CERAMIC_PORT', APP_CONFIG['port']))
        debug = os.environ.get('CERAMIC_DEBUG', 'false').lower() == 'true'
        
        print(f"🚀 启动 {APP_CONFIG['title']}...")
        print(f"📊 正在加载数据...")
        
        app = create_app()
        
        print(f"✅ 应用已准备就绪!")
        print(f"🌐 访问地址: http://127.0.0.1:{port}")
        print(f"💡 提示: 按 Ctrl+C 停止服务")
        
        # 运行应用
        app.run(
            debug=debug, 
            port=port, 
            host='127.0.0.1'
        )
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保数据文件存在于正确的位置")
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        print("请检查配置和依赖项")
        raise

if __name__ == '__main__':
    main()
