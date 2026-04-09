"""
Reclustering callback extracted from the main app module.
负责在前端触发重新聚类并刷新结果。
"""

import json
import subprocess
import sys
from pathlib import Path

import dash
from performance_utils import plot_cache
from dash import Input, Output, State, html

from data_processing import (
    count_clustering_samples,
    load_scope_reference,
    perform_agglomerative_clustering,
    perform_kmeans_clustering,
    perform_leiden_clustering,
    perform_spectral_clustering,
)


def register_recluster_callbacks(app, *, features_csv, image_root):
    """注册重新聚类回调。"""

    def _normalize_scope_value(value):
        """统一范围筛选值的比较形式。"""
        return str(value).strip()

    def _piece_id_from_name(value):
        """从文件名或样本名提取陶片主编号。"""
        stem = Path(str(value)).stem
        return stem.replace('_exterior', '').replace('_interior', '').lower()

    def _filter_features_by_scope(features_df, scoped_df):
        """按范围内的陶片主编号过滤特征表，保留同一陶片的正反面。"""
        work = features_df.copy()
        if 'filename' not in work.columns:
            raise ValueError("特征CSV缺少 filename 列，无法按范围过滤")

        scoped_piece_ids = set()

        if 'sample_id' in scoped_df.columns:
            scoped_piece_ids.update(
                scoped_df['sample_id'].dropna().astype(str).str.strip().str.lower()
            )

        if 'image_name' in scoped_df.columns:
            scoped_piece_ids.update(
                scoped_df['image_name'].dropna().astype(str).map(_piece_id_from_name)
            )

        if 'piece_id' in scoped_df.columns:
            scoped_piece_ids.update(
                scoped_df['piece_id'].dropna().astype(str).str.strip().str.lower()
            )

        scoped_piece_ids = {pid for pid in scoped_piece_ids if pid}
        if not scoped_piece_ids:
            return work.iloc[0:0].copy()

        work['_piece_id'] = work['filename'].astype(str).map(_piece_id_from_name)
        filtered = work[work['_piece_id'].isin(scoped_piece_ids)].copy()
        return filtered.drop(columns=['_piece_id'], errors='ignore')

    # ── 初始化聚类范围选项 ──────────────────────────────────────────────────
    @app.callback(
        Output('cluster-scope-unit', 'options'),
        Output('cluster-scope-part', 'options'),
        Output('cluster-scope-unit', 'value'),
        Output('cluster-scope-part', 'value'),
        Input('reload-trigger', 'data'),
        prevent_initial_call=False,
    )
    def init_scope_options(_):
        from app_core.data_cache import get_data_cache
        from app_core.callbacks.analytics.stratigraphy import _sorted_layers
        data_cache = get_data_cache()
        df = load_scope_reference()
        if df is None:
            df = data_cache['df']

        units = _sorted_layers([u for u in df['unit_C'].dropna().unique() if str(u).strip()])
        unit_opts = [{'label': str(v), 'value': v} for v in units]

        parts = sorted([p for p in df['part_C'].dropna().unique() if str(p).strip()])
        part_opts = [{'label': str(v), 'value': v} for v in parts]

        return unit_opts, part_opts, None, None

    @app.callback(
        [Output('recluster-status', 'children'),
         Output('reload-trigger', 'data'),
         Output('cluster-metadata-store', 'data'),
         Output('cluster-filter', 'value', allow_duplicate=True),
         Output('unit-filter', 'value', allow_duplicate=True),
         Output('part-filter', 'value', allow_duplicate=True),
         Output('type-filter', 'value', allow_duplicate=True)],
        Input('recluster-button', 'n_clicks'),
        [State('n-clusters-input', 'value'),
         State('cluster-mode-selector', 'value'),
         State('cluster-algorithm-selector', 'value'),
         State('pca-components-input', 'value'),
         State('stratified-clustering-checkbox', 'value'),
         State('cluster-scope-unit', 'value'),
         State('cluster-scope-part', 'value'),
         State('reload-trigger', 'data')],
        prevent_initial_call=True,
    )
    def perform_reclustering(n_clicks, n_clusters, cluster_mode, cluster_algorithm,
                             pca_components, stratified_value,
                             scope_unit, scope_part, current_trigger):
        """执行聚类算法并写回簇目录与元数据。

        Returns:
            tuple[str, int | NoUpdate]: 状态提示文本与刷新触发计数。
        """
        if n_clicks == 0 or n_clicks is None:
            return '', dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        try:
            cluster_algorithm = cluster_algorithm or 'kmeans'
            pca_comp = int(pca_components) if pca_components else None
            if pca_comp == 0:
                pca_comp = None
            # 检查是否启用分层聚类
            stratified_enabled = stratified_value and 'stratified' in stratified_value

            # ── 聚类范围过滤 ──────────────────────────────────────────────
            import pandas as pd
            base_dir = Path(__file__).parent.parent.parent
            data_csv = base_dir / 'sherd_cluster_table_clustered_only.csv'
            df_full = load_scope_reference()
            if df_full is None:
                df_full = pd.read_csv(data_csv)

            scope_filters = []
            if scope_unit:
                if 'unit_C' in df_full.columns:
                    target_unit = _normalize_scope_value(scope_unit)
                    df_full = df_full[df_full['unit_C'].astype(str).str.strip() == target_unit]
                    scope_filters.append(f'层={scope_unit}')
            if scope_part:
                if 'part_C' in df_full.columns:
                    target_part = _normalize_scope_value(scope_part)
                    df_full = df_full[df_full['part_C'].astype(str).str.strip() == target_part]
                    scope_filters.append(f'Part={scope_part}')

            if scope_filters and len(df_full) == 0:
                return html.Div(f'✗ 范围 [{", ".join(scope_filters)}] 内无数据',
                               style={'color': 'red', 'fontWeight': 'bold'}), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            scope_label = f'[{", ".join(scope_filters)}] ' if scope_filters else ''

            # 根据范围过滤特征数据
            features_df = pd.read_csv(features_csv)
            if scope_unit or scope_part:
                features_df = _filter_features_by_scope(features_df, df_full)
                if len(features_df) == 0:
                    return html.Div(f'✗ 范围 [{", ".join(scope_filters)}] 内无匹配特征数据',
                                   style={'color': 'red', 'fontWeight': 'bold'}), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

                # 写入临时特征文件
                scope_features_path = base_dir / 'temp_scope_features.csv'
                features_df.to_csv(scope_features_path, index=False)
                effective_features_csv = scope_features_path
            else:
                effective_features_csv = features_csv

            if stratified_enabled:
                # 分层聚类：对每个unit_C分别聚类
                import numpy as np

                if 'unit_C' not in df_full.columns:
                    return html.Div('✗ 数据中没有unit_C列，无法进行分层聚类',
                                   style={'color': 'red', 'fontWeight': 'bold'}), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

                # 读取（已过滤范围的）特征数据
                features_df = pd.read_csv(effective_features_csv)

                if 'sample_id' in df_full.columns:
                    unit_map = (
                        df_full[['sample_id', 'unit_C']]
                        .dropna(subset=['sample_id', 'unit_C'])
                        .assign(
                            sample_id=lambda frame: frame['sample_id'].astype(str).str.strip().str.lower(),
                            unit_C=lambda frame: frame['unit_C'].astype(str).str.strip(),
                        )
                        .drop_duplicates(subset=['sample_id'])
                    )
                    features_with_unit = features_df.copy()
                    features_with_unit['sample_id'] = features_with_unit['filename'].map(_piece_id_from_name)
                    features_with_unit = features_with_unit.merge(
                        unit_map,
                        on='sample_id',
                        how='left'
                    )
                else:
                    return html.Div('✗ 范围过滤后缺少 sample_id，无法匹配 unit_C 信息',
                                   style={'color': 'red', 'fontWeight': 'bold'}), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

                # 过滤掉unit_C为空的样本
                features_with_unit = features_with_unit.dropna(subset=['unit_C'])

                unit_sample_counts = count_clustering_samples(
                    features_with_unit,
                    cluster_mode=cluster_mode,
                    group_col='unit_C',
                )
                units = sorted(features_with_unit['unit_C'].unique())
                all_labels = []
                all_piece_ids = []
                all_centers = []
                unit_cluster_counts = {}
                skipped_units = []
                unit_silhouettes = {}

                for unit in units:
                    unit_mask = features_with_unit['unit_C'] == unit
                    unit_features = features_with_unit[unit_mask]
                    effective_unit_samples = int(unit_sample_counts.get(unit, 0))

                    # 根据平均聚类大小计算该地层的K值
                    avg_cluster_size = n_clusters  # 在分层聚类中，n_clusters表示平均聚类大小
                    unit_n_clusters = max(2, round(effective_unit_samples / avg_cluster_size))

                    # 跳过样本数不足的unit
                    min_samples = max(unit_n_clusters, 2)
                    if effective_unit_samples < min_samples:
                        skipped_units.append(f"{unit}({effective_unit_samples}片)")
                        continue

                    # 识别特征列（数值列，排除ID和元数据列）
                    exclude_cols = ['unit_C', 'sample_id', 'image_name', 'piece_id', 'cluster_id']
                    feature_cols = [c for c in unit_features.columns
                                   if c not in exclude_cols and pd.api.types.is_numeric_dtype(unit_features[c])]

                    if len(feature_cols) == 0:
                        skipped_units.append(f"{unit}(无特征)")
                        continue

                    # 识别ID列（filename、sample_id等）
                    id_col = None
                    for col in ['filename', 'sample_id', 'image_name', 'piece_id']:
                        if col in unit_features.columns:
                            id_col = col
                            break

                    # 为该unit创建临时特征文件，保留ID列和特征列
                    temp_features_path = base_dir / f'temp_features_{unit}.csv'
                    if id_col:
                        cols_to_save = [id_col] + feature_cols
                        unit_features[cols_to_save].to_csv(temp_features_path, index=False)
                    else:
                        # 如果没有ID列，使用第一列作为ID
                        unit_features[feature_cols].to_csv(temp_features_path, index=False)

                    # 对该unit进行聚类
                    try:
                        if cluster_algorithm == 'kmeans':
                            unit_result = perform_kmeans_clustering(
                                features_csv_path=temp_features_path,
                                n_clusters=unit_n_clusters,
                                cluster_mode=cluster_mode,
                                pca_components=pca_comp,
                            )
                        elif cluster_algorithm.startswith('agglomerative'):
                            _, _, linkage = cluster_algorithm.partition('-')
                            linkage = linkage or 'ward'
                            unit_result = perform_agglomerative_clustering(
                                features_csv_path=temp_features_path,
                                n_clusters=unit_n_clusters,
                                cluster_mode=cluster_mode,
                                linkage=linkage,
                                pca_components=pca_comp,
                            )
                        elif cluster_algorithm.startswith('spectral'):
                            _, _, assign_labels = cluster_algorithm.partition('-')
                            assign_labels = assign_labels or 'kmeans'
                            unit_result = perform_spectral_clustering(
                                features_csv_path=temp_features_path,
                                n_clusters=unit_n_clusters,
                                cluster_mode=cluster_mode,
                                assign_labels=assign_labels,
                                pca_components=pca_comp,
                            )
                        elif cluster_algorithm == 'leiden':
                            unit_result = perform_leiden_clustering(
                                features_csv_path=temp_features_path,
                                cluster_mode=cluster_mode,
                                pca_components=pca_comp,
                            )
                        else:
                            raise ValueError(f"不支持的聚类算法: {cluster_algorithm}")

                        # 为该unit的簇ID添加前缀
                        unit_labels = [f"{unit}_{label}" for label in unit_result['labels']]
                        all_labels.extend(unit_labels)
                        all_piece_ids.extend(unit_result['piece_ids'])
                        all_centers.append(unit_result['cluster_centers'])
                        unit_cluster_counts[unit] = unit_result['n_clusters']
                        unit_silhouettes[unit] = float(unit_result.get('silhouette_score', 0.0) or 0.0)

                    finally:
                        # 清理临时文件
                        temp_features_path.unlink(missing_ok=True)

                # 检查是否有成功聚类的unit
                if len(all_labels) == 0:
                    skip_msg = f"所有地层单位样本数不足（需要≥{max(n_clusters, 2)}片）" if skipped_units else "没有可聚类的数据"
                    return html.Div(f'✗ 分层聚类失败: {skip_msg}',
                                   style={'color': 'red', 'fontWeight': 'bold'}), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

                # 合并所有结果
                labels = np.array(all_labels)
                piece_ids = np.array(all_piece_ids)
                cluster_centers = np.vstack(all_centers) if all_centers else np.array([])

                # 计算整体轮廓系数（使用各unit轮廓系数的加权平均）
                total_samples = len(labels)
                weighted_silhouette = 0.0

                for unit in unit_cluster_counts.keys():
                    unit_mask = np.array([label.startswith(f"{unit}_") for label in labels])
                    unit_sample_count = unit_mask.sum()
                    if unit_sample_count > 0:
                        weighted_silhouette += unit_silhouettes.get(unit, 0.0) * unit_sample_count

                silhouette_avg = weighted_silhouette / total_samples if total_samples > 0 else 0.0

                # 计算唯一簇标签数量
                unique_labels = sorted(set(labels))

                clustering_result = {
                    'labels': labels,
                    'cluster_centers': cluster_centers,
                    'piece_ids': piece_ids,
                    'silhouette_score': silhouette_avg,
                    'n_clusters': len(unique_labels),
                    'algorithm': f'{cluster_algorithm} (分层)',
                    'selected_df': features_with_unit,
                    'stratified': True,
                    'unit_cluster_counts': unit_cluster_counts,
                    'skipped_units': skipped_units,
                }

            else:
                # 原有的全局聚类逻辑
                # 根据平均聚类大小计算K值
                total_samples = count_clustering_samples(
                    effective_features_csv,
                    cluster_mode=cluster_mode,
                )
                if total_samples < 2:
                    return html.Div('✗ 有效样本不足，无法完成聚类',
                                   style={'color': 'red', 'fontWeight': 'bold'}), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
                avg_cluster_size = n_clusters
                calculated_k = max(2, round(total_samples / avg_cluster_size))
                max_allowed_k = total_samples if total_samples <= 2 else total_samples - 1
                calculated_k = min(calculated_k, max_allowed_k)

                if cluster_algorithm == 'kmeans':
                    clustering_result = perform_kmeans_clustering(
                        features_csv_path=effective_features_csv,
                        n_clusters=calculated_k,
                        cluster_mode=cluster_mode,
                        pca_components=pca_comp,
                    )
                elif cluster_algorithm.startswith('agglomerative'):
                    _, _, linkage = cluster_algorithm.partition('-')
                    linkage = linkage or 'ward'
                    clustering_result = perform_agglomerative_clustering(
                        features_csv_path=effective_features_csv,
                        n_clusters=calculated_k,
                        cluster_mode=cluster_mode,
                        linkage=linkage,
                        pca_components=pca_comp,
                    )
                elif cluster_algorithm.startswith('spectral'):
                    _, _, assign_labels = cluster_algorithm.partition('-')
                    assign_labels = assign_labels or 'kmeans'
                    clustering_result = perform_spectral_clustering(
                        features_csv_path=effective_features_csv,
                        n_clusters=calculated_k,
                        cluster_mode=cluster_mode,
                        assign_labels=assign_labels,
                        pca_components=pca_comp,
                    )
                elif cluster_algorithm == 'leiden':
                    clustering_result = perform_leiden_clustering(
                        features_csv_path=effective_features_csv,
                        cluster_mode=cluster_mode,
                        pca_components=pca_comp,
                    )
                else:
                    raise ValueError(f"不支持的聚类算法: {cluster_algorithm}")

            labels = clustering_result['labels']
            cluster_centers = clustering_result['cluster_centers']
            piece_ids = clustering_result['piece_ids']
            silhouette_avg = clustering_result['silhouette_score']
            selected_df = clustering_result['selected_df']
            algo_name = clustering_result.get('algorithm', cluster_algorithm)

            output_dir = Path(__file__).parent.parent.parent / 'all_kmeans_new'
            output_dir.mkdir(exist_ok=True)

            # 对于分层聚类，标签是字符串；对于普通聚类，标签是整数
            if clustering_result.get('stratified'):
                piece_to_cluster = {str(pid): str(label) for pid, label in zip(piece_ids, labels)}
            else:
                piece_to_cluster = {str(pid): int(label) for pid, label in zip(piece_ids, labels)}

            metadata = {
                'n_clusters': int(clustering_result['n_clusters']),
                'cluster_centers': cluster_centers.tolist(),
                'silhouette_score': float(silhouette_avg),
                'cluster_mode': cluster_mode,
                'algorithm': algo_name,
                'piece_to_cluster': piece_to_cluster,
            }

            with open(output_dir / 'cluster_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            result = subprocess.run(
                [sys.executable, str(Path(__file__).parent.parent.parent / 'scripts' / 'build_table.py')],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent)
            )

            if result.returncode != 0:
                print(f"build_table.py 执行失败: {result.stderr}")
                raise RuntimeError(f"重新生成表格失败: {result.stderr}")

            print(f"build_table.py 执行成功: {result.stdout}")

            # 删除 UMAP 缓存，下次启动时重新计算新聚类的散点图
            umap_cache = Path(__file__).parent.parent.parent / 'umap_cache.npz'
            umap_cache.unlink(missing_ok=True)

            mode_names = {'merged': '融合', 'exterior': '仅外部', 'interior': '仅内部'}
            mode_display = mode_names.get(cluster_mode, cluster_mode)

            algo_display = {
                'kmeans': 'K-Means',
                'agglomerative-ward': '层次(ward)',
                'spectral-kmeans': '谱聚类',
                'leiden': 'Leiden (kNN 图)'
            }
            pca_display = f', PCA={pca_comp}维' if pca_comp else ''
            scope_display = f', 范围={", ".join(scope_filters)}' if scope_filters else ''
            status = f'✓ 聚类完成! 算法={algo_display.get(cluster_algorithm, algo_name)}, 模式={mode_display}, K={clustering_result["n_clusters"]}{pca_display}{scope_display}, 轮廓系数={silhouette_avg:.3f}'

            # 清理临时范围特征文件
            if scope_unit or scope_part:
                scope_features_path = base_dir / 'temp_scope_features.csv'
                scope_features_path.unlink(missing_ok=True)

            # 构建成功消息
            msg_parts = [
                html.Span(status, style={'color': 'green', 'fontWeight': 'bold'}),
                html.Br(),
                html.Span('数据已自动重新加载，新的聚类结果现在可见。', style={'marginTop': '10px', 'color': '#28a745'})
            ]

            # 如果是分层聚类且有跳过的单位，添加警告
            if clustering_result.get('stratified') and clustering_result.get('skipped_units'):
                skipped = clustering_result['skipped_units']
                msg_parts.extend([
                    html.Br(),
                    html.Span(f'⚠ 已跳过样本数不足的地层单位: {", ".join(skipped)}',
                             style={'marginTop': '8px', 'color': '#ff9800', 'fontSize': '12px'})
                ])

            success_msg = html.Div(msg_parts)

            new_trigger = (current_trigger or 0) + 1
            # 清除图表缓存，确保所有分析页面显示新聚类数据
            plot_cache.clear()
            # 把新 metadata（不含 piece_to_cluster，体积太大）写入客户端 Store
            store_metadata = {k: v for k, v in metadata.items() if k != 'piece_to_cluster'}
            scoped_unit_filter = [scope_unit] if scope_unit else None
            scoped_part_filter = [scope_part] if scope_part else None
            return (
                success_msg,
                new_trigger,
                store_metadata,
                [],
                scoped_unit_filter,
                scoped_part_filter,
                [],
            )

        except Exception as exc:
            import traceback

            error_details = traceback.format_exc()
            print(f"聚类错误: {error_details}")
            error_msg = html.Div(f'✗ 聚类失败: {str(exc)}', style={'color': 'red', 'fontWeight': 'bold'})
            return error_msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # 算法切换时动态显示/隐藏 K 输入框（Leiden 不需要 K）
    app.clientside_callback(
        """
        function(algorithm) {
            var needsK = ['kmeans', 'agglomerative-ward', 'spectral-kmeans'];
            var show = needsK.indexOf(algorithm) !== -1;
            return show ? {display: 'flex', alignItems: 'center', gap: '6px', padding: '0 4px'}
                        : {display: 'none'};
        }
        """,
        Output('n-clusters-group', 'style'),
        Input('cluster-algorithm-selector', 'value'),
    )

    return app
