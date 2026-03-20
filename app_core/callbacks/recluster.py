"""
Reclustering callback extracted from the main app module.
负责在前端触发重新聚类并刷新结果。
"""

import json
import subprocess
from pathlib import Path

import dash
from dash import Input, Output, State, html

from data_processing import (
    perform_agglomerative_clustering,
    perform_kmeans_clustering,
    perform_leiden_clustering,
    perform_spectral_clustering,
)


def register_recluster_callbacks(app, *, features_csv, image_root):
    """注册重新聚类回调。"""

    @app.callback(
        [Output('recluster-status', 'children'),
         Output('reload-trigger', 'data')],
        Input('recluster-button', 'n_clicks'),
        [State('n-clusters-input', 'value'),
         State('cluster-mode-selector', 'value'),
         State('cluster-algorithm-selector', 'value'),
         State('pca-components-input', 'value'),
         State('reload-trigger', 'data')]
    )
    def perform_reclustering(n_clicks, n_clusters, cluster_mode, cluster_algorithm, pca_components, current_trigger):
        """执行聚类算法并写回簇目录与元数据。

        Returns:
            tuple[str, int | NoUpdate]: 状态提示文本与刷新触发计数。
        """
        if n_clicks == 0 or n_clicks is None:
            return '', dash.no_update

        try:
            cluster_algorithm = cluster_algorithm or 'kmeans'
            pca_comp = int(pca_components) if pca_components else None
            if pca_comp == 0:
                pca_comp = None

            if cluster_algorithm == 'kmeans':
                clustering_result = perform_kmeans_clustering(
                    features_csv_path=features_csv,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode,
                    pca_components=pca_comp,
                )
            elif cluster_algorithm.startswith('agglomerative'):
                _, _, linkage = cluster_algorithm.partition('-')
                linkage = linkage or 'ward'
                clustering_result = perform_agglomerative_clustering(
                    features_csv_path=features_csv,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode,
                    linkage=linkage,
                    pca_components=pca_comp,
                )
            elif cluster_algorithm.startswith('spectral'):
                _, _, assign_labels = cluster_algorithm.partition('-')
                assign_labels = assign_labels or 'kmeans'
                clustering_result = perform_spectral_clustering(
                    features_csv_path=features_csv,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode,
                    assign_labels=assign_labels,
                    pca_components=pca_comp,
                )
            elif cluster_algorithm == 'leiden':
                clustering_result = perform_leiden_clustering(
                    features_csv_path=features_csv,
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
                ['python', str(Path(__file__).parent.parent.parent / 'scripts' / 'build_table.py')],
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
            status = f'✓ 聚类完成! 算法={algo_display.get(cluster_algorithm, algo_name)}, 模式={mode_display}, K={clustering_result["n_clusters"]}{pca_display}, 轮廓系数={silhouette_avg:.3f}'

            success_msg = html.Div([
                html.Span(status, style={'color': 'green', 'fontWeight': 'bold'}),
                html.Br(),
                html.Span('数据已自动重新加载，新的聚类结果现在可见。', style={'marginTop': '10px', 'color': '#28a745'})
            ])

            new_trigger = (current_trigger or 0) + 1
            return success_msg, new_trigger

        except Exception as exc:
            import traceback

            error_details = traceback.format_exc()
            print(f"聚类错误: {error_details}")
            error_msg = html.Div(f'✗ 聚类失败: {str(exc)}', style={'color': 'red', 'fontWeight': 'bold'})
            return error_msg, dash.no_update

    return app