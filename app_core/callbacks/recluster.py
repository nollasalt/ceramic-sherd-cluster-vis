"""Reclustering callback extracted from the main app module."""

import json
import shutil
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
    """Register reclustering callback."""

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

            if cluster_algorithm == 'kmeans':
                clustering_result = perform_kmeans_clustering(
                    features_csv_path=features_csv,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode
                )
            elif cluster_algorithm.startswith('agglomerative'):
                _, _, linkage = cluster_algorithm.partition('-')
                linkage = linkage or 'ward'
                clustering_result = perform_agglomerative_clustering(
                    features_csv_path=features_csv,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode,
                    linkage=linkage
                )
            elif cluster_algorithm.startswith('spectral'):
                _, _, assign_labels = cluster_algorithm.partition('-')
                assign_labels = assign_labels or 'kmeans'
                clustering_result = perform_spectral_clustering(
                    features_csv_path=features_csv,
                    n_clusters=n_clusters,
                    cluster_mode=cluster_mode,
                    assign_labels=assign_labels
                )
            elif cluster_algorithm == 'leiden':
                clustering_result = perform_leiden_clustering(
                    features_csv_path=features_csv,
                    cluster_mode=cluster_mode
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

            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(exist_ok=True)

            metadata = {
                'n_clusters': int(clustering_result['n_clusters']),
                'cluster_centers': cluster_centers.tolist(),
                'silhouette_score': float(silhouette_avg),
                'cluster_mode': cluster_mode,
                'algorithm': algo_name
            }

            with open(output_dir / 'cluster_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            piece_to_cluster = {pid: int(label) for pid, label in zip(piece_ids, labels)}

            image_root_abs = Path(image_root)
            if not image_root_abs.is_absolute():
                image_root_abs = Path(__file__).parent.parent.parent / image_root_abs
            cluster_file_map = {}

            for _, row in selected_df.iterrows():
                main_id = row['main_id']
                filename = row['filename']

                if main_id not in piece_to_cluster:
                    continue

                cluster_id = piece_to_cluster[main_id]
                cluster_dir = output_dir / f'cluster_{cluster_id}'
                cluster_dir.mkdir(exist_ok=True)

                src_path = image_root_abs / filename
                if src_path.exists():
                    dst_path = cluster_dir / filename
                    shutil.copy2(src_path, dst_path)
                    cluster_file_map[filename] = cluster_id

            print(f"已复制 {len(cluster_file_map)} 个图片文件到聚类目录")

            result = subprocess.run(
                ['python', str(Path(__file__).parent.parent.parent / 'build_table.py')],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent)
            )

            if result.returncode != 0:
                print(f"build_table.py 执行失败: {result.stderr}")
                raise RuntimeError(f"重新生成表格失败: {result.stderr}")

            print(f"build_table.py 执行成功: {result.stdout}")

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

            new_trigger = (current_trigger or 0) + 1
            return success_msg, new_trigger

        except Exception as exc:
            import traceback

            error_details = traceback.format_exc()
            print(f"聚类错误: {error_details}")
            error_msg = html.Div(f'✗ 聚类失败: {str(exc)}', style={'color': 'red', 'fontWeight': 'bold'})
            return error_msg, dash.no_update

    return app