"""
分析回调包。将原 analytics.py 按功能拆分为子模块。
外部导入路径不变：from app_core.callbacks.analytics import register_analytics_callbacks
"""
from pathlib import Path

from .cluster_size import register_cluster_size_callbacks
from .cluster_quality import register_cluster_quality_callbacks
from .category_breakdown import register_category_breakdown_callbacks
from .cluster_analysis import register_cluster_analysis_callbacks
from .representatives import register_representatives_callbacks
from .heatmap import register_heatmap_callbacks
from .similarity import register_similarity_callbacks
from .image_loader import register_image_loader_callbacks
from .stratigraphy import register_stratigraphy_callbacks
from .cooccurrence import register_cooccurrence_callbacks


def register_analytics_callbacks(app, *, image_root, image_search_dirs=None):
    """注册分析相关回调（规模、质量、簇分析、代表样本、相似度等）。"""
    search_dirs = []
    if image_root:
        search_dirs.append(Path(image_root))
    if image_search_dirs:
        search_dirs.extend(Path(p) for p in image_search_dirs)
    seen_dirs = []
    for p in search_dirs:
        if p not in seen_dirs:
            seen_dirs.append(p)
    search_dirs = seen_dirs

    register_cluster_size_callbacks(app)
    register_cluster_quality_callbacks(app)
    register_category_breakdown_callbacks(app)
    register_cluster_analysis_callbacks(app)
    register_representatives_callbacks(app, image_root=image_root)
    register_heatmap_callbacks(app)
    register_similarity_callbacks(app)
    register_image_loader_callbacks(app, search_dirs=search_dirs)
    register_stratigraphy_callbacks(app)
    register_cooccurrence_callbacks(app)
    return app
