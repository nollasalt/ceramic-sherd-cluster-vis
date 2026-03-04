"""回调注册入口，集中导出各功能模块的 register 函数。"""

from .scatter import register_scatter_callbacks
from .compare import register_compare_callbacks
from .cluster_panel import register_cluster_panel_callbacks
from .analytics import register_analytics_callbacks
from .recluster import register_recluster_callbacks
