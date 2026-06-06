# 交接说明

本文档面向后续维护者，补充 README 中没有展开的实现细节。

## 1. 架构概览

项目没有数据库，也没有独立 API 服务，核心就是一个 Dash 应用：

- 启动入口：`app_clusters.py`
- 布局定义：`app_core/layout.py`
- 页面布局：`app_core/tabs/*.py`
- 回调逻辑：`app_core/callbacks/` 与 `app_core/callbacks/analytics/`
- 前端增强：`assets/*.js`、`assets/style.css`

数据都来自本地文件，运行时通过 `app_core/data_cache.py` 放入进程内缓存供回调共享。

## 2. 启动时的数据流

`app_clusters.py` 的 `build_app_layout()` 启动时会做这些事：

1. 读取 `all_kmeans_new/cluster_metadata.json`。
2. 读取 `sherd_cluster_table_clustered_only.csv`。
3. 识别数值特征列。
4. 将 DataFrame 和元数据写入 `app_core/data_cache.py` 的全局缓存。
5. 构建页面布局。

这意味着：

- 应用是单进程内缓存模型。
- 如果未来改成多进程部署，不同 worker 之间不会自动共享这份缓存。

## 3. 重新聚类的数据流

前端“重新聚类”按钮对应：

- 入口回调：`app_core/callbacks/recluster.py`

执行链路：

1. 从 `all_features_dinov3.csv` 读取特征。
2. 按当前设置执行聚类。
3. 把新的 `piece_to_cluster` 写入 `all_kmeans_new/cluster_metadata.json`。
4. 用子进程执行 `scripts/build_table.py`。
5. 重建 `sherd_cluster_table_clustered_only.csv`。
6. 清理 `umap_cache.npz`。
7. 更新 URL 中的 `_reclustered` 参数，强制页面刷新。

所以如果“聚类成功但页面数据不对”，优先检查这三步：

- `cluster_metadata.json` 是否正确写入
- `scripts/build_table.py` 是否报错
- `sherd_cluster_table_clustered_only.csv` 是否真的重建

## 4. 当前推荐维护入口

### 改页面布局

- 到 `app_core/tabs/`

### 改交互逻辑

- 到 `app_core/callbacks/`
- 分析页逻辑主要在 `app_core/callbacks/analytics/`

### 改全局启动、数据装载、端口

- 到 `app_clusters.py`

### 改聚类或降维算法

- 到 `data_processing.py`

## 5. 重要文件说明

### `app_clusters.py`

负责：

- Dash 初始化
- 页面布局挂载
- 数据缓存初始化
- `/get_full_image` 路由
- 注册所有回调

### `data_processing.py`

负责：

- 读取聚类元数据
- 读取范围参考表
- K-Means / Agglomerative / Spectral / Leiden 聚类
- PCA / t-SNE / UMAP
- 图片转 base64

### `performance_utils.py`

负责：

- 图表缓存
- DataFrame 优化

### `app_core/callbacks/analytics.py`

这是历史遗留文件，注释已经写明：

- 同名目录包 `app_core/callbacks/analytics/` 才是当前真实生效的实现。
- 不要再把新逻辑写进这个单文件里。

## 6. 前端资源说明

`assets/` 里目前有一些对 Plotly / Dash 原生行为的补强：

- `image-modal.js`
  - 图片点击放大
- `modal-inline.js`
  - 部分内联弹窗逻辑
- `line-hover-highlight.js`
  - 地层流动与跨层趋势的悬浮高亮
- `style.css`
  - 全局样式

这些 JS 是直接随 Dash 静态资源加载的，不需要单独打包。

## 7. 数据文件依赖关系

### 主应用直接依赖

- `sherd_cluster_table_clustered_only.csv`
- `all_cutouts/`
- `all_kmeans_new/cluster_metadata.json`
- `scripts/jd_sherds_info.csv`

### 重新聚类额外依赖

- `all_features_dinov3.csv`

### 可安全删除并重建

- `umap_cache.npz`

## 8. 当前已知注意事项

### 1. `requirements.txt` 没有锁版本

如果在新机器上重建环境，可能出现版本差异。当前最稳妥的交接方式是直接交付现成 `venv`。

### 2. `scripts/run_all.ps1` 不是当前标准流程

它仍然引用旧文件路径：

- `kmeans_DINO.py`
- 根目录下的 `build_table.py`

当前实际重建脚本在 `scripts/build_table.py`。

### 3. Windows 适配最好

仓库内现成脚本和路径写法都偏向 Windows：

- PowerShell 启动脚本
- `venv\Scripts\...`
- 本地目录写法

如果后续迁移到 Linux，优先改启动脚本和路径处理。

### 4. 运行时初始化不算轻

虽然 UMAP 已按需计算，但应用启动仍会读取大 CSV，并初始化大量回调。首次启动慢一些是正常现象。

## 9. 建议的交付包内容

建议至少包含以下内容：

- 代码目录
- `venv/`
- `all_cutouts/`
- `all_features_dinov3.csv`
- `sherd_cluster_table_clustered_only.csv`
- `all_kmeans_new/`
- `scripts/jd_sherds_info.csv`

如果缺少图片或 CSV，应用只能部分运行，很多页面会直接失效。

## 10. 最低交接验收

交接给别人之前，建议现场确认这几步：

1. 能执行 `.\venv\Scripts\python.exe .\app_clusters.py`
2. 浏览器能打开 `http://127.0.0.1:9357`
3. 任意图片可以点开大图
4. 点击“重新聚类”后能刷新页面
5. `scripts/build_table.py` 能单独执行成功

