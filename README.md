# 陶片聚类交互可视化

本项目是一个基于 Dash 的本地交互式可视化工具，用于浏览陶片聚类结果、重新聚类，并从地层、器类、器部、相似性等多个角度分析簇的结构。

推荐先看本文档完成启动，再看 [docs/HANDOVER.md](docs/HANDOVER.md) 了解维护细节。

## 1. 当前代码的实际入口

- Web 应用入口：`app_clusters.py`
- 默认端口：`9357`
- 本地访问地址：`http://127.0.0.1:9357`
- 可选后台启动脚本：`deploy.ps1`

应用启动时会读取本地 CSV、图片目录和聚类元数据，不依赖数据库或外部后端服务。

## 2. 当前环境快照

这是仓库当前自带虚拟环境中的实际版本，适合作为交接参考：

- 操作系统：Windows
- Shell：PowerShell
- Python：`3.13.5`
- Dash：`3.3.0`
- pandas：`2.3.3`
- plotly：`5.24.1`
- numpy：`2.3.5`
- scipy：`1.16.3`
- scikit-learn：`1.8.0`
- python-igraph：`1.0.0`
- leidenalg：`0.11.0`
- torch：`2.7.1+cu118`
- torchvision：`0.22.1+cu118`
- Pillow：`12.1.0`
- umap-learn：`0.5.9.post2`
- timm：`1.0.24`

说明：

- `requirements.txt` 目前只有包名，没有锁定版本。
- 如果要完全复现当前环境，建议直接复用仓库内现有 `venv`，或手动按上面的版本创建新环境。

## 3. 关键数据文件

应用依赖以下本地文件和目录：

- `sherd_cluster_table_clustered_only.csv`
  - 应用主数据表。
  - `app_clusters.py` 启动时直接读取它。
- `all_features_dinov3.csv`
  - 重新聚类使用的特征表。
- `all_cutouts/`
  - 图片目录，页面缩略图和大图查看都依赖它。
- `all_kmeans_new/cluster_metadata.json`
  - 聚类元数据，包含 `piece_to_cluster` 映射。
- `scripts/jd_sherds_info.csv`
  - 参考信息表。
  - 用于补充 `unit_C`、`part_C`、`type_C` 等字段，也用于聚类范围筛选。
- `umap_cache.npz`
  - UMAP 缓存文件，可删除，应用会在需要时重建。

## 4. 首次启动

### 4.1 使用仓库内现成 `venv`

```powershell
cd D:\Code\Project\ceramic-sherd-cluster-vis\cluster\src1
.\venv\Scripts\python.exe .\app_clusters.py
```

然后打开：

```text
http://127.0.0.1:9357
```

### 4.2 在新机器上创建环境

```powershell
cd D:\Code\Project\ceramic-sherd-cluster-vis\cluster\src1
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python .\app_clusters.py
```

## 5. 常用运行方式

### 5.1 前台直接运行

适合开发和临时查看：

```powershell
.\venv\Scripts\python.exe .\app_clusters.py
```

### 5.2 修改端口或调试模式

`app_clusters.py` 支持两个环境变量：

- `CERAMIC_PORT`
- `CERAMIC_DEBUG`

示例：

```powershell
$env:CERAMIC_PORT = "9360"
$env:CERAMIC_DEBUG = "true"
.\venv\Scripts\python.exe .\app_clusters.py
```

### 5.3 使用 `deploy.ps1` 后台运行

适合交付给本机长期运行：

```powershell
.\deploy.ps1 start
.\deploy.ps1 status
.\deploy.ps1 log
.\deploy.ps1 stop
```

说明：

- `deploy.ps1` 会尝试启动 Dash 应用。
- 如果 `frp_0.66.0_windows_amd64\frpc.exe` 存在，也会尝试一起启动 FRP 隧道。
- FRP 不是应用运行必需项，只是远程访问辅助。

## 6. 重新聚类与数据重建

### 6.1 页面内重新聚类

页面顶部支持以下聚类参数：

- 聚类算法
- 聚类模式
- 平均聚类大小
- PCA 预处理
- 分层聚类
- 层位 / Part 范围聚类

点击“重新聚类”后，系统会自动：

1. 基于 `all_features_dinov3.csv` 执行聚类。
2. 写回 `all_kmeans_new/cluster_metadata.json`。
3. 自动调用 `scripts/build_table.py` 重建 `sherd_cluster_table_clustered_only.csv`。
4. 删除 `umap_cache.npz`，等待下次进入散点图页时重新计算。
5. 刷新页面，载入新的聚类结果。

### 6.2 手动重建总表

如果你手动替换了 `cluster_metadata.json`，需要同步重建主表：

```powershell
.\venv\Scripts\python.exe .\scripts\build_table.py
```

## 7. 目录说明

```text
src1/
├─ app_clusters.py                  Dash 应用入口
├─ data_processing.py               聚类、降维、图片与元数据工具
├─ performance_utils.py             图表缓存和 DataFrame 优化
├─ app_core/
│  ├─ layout.py                     主布局
│  ├─ data_cache.py                 服务端内存缓存
│  ├─ callbacks/                    各类 Dash 回调
│  └─ tabs/                         各页面布局
├─ assets/                          前端 JS/CSS 资源
├─ scripts/
│  ├─ build_table.py                重建主数据表
│  └─ jd_sherds_info.csv            参考信息表
├─ all_cutouts/                     图片
├─ all_kmeans_new/                  聚类元数据目录
├─ sherd_cluster_table_clustered_only.csv
└─ all_features_dinov3.csv
```

## 8. 需要特别注意的脚本

### 推荐使用

- `app_clusters.py`
- `deploy.ps1`
- `scripts/build_table.py`

### 不建议直接依赖

- `scripts/run_all.ps1`

原因：

- 该脚本仍引用 `kmeans_DINO.py`，但当前仓库根目录下没有这个文件。
- 它还假定 `build_table.py` 位于项目根目录，而当前实际路径是 `scripts/build_table.py`。
- 因此它更像历史遗留脚本，不应作为当前交接后的标准启动方式。

## 9. 常见问题

### 页面打开很慢

- 首次进入某些分析页会现场构建图表。
- 散点图页的 UMAP 已改为按需计算，不会在应用启动时立即运行。

### 图片点开失败

- 确认 `all_cutouts/` 中存在对应图片。
- 图片大图接口由 `/get_full_image` 提供，路径解析基于文件名匹配。

### 重新聚类后结果没更新

- 确认 `all_kmeans_new/cluster_metadata.json` 已写入。
- 确认 `scripts/build_table.py` 执行成功。
- 必要时删除 `umap_cache.npz` 后重启应用。

## 10. 交接建议

- 优先把本目录整体打包给接手人，包括 `venv`、CSV、图片目录和 `all_kmeans_new/`。
- 至少确认接手方能运行一次：

```powershell
.\venv\Scripts\python.exe .\app_clusters.py
```

- 如果只交代码不交数据，这个项目几乎无法完整运行。
