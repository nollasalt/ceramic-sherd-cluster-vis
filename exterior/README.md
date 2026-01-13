# 陶片正面图像聚类分析系统 (Test Version)

## 📋 概述

这是原项目的简化版本，**仅处理陶片正面(exterior)图像**，不进行正反面融合。适用于：
- 快速原型验证
- 单面图像分析
- 性能对比测试
- 简化的聚类分析

## 🗂️ 文件结构

```
test/
├── model.py                          # DINOv3模型定义
├── DINOV3_features_exterior.py       # 提取正面图像特征
├── kmeans_exterior_only.py           # 正面图像K-means聚类
├── build_exterior_table.py           # 构建正面图像数据表
├── app_exterior_clusters.py          # 正面图像可视化应用
├── run_exterior_analysis.ps1         # 一键运行脚本
└── README.md                         # 本文件
```

## 🚀 快速开始

### 1. 一键运行（推荐）
```powershell
cd test
.\run_exterior_analysis.ps1
```

### 2. 分步运行
```powershell
# 进入test目录
cd test

# 1. 提取正面图像特征
python DINOV3_features_exterior.py

# 2. 正面图像聚类
python kmeans_exterior_only.py

# 3. 构建数据表
python build_exterior_table.py

# 4. 启动可视化应用
python app_exterior_clusters.py
```

## 📊 数据流程

```mermaid
graph LR
    A[all_cutouts/*.png] --> B[筛选exterior图像]
    B --> C[DINOV3特征提取]
    C --> D[K-means聚类]
    D --> E[数据表构建]
    E --> F[可视化应用]
```

## 🔧 配置说明

### DINOV3_features_exterior.py
- **输入**: `../all_cutouts/` 中的所有图像
- **筛选**: 仅保留包含 "exterior" 的文件
- **输出**: `exterior_features_dinov3.csv`

### kmeans_exterior_only.py
- **输入**: `exterior_features_dinov3.csv`
- **聚类数**: 40个聚类（可调整）
- **输出**: `exterior_kmeans_results/` 目录

### build_exterior_table.py
- **输入**: 特征文件 + 聚类结果 + 陶片信息
- **输出**: `exterior_cluster_table.csv`

### app_exterior_clusters.py
- **端口**: 9001 (避免与主应用冲突)
- **功能**: PCA/t-SNE/UMAP降维可视化

## 📈 与原版本的差异

| 特性 | 原版本 | Test版本 |
|------|--------|----------|
| 图像处理 | 正反面融合 | 仅正面 |
| 数据量 | ~2000陶片 × 2 = 4000图像 | ~2000正面图像 |
| 特征维度 | 128×2=256维 | 128维 |
| 聚类对象 | 陶片(融合特征) | 单张图像 |
| 处理速度 | 较慢 | 更快 |
| 应用端口 | 9000 | 9001 |

## 🎯 适用场景

1. **快速验证**: 测试DINOv3特征提取效果
2. **算法对比**: 比较不同聚类算法性能
3. **参数调优**: 快速测试不同的聚类参数
4. **单面分析**: 专门分析陶片正面特征

## ⚠️ 注意事项

1. **数据依赖**: 需要上级目录的 `all_cutouts/` 和 `dinov3_epoch_100.pth`
2. **命名约定**: 图像文件名必须包含 "_exterior"
3. **输出目录**: 所有输出文件在 `test/` 目录下
4. **端口冲突**: 使用9001端口，与主应用区分

## 🔍 输出文件说明

- `exterior_features_dinov3.csv`: 正面图像DINOv3特征
- `exterior_kmeans_results/`: 聚类结果目录
- `exterior_cluster_table.csv`: 最终分析表格
- `exterior_kmeans_results/cluster_metadata.json`: 聚类元数据

## 📱 可视化界面

访问 http://127.0.0.1:9001/ 查看：
- 降维散点图 (PCA/t-SNE/UMAP)
- 聚类特征热力图
- 聚类相似度矩阵
- 交互式筛选功能

## 🛠️ 故障排除

### 常见问题

1. **找不到图像**: 检查 `../all_cutouts/` 路径
2. **没有exterior图像**: 确认文件命名包含 "_exterior"
3. **模型加载失败**: 检查 `../dinov3_epoch_100.pth` 是否存在
4. **CUDA错误**: 确认PyTorch CUDA版本正确安装

### 检查步骤

```powershell
# 检查图像文件
ls ../all_cutouts/*exterior*

# 检查模型文件
ls ../dinov3_epoch_100.pth

# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## 📞 支持

如遇问题，请检查：
1. 原项目是否正常运行
2. 所有依赖文件是否存在
3. Python环境配置是否正确