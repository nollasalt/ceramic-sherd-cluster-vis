# 自动化运行仅正面图像的聚类分析和可视化构建脚本
# ==================================================

# 设置UTF-8编码以正确显示中文
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 设置脚本执行策略（如果需要）
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 项目根目录
$PROJECT_ROOT = "d:\Code\Project\ceramic-sherd-cluster-vis\cluster\src1"
$TEST_ROOT = "$PROJECT_ROOT\exterior"
$VENV_PATH = "$PROJECT_ROOT\venv\Scripts\python.exe"

# 检查虚拟环境Python是否存在
if (-not (Test-Path $VENV_PATH)) {
    Write-Host "虚拟环境Python不存在，请先创建虚拟环境" -ForegroundColor Red
    Write-Host "使用命令: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# 切换到test目录
Set-Location $TEST_ROOT

# 函数：运行Python脚本
function Run-PythonScript {
    param (
        [string]$ScriptPath,
        [string]$Description
    )
    
    Write-Host "`n正在$Description..." -ForegroundColor Cyan
    Write-Host "命令: $VENV_PATH $ScriptPath" -ForegroundColor Gray
    
    # 运行脚本
    & $VENV_PATH $ScriptPath
    
    # 检查返回值
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$Description失败，返回码: $LASTEXITCODE" -ForegroundColor Red
        return $false
    } else {
        Write-Host "$Description成功完成" -ForegroundColor Green
        return $true
    }
}

# 开始执行流程
Write-Host "开始执行仅正面图像的聚类分析构建流程" -ForegroundColor Blue
Write-Host "项目路径: $PROJECT_ROOT" -ForegroundColor Blue
Write-Host "测试路径: $TEST_ROOT" -ForegroundColor Blue

Write-Host "`n 处理流程:" -ForegroundColor Yellow
Write-Host "  1. 提取正面图像DINOv3特征" -ForegroundColor Yellow
Write-Host "  2. 对正面图像进行K-means聚类" -ForegroundColor Yellow
Write-Host "  3. 构建正面图像聚类表格" -ForegroundColor Yellow
Write-Host "  4. 启动正面图像可视化应用" -ForegroundColor Yellow

# 步骤1: 运行DINOV3_features_exterior.py提取正面特征
if (-not (Run-PythonScript -ScriptPath "$TEST_ROOT\DINOV3_features_exterior.py" -Description "提取正面图像DINOv3特征")) {
    Write-Host "`n 构建流程失败，已终止" -ForegroundColor Red
    exit 1
}

# 步骤2: 运行kmeans_exterior_only.py进行正面图像聚类
if (-not (Run-PythonScript -ScriptPath "$TEST_ROOT\kmeans_exterior_only.py" -Description "正面图像K-means聚类")) {
    Write-Host "`n 构建流程失败，已终止" -ForegroundColor Red
    exit 1
}

# 步骤3: 运行build_exterior_table.py构建表格
if (-not (Run-PythonScript -ScriptPath "$TEST_ROOT\build_exterior_table.py" -Description "构建正面图像聚类表格")) {
    Write-Host "`n 构建流程失败，已终止" -ForegroundColor Red
    exit 1
}

# 显示处理完成的统计信息
Write-Host "`n 处理完成统计:" -ForegroundColor Green
if (Test-Path "$TEST_ROOT\exterior_features_dinov3.csv") {
    $features = Import-Csv "$TEST_ROOT\exterior_features_dinov3.csv"
    Write-Host "  - 提取特征: $($features.Count) 张正面图像" -ForegroundColor Green
}

if (Test-Path "$TEST_ROOT\exterior_cluster_table.csv") {
    $table = Import-Csv "$TEST_ROOT\exterior_cluster_table.csv"
    Write-Host "  - 聚类表格: $($table.Count) 条记录" -ForegroundColor Green
    $clusters = ($table | Group-Object cluster_id).Count
    Write-Host "  - 聚类数量: $clusters 个" -ForegroundColor Green
}

if (Test-Path "$TEST_ROOT\exterior_kmeans_results") {
    $clusterDirs = Get-ChildItem "$TEST_ROOT\exterior_kmeans_results" -Directory | Where-Object { $_.Name -match "^cluster_\d+" }
    Write-Host "  - 聚类目录: $($clusterDirs.Count) 个" -ForegroundColor Green
}

# 步骤4: 运行app_exterior_clusters.py启动可视化应用
Write-Host "`n 所有构建步骤已完成，现在启动正面图像可视化应用..." -ForegroundColor Green
Write-Host " 访问地址: http://127.0.0.1:9001/" -ForegroundColor Yellow
Write-Host " 按 Ctrl+C 停止应用" -ForegroundColor Yellow

# 启动应用（不检查返回值，因为它是长运行进程）
& $VENV_PATH $TEST_ROOT\app_exterior_clusters.py

Write-Host "`n 正面图像聚类可视化应用已启动完成" -ForegroundColor Blue