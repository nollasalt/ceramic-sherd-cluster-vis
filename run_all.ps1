# 自动化运行聚类分析和可视化构建脚本
# ====================================-

# 设置脚本执行策略（如果需要）
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 项目根目录
$PROJECT_ROOT = "d:\Code\Project\ceramic-sherd-cluster-vis\cluster\src1"
$VENV_PATH = "$PROJECT_ROOT\venv\Scripts\python.exe"

# 检查虚拟环境Python是否存在
if (-not (Test-Path $VENV_PATH)) {
    Write-Host "虚拟环境Python不存在，请先创建虚拟环境" -ForegroundColor Red
    Write-Host "使用命令: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# 函数：运行Python脚本
function Run-PythonScript {
    param (
        [string]$ScriptPath,
        [string]$Description
    )
    
    Write-Host "\n正在$Description..." -ForegroundColor Cyan
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
Write-Host "开始执行完整的聚类分析构建流程" -ForegroundColor Blue
Write-Host "项目路径: $PROJECT_ROOT" -ForegroundColor Blue

# 步骤1: 运行kmeans_DINO.py进行聚类分析
if (-not (Run-PythonScript -ScriptPath "$PROJECT_ROOT\kmeans_DINO.py" -Description "运行聚类分析")) {
    Write-Host "\n构建流程失败，已终止" -ForegroundColor Red
    exit 1
}

# 步骤2: 运行build_table.py构建表格
if (-not (Run-PythonScript -ScriptPath "$PROJECT_ROOT\build_table.py" -Description "构建聚类表格")) {
    Write-Host "\n构建流程失败，已终止" -ForegroundColor Red
    exit 1
}

# 步骤3: 运行app_clusters.py启动可视化应用
Write-Host "\n所有构建步骤已完成，现在启动可视化应用..." -ForegroundColor Green
Write-Host "访问地址: http://127.0.0.1:9000/" -ForegroundColor Yellow
Write-Host "按 Ctrl+C 停止应用" -ForegroundColor Yellow

# 启动应用（不检查返回值，因为它是长运行进程）
& $VENV_PATH $PROJECT_ROOT\app_clusters.py

Write-Host "\n🏁 可视化应用已启动完成" -ForegroundColor Blue