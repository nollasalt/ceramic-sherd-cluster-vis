# deploy.ps1 - 远程部署管理脚本
# 用法:
#   .\deploy.ps1 start   启动 frpc + 应用
#   .\deploy.ps1 stop    停止应用（+ 可选 frpc）
#   .\deploy.ps1 status  查看运行状态
#   .\deploy.ps1 log     实时查看应用日志

param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'log')]
    [string]$Action = 'status'
)

# 路径配置
$ROOT      = $PSScriptRoot
$PYTHON    = "$ROOT\venv\Scripts\python.exe"
$APP       = "$ROOT\app_clusters.py"
$FRPC_DIR  = "$ROOT\frp_0.66.0_windows_amd64"
$FRPC_EXE  = "$FRPC_DIR\frpc.exe"
$FRPC_CONF = "$FRPC_DIR\frpc.toml"
$APP_LOG   = "$ROOT\app_clusters.log"
$APP_ERR   = "$ROOT\app_clusters_error.log"
$FRPC_LOG  = "$FRPC_DIR\frpc.log"

# 辅助函数

function Get-AppProcess {
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*app_clusters.py*" }
}

function Get-FrpcProcess {
    Get-Process -Name 'frpc' -ErrorAction SilentlyContinue
}

function Write-Status {
    param([string]$Msg, [string]$Color = 'Cyan')
    Write-Host $Msg -ForegroundColor $Color
}

# start

function Start-Services {
    if (-not (Test-Path $PYTHON)) {
        Write-Status "未找到虚拟环境: $PYTHON" Red
        Write-Status "请先运行: python -m venv venv" Yellow
        exit 1
    }

    # 启动 frpc
    if (Test-Path $FRPC_EXE) {
        if (Get-FrpcProcess) {
            Write-Status "frpc 已在运行，跳过" Yellow
        } else {
            $frpcArgs = @{
                WindowStyle            = 'Hidden'
                FilePath               = $FRPC_EXE
                ArgumentList           = "-c `"$FRPC_CONF`""
                RedirectStandardOutput = $FRPC_LOG
            }
            Start-Process @frpcArgs
            Start-Sleep -Milliseconds 600
            if (Get-FrpcProcess) {
                Write-Status "frpc 已启动" Green
            } else {
                Write-Status "frpc 启动失败，请检查配置文件" Red
            }
        }
    } else {
        Write-Status "未找到 frpc，跳过隧道启动" Yellow
    }

    # 启动应用
    if (Get-AppProcess) {
        Write-Status "应用已在运行，跳过" Yellow
    } else {
        '' | Set-Content $APP_LOG
        '' | Set-Content $APP_ERR

        $appArgs = @{
            WindowStyle             = 'Hidden'
            FilePath                = $PYTHON
            ArgumentList            = "`"$APP`""
            WorkingDirectory        = $ROOT
            RedirectStandardOutput  = $APP_LOG
            RedirectStandardError   = $APP_ERR
        }
        Start-Process @appArgs

        Write-Status "等待应用启动..." Gray
        $started = $false
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep -Seconds 1
            if (Get-AppProcess) { $started = $true; break }
        }

        if ($started) {
            Write-Status "应用已启动" Green
        } else {
            Write-Status "应用启动超时，错误日志:" Red
            Get-Content $APP_ERR -Tail 10 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Host "  $_" -ForegroundColor DarkRed }
            exit 1
        }
    }

    Show-Status
}

# stop

function Stop-Services {
    $appProcs = Get-AppProcess
    if ($appProcs) {
        $appProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        Write-Status "应用已停止" Green
    } else {
        Write-Status "应用未在运行" Yellow
    }

    $frpc = Get-FrpcProcess
    if ($frpc) {
        $confirm = Read-Host "是否同时停止 frpc？[y/N]"
        if ($confirm -eq 'y' -or $confirm -eq 'Y') {
            $frpc | Stop-Process -Force
            Write-Status "frpc 已停止" Green
        }
    }
}

# status

function Show-Status {
    Write-Host ""
    Write-Host "---- 运行状态 ----" -ForegroundColor DarkGray

    $appProcs = Get-AppProcess
    if ($appProcs) {
        $pid_ = ($appProcs | Select-Object -First 1).ProcessId
        $mem  = [math]::Round((Get-Process -Id $pid_ -ErrorAction SilentlyContinue).WorkingSet64 / 1MB, 1)
        Write-Status "  应用  [运行中]  PID=$pid_  内存 ${mem} MB" Green
    } else {
        Write-Status "  应用  [未运行]" DarkGray
    }

    $frpc = Get-FrpcProcess
    if ($frpc) {
        Write-Status "  frpc  [运行中]  PID=$($frpc.Id)" Green
    } else {
        Write-Status "  frpc  [未运行]" DarkGray
    }

    if ($appProcs) {
        Write-Host ""
        Write-Status "  日志: $APP_LOG" Gray
        Write-Status "  错误: $APP_ERR" Gray
        $lastLine = Get-Content $APP_LOG -Tail 1 -ErrorAction SilentlyContinue
        if ($lastLine) { Write-Status "  最新: $lastLine" Gray }
    }

    Write-Host "------------------" -ForegroundColor DarkGray
    Write-Host ""
}

# log

function Watch-Log {
    if (-not (Test-Path $APP_LOG)) {
        Write-Status "日志文件不存在: $APP_LOG" Yellow
        return
    }
    Write-Status "实时日志（Ctrl+C 退出）" Gray
    Get-Content $APP_LOG -Wait -Tail 20
}

# 入口

switch ($Action) {
    'start'  { Start-Services }
    'stop'   { Stop-Services }
    'status' { Show-Status }
    'log'    { Watch-Log }
}
