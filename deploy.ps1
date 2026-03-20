# deploy.ps1 - Remote deployment management
# Usage:
#   .\deploy.ps1 start   Start frpc + app
#   .\deploy.ps1 stop    Stop app (optionally frpc)
#   .\deploy.ps1 status  Show running status
#   .\deploy.ps1 log     Tail app log

param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'log')]
    [string]$Action = 'status'
)

$ROOT      = $PSScriptRoot
$PYTHON    = "$ROOT\venv\Scripts\python.exe"
$APP       = "$ROOT\app_clusters.py"
$FRPC_DIR  = "$ROOT\frp_0.66.0_windows_amd64"
$FRPC_EXE  = "$FRPC_DIR\frpc.exe"
$FRPC_CONF = "$FRPC_DIR\frpc.toml"
$APP_LOG   = "$ROOT\app_clusters.log"
$APP_ERR   = "$ROOT\app_clusters_error.log"
$FRPC_LOG  = "$FRPC_DIR\frpc.log"

function Get-AppProcess {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*app_clusters.py*" }
}

function Get-FrpcProcess {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*$FRPC_CONF*" }
}

function Write-Msg {
    param([string]$Text, [string]$Color = 'Cyan')
    Write-Host $Text -ForegroundColor $Color
}

function Start-Services {
    if (-not (Test-Path $PYTHON)) {
        Write-Msg "[ERROR] venv not found: $PYTHON" Red
        Write-Msg "Run: python -m venv venv" Yellow
        exit 1
    }

    if (Test-Path $FRPC_EXE) {
        if (Get-FrpcProcess) {
            Write-Msg "[SKIP] frpc already running" Yellow
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
                Write-Msg "[OK] frpc started" Green
            } else {
                Write-Msg "[WARN] frpc failed to start, check config" Red
            }
        }
    } else {
        Write-Msg "[SKIP] frpc not found, skipping tunnel" Yellow
    }

    if (Get-AppProcess) {
        Write-Msg "[SKIP] app already running" Yellow
    } else {
        '' | Set-Content $APP_LOG
        '' | Set-Content $APP_ERR

        $appArgs = @{
            WindowStyle            = 'Hidden'
            FilePath               = $PYTHON
            ArgumentList           = "`"$APP`""
            WorkingDirectory       = $ROOT
            RedirectStandardOutput = $APP_LOG
            RedirectStandardError  = $APP_ERR
        }
        Start-Process @appArgs

        Write-Msg "[WAIT] waiting for app to start..." Gray
        $started = $false
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep -Seconds 1
            if (Get-AppProcess) {
                $started = $true
                break
            }
        }

        if ($started) {
            Write-Msg "[OK] app started" Green
        } else {
            Write-Msg "[ERROR] app did not start within 15s, last errors:" Red
            Get-Content $APP_ERR -Tail 10 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Host "  $_" -ForegroundColor DarkRed }
            exit 1
        }
    }

    Show-Status
}

function Stop-Services {
    $appProcs = Get-AppProcess
    if ($appProcs) {
        $appProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        Write-Msg "[OK] app stopped" Green
    } else {
        Write-Msg "[INFO] app is not running" Yellow
    }

    $frpc = Get-FrpcProcess
    if ($frpc) {
        $answer = Read-Host "Stop frpc tunnel too? [y/N]"
        if ($answer -eq 'y' -or $answer -eq 'Y') {
        $frpc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
            Write-Msg "[OK] frpc stopped" Green
        }
    }
}

function Show-Status {
    Write-Host ""
    Write-Host "---- Status ----" -ForegroundColor DarkGray

    $appProcs = Get-AppProcess
    if ($appProcs) {
        $pid_ = ($appProcs | Select-Object -First 1).ProcessId
        $proc = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
        $mem  = if ($proc) { [math]::Round($proc.WorkingSet64 / 1MB, 1) } else { '?' }
        Write-Msg "  App   [RUNNING]  PID=$pid_  Mem=${mem}MB" Green
    } else {
        Write-Msg "  App   [STOPPED]" DarkGray
    }

    $frpc = Get-FrpcProcess
    if ($frpc) {
        Write-Msg "  frpc  [RUNNING]  PID=$($frpc.ProcessId)" Green
    } else {
        Write-Msg "  frpc  [STOPPED]" DarkGray
    }

    if ($appProcs) {
        Write-Host ""
        Write-Msg "  Log : $APP_LOG" Gray
        Write-Msg "  Err : $APP_ERR" Gray
        $last = Get-Content $APP_LOG -Tail 1 -ErrorAction SilentlyContinue
        if ($last) { Write-Msg "  Last: $last" Gray }
    }

    Write-Host "----------------" -ForegroundColor DarkGray
    Write-Host ""
}

function Watch-Log {
    if (-not (Test-Path $APP_LOG)) {
        Write-Msg "[WARN] log file not found: $APP_LOG" Yellow
        return
    }
    Write-Msg "Tailing log (Ctrl+C to exit)" Gray
    Get-Content $APP_LOG -Wait -Tail 20
}

switch ($Action) {
    'start'  { Start-Services }
    'stop'   { Stop-Services }
    'status' { Show-Status }
    'log'    { Watch-Log }
}
