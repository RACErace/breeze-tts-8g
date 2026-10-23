# Start the accelerated Breeze-TTS-2 WebUI natively on Windows.
# Override with BREEZE_MODEL_DIR / BREEZE_PYTHON / WEBUI_PORT.
$ErrorActionPreference = 'Continue'
$repo = $PSScriptRoot
if (-not $repo) { $repo = (Get-Location).Path }
$model = if ($env:BREEZE_MODEL_DIR) {
    $env:BREEZE_MODEL_DIR
} else {
    Join-Path $env:USERPROFILE 'models\breeze-tts-2'
}
$py = if ($env:BREEZE_PYTHON) {
    $env:BREEZE_PYTHON
} elseif (Test-Path (Join-Path $repo '.venv\Scripts\python.exe')) {
    Join-Path $repo '.venv\Scripts\python.exe'
} elseif ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX 'python.exe'
} else {
    (Get-Command python).Source
}
$port = if ($env:WEBUI_PORT) { $env:WEBUI_PORT } else { '7860' }

$env:PYTHONPATH = $repo
# Windows: the static CUDA launcher passes 64-bit kernel pointers through
# ctypes.c_long, which is 32-bit there; use the classic launcher instead.
$env:TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER = '0'
$env:TORCHINDUCTOR_CACHE_DIR = if ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA 'breeze-inductor'
} else {
    Join-Path $env:USERPROFILE '.cache\breeze-inductor'
}

$pidFile = Join-Path $env:USERPROFILE 'breeze-webui.pid'
$outLog = Join-Path $env:USERPROFILE 'breeze-webui-win.log'
$errLog = Join-Path $env:USERPROFILE 'breeze-webui-win.err.log'

if (Test-Path $pidFile) {
    $old = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($old) { Stop-Process -Id $old -Force -ErrorAction SilentlyContinue }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -like '*webui_server.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

$proc = Start-Process -FilePath $py -ArgumentList @(
    'webui_server.py', $model,
    '--port', $port,
    '--max-seq-len', '1024',
    '--fast-depth-decoder', '--fast-codec', '--fast-backbone-decode'
) -WorkingDirectory $repo -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
$proc.Id | Set-Content $pidFile
Write-Output "started pid=$($proc.Id) log=$outLog"

# The first start compiles the graph stages (a few minutes); later starts reuse
# the Inductor cache.
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        Write-Output 'PROCESS EXITED'
        Get-Content $errLog -Tail 40 -ErrorAction SilentlyContinue
        exit 1
    }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            Write-Output "READY after $($i * 2)s"
            Write-Output $r.Content
            Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/status" -TimeoutSec 5 -UseBasicParsing |
                Select-Object -ExpandProperty Content
            exit 0
        }
    } catch { }
}
Write-Output 'TIMEOUT waiting for WebUI'
exit 1
