# Start the ServiceSense demo: API + web, in one go.
#
#   .\run_demo.ps1            start both, print the URL
#   .\run_demo.ps1 -Rebuild   re-run the pipeline into DuckDB first
#
# OPENBLAS_NUM_THREADS is pinned because this box has ~3.9GB of RAM and
# OpenBLAS reserves large per-thread buffers at init; unpinned, the first
# /api/asset request dies with an allocation failure.
param([switch]$Rebuild)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"

if ($Rebuild) {
    Write-Host "Rebuilding the pipeline into DuckDB..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe -m pipeline.build_all
}

if (-not (Test-Path "data\servicesense.duckdb")) {
    Write-Host "No DuckDB found. Run with -Rebuild first." -ForegroundColor Red
    exit 1
}

Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process node   -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Starting API on :8000..." -ForegroundColor Cyan
Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "-m","uvicorn","api.main:app","--port","8000","--log-level","warning" `
  -WindowStyle Hidden -RedirectStandardOutput "data\_api.log" -RedirectStandardError "data\_api.err"

Write-Host "Starting web on :5173..." -ForegroundColor Cyan
Push-Location web
# npm on Windows is npm.cmd, a batch file. Start-Process cannot launch it
# directly ("%1 is not a valid Win32 application") because CreateProcess only
# runs real executables, so it goes through cmd.exe.
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","npm run dev" -WindowStyle Hidden `
  -RedirectStandardOutput "..\data\_vite.log" -RedirectStandardError "..\data\_vite.err"
Pop-Location

for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 1
    try { $h = Invoke-RestMethod "http://127.0.0.1:8000/api/health" -TimeoutSec 3; break } catch {}
}
if ($h) {
    Write-Host "`nAPI ready - $($h.recommendations) recommendation cards loaded" -ForegroundColor Green
    Write-Host "Open http://localhost:5173/`n" -ForegroundColor Green
} else {
    Write-Host "`nAPI did not come up. See data\_api.err" -ForegroundColor Red
}
