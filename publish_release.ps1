# Publish the prebuilt demo artefacts as a GitHub Release.
#
# Why a Release rather than git: the DuckDB is 232MB and telemetry_daily.parquet
# is 182MB, both over GitHub's 100MB per-file hard reject limit. Release assets
# allow up to 2GB each and live outside git history, so the clone stays ~7MB
# while judges can still get a one-click demo with no pipeline run.
#
#   .\publish_release.ps1              # DuckDB only (~232MB) - enough to run the demo
#   .\publish_release.ps1 -Full        # also the heavy parquets, for re-running features
#   .\publish_release.ps1 -Tag v1.1    # a different tag
param(
    [string]$Tag = "v1.0-demo",
    [switch]$Full,
    [string]$Title = "Prebuilt demo database"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$gh = "C:\Program Files\GitHub CLI\gh.exe"

if (-not (Test-Path $gh)) { Write-Host "GitHub CLI not found at $gh" -ForegroundColor Red; exit 1 }
& $gh auth status 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in to gh. Run this first:" -ForegroundColor Yellow
    Write-Host "    & '$gh' auth login" -ForegroundColor Cyan
    exit 1
}

$assets = @("data\servicesense.duckdb")
if ($Full) {
    $assets += "data\synthetic\telemetry_daily.parquet"
    $assets += "data\synthetic\train_snapshots.parquet"
    $assets += "data\synthetic\asset_features.parquet"
}

$missing = $assets | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host "Missing - run .\run_demo.ps1 -Rebuild first:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "    $_" }
    exit 1
}

$total = ($assets | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum / 1MB
Write-Host ("Uploading {0} asset(s), {1:N0} MB total. This takes a few minutes." -f $assets.Count, $total) -ForegroundColor Cyan
$assets | ForEach-Object { Write-Host ("    {0,8:N0} MB  {1}" -f ((Get-Item $_).Length / 1MB), $_) }

$notes = @"
Prebuilt artefacts so the demo runs without the ~6 minute pipeline build.

**To use:** download ``servicesense.duckdb`` into ``data/`` and run ``.\run_demo.ps1``
(no ``-Rebuild`` flag), then open http://localhost:5173/

Generated from the frozen seed (42). Rebuilding from source reproduces these
byte-for-byte: ``python -m pipeline.build_all``.

These are Release assets rather than committed files because both the database
(232 MB) and the daily telemetry (182 MB) exceed GitHub's 100 MB per-file limit.
The repo itself carries the result tables and the trained model, so the reported
scores are verifiable without this download.
"@

# Reuse the release if it already exists, so re-running just replaces assets.
& $gh release view $Tag 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Release $Tag exists - replacing assets." -ForegroundColor Yellow
    & $gh release upload $Tag @assets --clobber
} else {
    & $gh release create $Tag @assets --title $Title --notes $notes
}
if ($LASTEXITCODE -ne 0) { Write-Host "Upload failed." -ForegroundColor Red; exit 1 }

Write-Host "`nDone." -ForegroundColor Green
& $gh release view $Tag --json url --jq .url
