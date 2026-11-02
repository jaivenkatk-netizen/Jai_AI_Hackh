# Create the GitHub repo and push. Run AFTER `gh auth login`.
#
#   .\setup_github.ps1                 # private repo named servicesense
#   .\setup_github.ps1 -Public         # public instead
#   .\setup_github.ps1 -Name my-repo   # different name
#
# Defaults to PRIVATE on purpose: the commit messages and README quote the
# challenge brief, and making a repo public later is one click whereas
# un-publishing something already indexed is not.
param(
    [string]$Name = "servicesense",
    [switch]$Public,
    [string]$Description = "ServiceSense - HVAC predictive maintenance to inventory (hackathon A01)"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$gh = "C:\Program Files\GitHub CLI\gh.exe"

if (-not (Test-Path $gh)) { Write-Host "GitHub CLI not found at $gh" -ForegroundColor Red; exit 1 }

& $gh auth status 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Run this first, then re-run this script:" -ForegroundColor Yellow
    Write-Host "    & '$gh' auth login" -ForegroundColor Cyan
    exit 1
}

if (git status --porcelain) {
    Write-Host "Uncommitted changes present - commit them first." -ForegroundColor Yellow
    git status --short
    exit 1
}

$visibility = if ($Public) { "--public" } else { "--private" }
Write-Host "Creating $(if ($Public) {'PUBLIC'} else {'private'}) repo '$Name' and pushing..." -ForegroundColor Cyan

& $gh repo create $Name $visibility --source . --description $Description --push
if ($LASTEXITCODE -ne 0) { Write-Host "Repo create/push failed." -ForegroundColor Red; exit 1 }

Write-Host "`nDone. Remote:" -ForegroundColor Green
git remote -v
& $gh repo view --web --json url --jq .url 2>$null
