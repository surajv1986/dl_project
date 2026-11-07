param(
    [string]$Python = "",
    [string]$Question = "What was the name of the Apollo 11 lunar module?"
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
if (-not $Python) {
    $localPython = Join-Path (Split-Path -Parent $projectRoot) "dl_project\.venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $localPython) {
        $Python = $localPython
    } else {
        $Python = "python"
    }
}

function Invoke-Stage {
    param([string]$Script, [string[]]$Arguments)
    & $Python -X utf8 (Join-Path $projectRoot $Script) @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Script failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Wikipedia dataset pilot: verify and demonstrate local evidence retrieval"
Invoke-Stage "dataset_pipeline.py" @("verify")
Invoke-Stage "offline_check.py" @()
Invoke-Stage "retrieval_baseline.py" @("query", "--question", $Question, "--top-k", "3", "--output", (Join-Path $projectRoot "reports\demo_query.json"))
Write-Host "Complete. Presentation materials and run instructions are in the project README."
