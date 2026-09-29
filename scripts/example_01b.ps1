# Run one or all scenarios from examples/input/example_01b.
# PowerShell port of example_01b.sh.

$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage:
  ./scripts/example_01b.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_01b.ps1 --no-plots       # defaults to base
  ./scripts/example_01b.ps1 all --no-plots
  ./scripts/example_01b.ps1 base --no-plots

Outputs are written to outputs/example_01b/SCENARIO.
"@
}

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    Show-Usage
    exit 0
}

$projectDir = Split-Path $PSScriptRoot -Parent
$datasetName = "example_01b"
$inputDir = Join-Path $projectDir "examples/input/$datasetName"
$configFile = Join-Path $inputDir "config.yaml"

if (-not (Test-Path $configFile)) {
    Write-Error "Dataset has no config.yaml: $inputDir"
    exit 2
}

$remaining = [System.Collections.Generic.List[string]]::new()
$remaining.AddRange([string[]]$args)
$scenarioSelector = "base"
if ($remaining.Count -gt 0 -and -not $remaining[0].StartsWith("-")) {
    $scenarioSelector = $remaining[0]
    $remaining.RemoveAt(0)
}

if (-not (Get-Command conda.exe -ErrorAction SilentlyContinue)) {
    Write-Error "Cannot find conda in PATH."
    exit 1
}
(& conda.exe shell.powershell hook) | Out-String | Invoke-Expression
conda activate ee_trade
$env:PYTHONPATH = "$projectDir$(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" })"

if ($scenarioSelector -eq "all") {
    $scenarios = python -c @"
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(r'''$configFile''').read_text(encoding="utf-8"))
if not isinstance(config, dict) or not config:
    raise SystemExit("No scenarios found in $configFile")
for name in config:
    print(name)
"@
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    $scenarios = @($scenarioSelector)
}

foreach ($scenario in $scenarios) {
    $outputDir = Join-Path $projectDir "outputs/$datasetName/$scenario"
    Write-Host "Running dataset=$datasetName scenario=$scenario"
    Write-Host "Output: $outputDir"
    python -m electricity_market_sim `
        --input-dir $inputDir `
        --output-dir $outputDir `
        --scenario $scenario `
        @remaining
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
