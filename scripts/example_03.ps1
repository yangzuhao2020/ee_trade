# Run one or all scenarios from examples/input/example_03.
# PowerShell port of example_03.sh.

function Show-Usage {
    @"
Usage:
  ./scripts/example_03.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_03.ps1 --no-plots                         # defaults to base_case_2019_with_DSM
  ./scripts/example_03.ps1 base_case_2019_with_DSM
  ./scripts/example_03.ps1 all --no-plots

Outputs are written to outputs/example_03/SCENARIO.
"@
}

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    Show-Usage
    exit 0
}

$projectDir = Split-Path $PSScriptRoot -Parent
$datasetName = "example_03"
$inputDir = Join-Path $projectDir "examples/input/$datasetName"
$configFile = Join-Path $inputDir "config.yaml"

$remaining = [System.Collections.Generic.List[string]]::new()
$remaining.AddRange([string[]]$args)
$scenarioSelector = "base"
if ($remaining.Count -gt 0 -and -not $remaining[0].StartsWith("-")) {
    $scenarioSelector = $remaining[0]
    $remaining.RemoveAt(0)
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
    python -m electricity_market_sim.main `
        --input-dir $inputDir `
        --output-dir $outputDir `
        --scenario $scenario `
        @remaining
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
