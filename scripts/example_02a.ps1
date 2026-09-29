# Train or evaluate one or all scenarios from examples/input/example_02a.
# PowerShell port of example_02a.sh.

$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage:
  ./scripts/example_02a.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_02a.ps1 --no-plots
  ./scripts/example_02a.ps1 tiny --training-episodes 4 --training-runs 1 --no-plots
  ./scripts/example_02a.ps1 base --learning-mode evaluate --no-plots

The default scenario is base and the default learning mode is train.
The "all" selector runs every scenario supported by the current simulator.
Outputs are written to outputs/example_02a/SCENARIO.
"@
}

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    Show-Usage
    exit 0
}

$projectDir = Split-Path $PSScriptRoot -Parent
$datasetName = "example_02a"
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

from electricity_market_sim.config import load_market_settings
from electricity_market_sim.errors import InputValidationError

config_path = Path(r'''$configFile''')
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict) or not config:
    raise SystemExit(f"No scenarios found in {config_path}")
for scenario_name in config:
    try:
        load_market_settings(config_path, scenario=scenario_name)
    except InputValidationError as exc:
        print(f"Skipping unsupported scenario {scenario_name}: {exc}", file=sys.stderr)
        continue
    print(scenario_name)
"@
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not $scenarios) {
        Write-Error "No supported scenarios found in $configFile"
        exit 2
    }
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
