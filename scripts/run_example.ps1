param([string]$DatasetName)

$projectDir = Split-Path $PSScriptRoot -Parent
$inputDir = Join-Path $projectDir "examples/input/$DatasetName"
$configFile = Join-Path $inputDir "config.yaml"

$remaining = @($args)
$scenarioSelector = "base"
if ($remaining.Count -gt 0 -and -not $remaining[0].StartsWith("-")) {
    $scenarioSelector = $remaining[0]
    $remaining = @($remaining | Select-Object -Skip 1)
}

(& conda.exe shell.powershell hook) | Out-String | Invoke-Expression
conda activate ee_trade
$env:PYTHONPATH = "$projectDir$(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" })"

if ($scenarioSelector -eq "all") {
    $scenarios = python -c @'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))
for name in config:
    print(name)
'@ $configFile
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    $scenarios = @($scenarioSelector)
}

foreach ($scenario in $scenarios) {
    $outputDir = Join-Path $projectDir "outputs/$DatasetName/$scenario"
    Write-Host "Running dataset=$DatasetName scenario=$scenario"
    Write-Host "Output: $outputDir"
    python -m electricity_market_sim.main `
        --input-dir $inputDir `
        --output-dir $outputDir `
        --scenario $scenario `
        @remaining
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
exit $LASTEXITCODE
