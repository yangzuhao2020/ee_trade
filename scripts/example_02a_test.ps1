param([string]$Scenario = "base")

$projectDir = Split-Path $PSScriptRoot -Parent

(& conda.exe shell.powershell hook) | Out-String | Invoke-Expression
conda activate ee_trade
$env:PYTHONPATH = "$projectDir$(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" })"

python -m electricity_market_sim.main `
    --input-dir (Join-Path $projectDir "examples/input/example_02a") `
    --output-dir (Join-Path $projectDir "outputs/example_02a_test/$Scenario") `
    --scenario $Scenario `
    --learning-mode test `
    --checkpoint (Join-Path $projectDir "outputs/example_02a/$Scenario/checkpoints/runs.json") `
    @args
exit $LASTEXITCODE
