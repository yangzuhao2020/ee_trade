param(
    [string]$Scenario = "grpo_base",
    [int]$Rounds = 200
)

$ErrorActionPreference = "Stop"
& "$PSScriptRoot/example_02a.ps1" $Scenario --learning-mode train --training-rounds $Rounds @args
exit $LASTEXITCODE
