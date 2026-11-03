param([string]$Scenario = "grpo_base")

& "$PSScriptRoot/example_02a_test.ps1" $Scenario @args
exit $LASTEXITCODE
