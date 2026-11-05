# Run one or all scenarios from examples/input/example_03.
# PowerShell port of example_03.sh.

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    @"
Usage:
  ./scripts/example_03.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_03.ps1 --no-plots                         # defaults to base_case_2019_with_DSM
  ./scripts/example_03.ps1 base_case_2019_with_DSM
  ./scripts/example_03.ps1 all --no-plots

Outputs are written to outputs/example_03/SCENARIO.
"@
    exit 0
}

& "$PSScriptRoot/run_example.ps1" "example_03" @args
exit $LASTEXITCODE
