# Run one or all scenarios from examples/input/example_01h_saving.
# PowerShell port of example_01h.sh.

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    @"
Usage:
  ./scripts/example_01h.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_01h.ps1 --no-plots       # defaults to base
  ./scripts/example_01h.ps1 base --no-plots
  ./scripts/example_01h.ps1 all

Inputs are read from examples/input/example_01h_saving.
Outputs are written to outputs/example_01h_saving/SCENARIO.
"@
    exit 0
}

& "$PSScriptRoot/run_example.ps1" "example_01h_saving" @args
exit $LASTEXITCODE
