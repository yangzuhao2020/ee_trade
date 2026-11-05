# Run one or all scenarios from examples/input/example_01a.
# PowerShell port of example_01a.sh.

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    @"
Usage:
  ./scripts/example_01a.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_01a.ps1 --no-plots       # defaults to base
  ./scripts/example_01a.ps1 all --no-plots
  ./scripts/example_01a.ps1 base_with_exchanges --no-plots

Outputs are written to outputs/example_01a/SCENARIO.
"@
    exit 0
}

& "$PSScriptRoot/run_example.ps1" "example_01a" @args
exit $LASTEXITCODE
