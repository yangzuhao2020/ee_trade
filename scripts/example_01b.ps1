# Run one or all scenarios from examples/input/example_01b.
# PowerShell port of example_01b.sh.

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    @"
Usage:
  ./scripts/example_01b.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_01b.ps1 --no-plots       # defaults to base
  ./scripts/example_01b.ps1 all --no-plots
  ./scripts/example_01b.ps1 base --no-plots

Outputs are written to outputs/example_01b/SCENARIO.
"@
    exit 0
}

& "$PSScriptRoot/run_example.ps1" "example_01b" @args
exit $LASTEXITCODE
