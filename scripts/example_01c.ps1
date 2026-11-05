# Run one or all scenarios from examples/input/example_01c.
# PowerShell port of example_01c.sh.

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    @"
Usage:
  ./scripts/example_01c.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_01c.ps1 --no-plots            # defaults to eom_only
  ./scripts/example_01c.ps1 all --no-plots
  ./scripts/example_01c.ps1 eom_only --no-plots

Outputs are written to outputs/example_01c/SCENARIO.
"@
    exit 0
}

& "$PSScriptRoot/run_example.ps1" "example_01c" @args
exit $LASTEXITCODE
