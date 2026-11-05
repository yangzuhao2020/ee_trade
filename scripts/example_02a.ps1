# Train or evaluate one or all scenarios from examples/input/example_02a.
# PowerShell port of example_02a.sh.

if ($args.Count -gt 0 -and $args[0] -eq "--help") {
    @"
Usage:
  ./scripts/example_02a.ps1 [SCENARIO|all] [simulator options]

Examples:
  ./scripts/example_02a.ps1 --no-plots
  ./scripts/example_02a.ps1 tiny --training-episodes 4 --training-runs 1 --no-plots
  ./scripts/example_02a.ps1 grpo_tiny --training-steps 1728 --training-runs 1 --no-plots
  ./scripts/example_02a.ps1 base --learning-mode evaluate --no-plots

The default scenario is base and the default learning mode is train.
The "all" selector runs every scenario supported by the current simulator.
Outputs are written to outputs/example_02a/SCENARIO.
"@
    exit 0
}

& "$PSScriptRoot/run_example.ps1" "example_02a" @args
exit $LASTEXITCODE
