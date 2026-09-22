#!/usr/bin/env bash
# Run one or all scenarios from examples/input/example_01b.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./example_01b.sh [SCENARIO|all] [simulator options]

Examples:
  ./example_01b.sh --no-plots       # defaults to base
  ./example_01b.sh all --no-plots
  ./example_01b.sh base --no-plots

Outputs are written to outputs/example_01b/SCENARIO.
EOF
}

if (($# > 0)) && [[ "$1" == "--help" ]]; then
  usage
  exit 0
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_name="example_01b"
input_dir="$project_dir/examples/input/$dataset_name"
config_file="$input_dir/config.yaml"

if [[ ! -f "$config_file" ]]; then
  echo "Dataset has no config.yaml: $input_dir" >&2
  exit 2
fi

scenario_selector="base"
if (($# > 0)) && [[ "$1" != --* ]]; then
  scenario_selector="$1"
  shift
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Cannot find conda in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate ee_trade
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$scenario_selector" == "all" ]]; then
  scenario_output="$(
    python - "$config_file" <<'PY'
from pathlib import Path
import sys

import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict) or not config:
    raise SystemExit(f"No scenarios found in {config_path}")
for scenario_name in config:
    print(scenario_name)
PY
  )"
  mapfile -t scenarios <<<"$scenario_output"
else
  scenarios=("$scenario_selector")
fi

for scenario in "${scenarios[@]}"; do
  output_dir="$project_dir/outputs/$dataset_name/$scenario"
  echo "Running dataset=$dataset_name scenario=$scenario"
  echo "Output: $output_dir"
  python -m electricity_market_sim \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --scenario "$scenario" \
    "$@"
done
