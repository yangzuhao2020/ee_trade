#!/usr/bin/env bash
# Train or evaluate one or all scenarios from examples/input/example_02a.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./example_02a.sh [SCENARIO|all] [simulator options]

Examples:
  ./example_02a.sh --no-plots
  ./example_02a.sh tiny --training-episodes 4 --training-runs 1 --no-plots
  ./example_02a.sh base --learning-mode evaluate --no-plots

The default scenario is base and the default learning mode is train.
The "all" selector runs every scenario supported by the current simulator.
Outputs are written to outputs/example_02a/SCENARIO.
EOF
}

if (($# > 0)) && [[ "$1" == "--help" ]]; then
  usage
  exit 0
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_name="example_02a"
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

from electricity_market_sim.config import load_market_settings
from electricity_market_sim.errors import InputValidationError

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict) or not config:
    raise SystemExit(f"No scenarios found in {config_path}")
for scenario_name in config:
    try:
        load_market_settings(config_path, scenario=scenario_name)
    except InputValidationError as exc:
        print(f"Skipping unsupported scenario {scenario_name}: {exc}", file=sys.stderr)
        continue
    print(scenario_name)
PY
  )"
  if [[ -z "$scenario_output" ]]; then
    echo "No supported scenarios found in $config_file" >&2
    exit 2
  fi
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
