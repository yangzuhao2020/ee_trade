#!/usr/bin/env bash
# Run the electricity-market simulator from any working directory.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scenario="base_with_exchanges"
if (($# > 0)); then
  case "$1" in
    base|base_with_exchanges)
      scenario="$1"
      shift
      ;;
    --*)
      ;;
    *)
      echo "Usage: $0 [base|base_with_exchanges] [output_dir] [--no-plots]" >&2
      exit 2
      ;;
  esac
fi

case "$scenario" in
  base)
    default_output_dir="$project_dir/outputs/example_01a"
    ;;
  base_with_exchanges)
    default_output_dir="$project_dir/outputs/example_01a_exchange"
    ;;
  *)
    echo "Usage: $0 [base|base_with_exchanges] [output_dir] [--no-plots]" >&2
    exit 2
    ;;
esac

output_dir="$default_output_dir"
if (($# > 0)) && [[ "$1" != --* ]]; then
  output_dir="$1"
  shift
fi

conda_setup="/home/vr-y/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$conda_setup" ]]; then
  echo "Cannot find conda setup script: $conda_setup" >&2
  exit 1
fi

source "$conda_setup"
conda activate ee_trade

export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m electricity_market_sim \
  --input-dir "$project_dir/examples/input/example_01a" \
  --output-dir "$output_dir" \
  --scenario "$scenario" \
  "$@"
