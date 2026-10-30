#!/usr/bin/env bash
# Reproduce every number and figure of the CPC manuscript.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 analysis =="
python3 code/cpc_analysis.py

echo "== 2/3 verification of the manuscript's numbers =="
python3 code/verify_claims.py

echo "== 3/3 figures =="
python3 code/make_figures_cpc.py

echo
echo "done. paper: paper/main.pdf   logs: logs/"
