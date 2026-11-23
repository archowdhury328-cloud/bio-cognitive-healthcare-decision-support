#!/usr/bin/env bash
# Reproduce every published result. Usage: bash scripts/reproduce.sh [synthetic|mimic|all]
# Raw clinical data and patient-level outputs stay under data/raw and runs_local (both git-ignored).
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
which="${1:-all}"

run_dataset() {  # $1 = dataset (synthetic|mimic), $2 = results dir, $3 = tag
  python -m biocog cohort --dataset "$1" --run "runs_local/$1"
  python -m biocog ablate --dataset "$1" --run "runs_local/$1"
  python -m biocog federated --dataset "$1" --run "runs_local/$1"
  python -m biocog update --dataset "$1" --run "runs_local/$1"
  python -m biocog report --run "runs_local/$1" --out "$2" --title "$4" --tag "$3"
  cp "runs_local/$1/federated.json" "$2/federated.json"
  cp "runs_local/$1/run_manifest.json" "$2/run_manifest.json"
  cp "runs_local/$1/registry/registry.json" "$2/registry.json"
  cp "runs_local/$1/models/proposed/meta.json" "$2/proposed_training_log.json"
}

if [[ "$which" == "synthetic" || "$which" == "all" ]]; then
  run_dataset synthetic results/synthetic "SYNTHETIC DATA - not clinical evidence" "Synthetic benchmark (400 generated patients)"
  python -m biocog route --out results/synthetic
fi
if [[ "$which" == "mimic" || "$which" == "all" ]]; then
  python -m biocog fetch-data
  run_dataset mimic results/mimic_demo "MIMIC-IV demo v2.2 (100 patients), aggregate only" "MIMIC-IV Clinical Database Demo v2.2: glucose-excursion engineering example"
fi
