#!/bin/bash
# Certainty score of the decodes (8 Berlin slides) and per-type thresholds, after substrate_decode.sbatch.
# One checkpoint per slide first, then the slides are pooled in the order of certainty.slides in
# config/substrate.yaml. ALT_ROOT (default: roots.alt_decodes) holds the midground / adaptive decodes used to
# choose the threshold; with ALT_ROOT=none the threshold is certainty.fallback_threshold.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/substrate_certainty.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_substrate_certainty
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"
source workflow/hpc/env.sh
PYS="$(hhca_env substrate)/bin/python"
PY="$(hhca_env certainty)/bin/python"
CFG=config/substrate.yaml
OUT="$REPO/$($PYS scripts/substrate/config_get.py $CFG key roots.results)"
SLIDES="$($PYS scripts/substrate/config_get.py $CFG key certainty.slides)"
CHUNK="$($PYS scripts/substrate/config_get.py $CFG key certainty.chunksize)"
THR="$($PYS scripts/substrate/config_get.py $CFG key certainty.fallback_threshold)"
# settings of the score and of the per-type thresholds, passed on as JSON (the certainty environment has no YAML reader)
CERT="$($PYS scripts/substrate/config_get.py $CFG json certainty)"
WEIGHTS="$($PYS scripts/substrate/config_get.py $CFG json certainty.weights)"
TAU="$($PYS scripts/substrate/config_get.py $CFG json tau)"
ALT_ROOT="${ALT_ROOT:-$($PYS scripts/substrate/config_get.py $CFG key roots.alt_decodes)}"
ALT_ARGS=(--alt-root "$ALT_ROOT"); [ "$ALT_ROOT" = none ] && ALT_ARGS=()
COMMON=(--decode-root "$OUT/decode" --atlas-dir "$OUT/model" --outdir "$OUT/certainty" --params "$CERT" "${ALT_ARGS[@]}")
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date); alt decodes: $ALT_ROOT"
for s in $SLIDES; do
  "$PY" -u scripts/substrate/build_certainty_score.py "${COMMON[@]}" --slides "$s" --checkpoint-only --chunksize "$CHUNK"
done
JOINED=""; for s in $SLIDES; do JOINED="${JOINED:+$JOINED,}$s"; done
"$PY" -u scripts/substrate/build_certainty_score.py "${COMMON[@]}" --slides "$JOINED" --chunksize "$CHUNK" --threshold "$THR"
# per-type thresholds
"$PY" -u scripts/substrate/fit_type_thresholds.py --checkpoints "$OUT/certainty/checkpoints" \
  --cert-tables "$OUT/certainty/tables" --atlas-dir "$OUT/model" --outdir "$OUT/tau" --cache "$OUT/tau/cache" \
  --weights "$WEIGHTS" --params "$TAU"
"$PY" -u scripts/substrate/assign_with_thresholds.py --checkpoints "$OUT/certainty/checkpoints" \
  --cert-tables "$OUT/certainty/tables" --atlas-dir "$OUT/model" --outdir "$OUT/tau" --cache "$OUT/tau/cache" \
  --decode-root "$OUT/decode" --weights "$WEIGHTS" --params "$TAU"
echo "done $(date)"
