#!/bin/bash
# CMFLAG-clean 2 um and 8 um h5ads of one slide, after substrate_cmflag_flags.sbatch: the h5ads of
# results/substrate/h5ad_{2um,8um}/ without the bins of the four spillover flags.
# Outputs: results/substrate/clean_h5ad/{h5ad_2um_CMFLAG_clean,h5ad_8um_CMFLAG_clean,summary}/
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/substrate_cmflag_clean.sbatch SLIDE
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_cmflag_clean
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -euo pipefail
SLIDE="${1:?usage: sbatch workflow/hpc/substrate_cmflag_clean.sbatch SLIDE}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TMPDIR" logs
source workflow/hpc/env.sh
PY="$(hhca_env substrate_cmflag)/bin/python"
OUT=results/substrate/clean_h5ad
FL="results/substrate/cmflag/flags/${SLIDE}_clean_bins.parquet"
H2="results/substrate/h5ad_2um/$SLIDE.h5ad"
H8="results/substrate/h5ad_8um/$SLIDE.h5ad"
O2="$OUT/h5ad_2um_CMFLAG_clean"; O8="$OUT/h5ad_8um_CMFLAG_clean"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date); slide=$SLIDE flags=$FL"
"$PY" -u scripts/substrate/cmflag/build_cmflag_clean_h5ads.py "$SLIDE" \
      --h5ad-2um "$H2" --h5ad-8um "$H8" \
      --flags "$FL" --out-2um "$O2" --out-8um "$O8" --summary-dir "$OUT/summary"
rm -rf "$TMPDIR"
echo "done $(date)"
