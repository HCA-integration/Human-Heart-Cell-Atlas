#!/bin/bash
# Xenium downstream analyses (Fig 6 c-l, ED9 a-c, ED6 i-j): label tables, per-core features, statistics and
# panels (scripts/xenium/downstream/run_downstream.sh; settings in config/xenium_downstream.yaml).
# Needs results/xenium/{healthy,rejection}/xenium_cellpin_labels.csv (xenium_cellpin.sbatch) and the extraction
# of GSE290577_heart_spatial.rds (xenium_inputs.sbatch).
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/xenium_downstream.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_xenium_downstream
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$TMPDIR" logs results/xenium/downstream
source workflow/hpc/env.sh
PY="$(hhca_env xenium_downstream)/bin/python"; export PY
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
"$PY" -m pip freeze > results/xenium/downstream/env_freeze.txt
bash scripts/xenium/downstream/run_downstream.sh
rm -rf "$TMPDIR"
echo "done $(date)"
