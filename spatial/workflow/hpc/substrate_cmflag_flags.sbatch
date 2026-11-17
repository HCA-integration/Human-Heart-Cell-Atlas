#!/bin/bash
# CMFLAG for one Visium HD slide: H&E registration and spillover flags, after substrate_h5ad.sbatch and
# substrate_cmflag_env.sbatch. Inputs per slide: config/substrate_cmflag.yaml.
# Outputs: results/substrate/cmflag/{flags,registration}/.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/substrate_cmflag_flags.sbatch SLIDE
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_cmflag
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=04:00:00
set -euo pipefail
SLIDE="${1:?usage: sbatch workflow/hpc/substrate_cmflag_flags.sbatch SLIDE}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TMPDIR" logs results/substrate/cmflag
source workflow/hpc/env.sh
PY="$(hhca_env substrate_cmflag)/bin/python"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date); slide=$SLIDE"
"$PY" -u scripts/substrate/cmflag/run_slide.py --config config/substrate_cmflag.yaml --slide "$SLIDE"
rm -rf "$TMPDIR"
echo "done $(date)"
