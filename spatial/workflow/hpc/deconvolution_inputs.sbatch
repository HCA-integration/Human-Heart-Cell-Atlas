#!/bin/bash
# Deconvolution, public inputs: download the Kanemaru et al. 2023 Visium objects (Fig 3 c, ED8 b-c).
# The Noseda Space Ranger outputs are fetched separately with scripts/deconvolution/download_noseda_mdc.sh.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/deconvolution_inputs.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_deconv_inputs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
source workflow/hpc/deconvolution_common.sh
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR" logs
source workflow/hpc/env.sh
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"

bash scripts/deconvolution/download_kanemaru.sh "$KANEMARU"
echo "done $(date)"
