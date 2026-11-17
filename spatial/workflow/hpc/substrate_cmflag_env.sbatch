#!/bin/bash
# Build the environment of the CMFLAG part of pipeline `substrate` before the per-slide jobs use it.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/substrate_cmflag_env.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_cmflag_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR" logs results/substrate/cmflag
source workflow/hpc/env.sh
PREFIX="$(hhca_env substrate_cmflag)"
"$PREFIX/bin/python" -c "import numpy, scipy, pandas, pyarrow, skimage, cv2, tifffile, scanpy, anndata, aicspylibczi, yaml"
rm -rf "$TMPDIR"
echo "env ready: $PREFIX"
