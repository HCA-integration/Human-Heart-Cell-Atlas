#!/bin/bash
# Build the environment of pipeline `nerve_macrophage` (envs/nerve_macrophage.yml -> .envs/nerve_macrophage)
# before the pipeline job starts.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/nerve_macrophage_env.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_nm_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR" logs results/nerve_macrophage
source workflow/hpc/env.sh
PREFIX="$(hhca_env nerve_macrophage)"
"$PREFIX/bin/python" -c "import numpy, scipy, pandas, sklearn, skimage, scanpy, anndata, statsmodels, tifffile, drug2cell, matplotlib, yaml; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'scanpy', scanpy.__version__, 'anndata', anndata.__version__, 'statsmodels', statsmodels.__version__)"
MAMBA_ROOT_PREFIX="$REPO/.tools/mamba_root" "$(hhca_mamba)" list -p "$PREFIX" > "$REPO/results/nerve_macrophage/env_list.txt" 2>/dev/null || true
rm -rf "$TMPDIR"
echo "env ready: $PREFIX"
