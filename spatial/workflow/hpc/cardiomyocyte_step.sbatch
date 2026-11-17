#!/bin/bash
# One step of pipeline `cardiomyocyte` (Fig 1 j, Fig 3 d-l, ED6 d-f); the steps are listed in
# scripts/cardiomyocyte/run_cardiomyocyte.sh. Settings: config/cardiomyocyte.yaml; outputs: results/cardiomyocyte/.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" [--mem=..] workflow/hpc/cardiomyocyte_step.sbatch STEP [SLIDE]
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
# CM_ENV picks the environment (default cardiomyocyte); step fig1j_locate needs OpenCV: CM_ENV=substrate_cmflag.
# Memory: msearch 48G (a full 2 um h5ad is held in RAM), atlas_pb 32G, the other steps up to 24G.
#SBATCH --job-name=hhca_cm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
STEP="${1:?usage: sbatch workflow/hpc/cardiomyocyte_step.sbatch STEP [SLIDE]}"
SLIDE="${2:-}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" logs
source workflow/hpc/env.sh
PY="$(hhca_env "${CM_ENV:-cardiomyocyte}")/bin/python"; export PY
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date); step=$STEP slide=$SLIDE"
"$PY" -c "import numpy, scipy, pandas, anndata; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'anndata', anndata.__version__)"
bash scripts/cardiomyocyte/run_cardiomyocyte.sh "$STEP" "$SLIDE"
rm -rf "$TMPDIR"
echo "done $(date)"
