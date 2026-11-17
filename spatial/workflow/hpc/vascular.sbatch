#!/bin/bash
# Pipeline `vascular` on the 11 Visium HD sections: vessel structures, calibre, wall composition and markers,
# RAAS model and drug-target calibre selectivity, then the panels Fig 4 g-k.
# Inputs: config/vascular.yaml. Outputs: results/vascular/ and results/figures/Fig4/panel_{g,h,i,j,k}.{png,pdf}.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/vascular.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_vascular
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" logs results/vascular
source workflow/hpc/env.sh
# keep the package cache inside the checkout, so a lock left in a shared cache cannot block the build
export CONDA_PKGS_DIRS="$REPO/.tools/mamba_root/pkgs"
PY="$(hhca_env vascular)/bin/python"; export PY
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
"$PY" -c "import numpy, scipy, pandas, statsmodels, scanpy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'statsmodels', statsmodels.__version__, 'scanpy', scanpy.__version__)"
scripts/vascular/run_vascular.sh
for p in g h i j k; do "$PY" "figures/Fig4/panel_$p.py"; done
rm -rf "$TMPDIR"
echo "done $(date)"
