#!/bin/bash
# Deconvolution, ground-truth simulation from the Visium HD decodes.
#   build  one HD section per array task: 55 um pseudo-spots from the 2 um FICTURE decode (build_sim.py)
#   prep   subsample to at most 5,000 spots per section (prep_sim_inputs.py)
#   score  score every method against the known composition (score_gt.py)
# The methods themselves run through deconvolution_{gpu,cpu}.sbatch with DATASET=gtsim.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" --array=0-3 workflow/hpc/deconvolution_gtsim.sbatch build
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_deconv_gtsim
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
STEP="${1:?usage: deconvolution_gtsim.sbatch build|prep|score}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
source workflow/hpc/deconvolution_common.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR" logs "$D/gtsim/sim_data" "$D/gtsim/data"
source workflow/hpc/env.sh
PY="$(hhca_env deconv_cpu)/bin/python"
echo "job ${SLURM_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname), $(date): $STEP"
case "$STEP" in
  build) S="${GT_SLIDES[${SLURM_ARRAY_TASK_ID:-0}]}"
         "$PY" scripts/deconvolution/build_sim.py --slide-dir "results/substrate/decode/$S" \
               --factor-order "$FACTOR_ORDER" --out "$D/gtsim/sim_data" "$S" ;;
  prep)  "$PY" scripts/deconvolution/prep_sim_inputs.py --sim "$D/gtsim/sim_data" --out "$D/gtsim/data" --max-spots 5000 --seed 0 ;;
  score) "$PY" scripts/deconvolution/score_gt.py --outputs "$D/gtsim/outputs" --truth "$D/gtsim/data" \
               --tables "$D/gtsim/tables" --figures "$D/gtsim/figures" --factor-order "$FACTOR_ORDER" --refs hhca ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "done $(date)"
