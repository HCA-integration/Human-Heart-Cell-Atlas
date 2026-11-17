#!/bin/bash
# Deconvolution, CPU methods: one method on one slide of a dataset (array index = slide).
#   STEP    ucaspatial | RCTD | SPOTlight | gt_nnls (ground-truth simulation only)
#   DATASET benchmark (11 slides) | gtsim (4 simulated HD sections)
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" --array=0-10%3 workflow/hpc/deconvolution_cpu.sbatch RCTD benchmark
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_deconv_cpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
STEP="${1:?usage: deconvolution_cpu.sbatch STEP DATASET}"; DATASET="${2:?usage: deconvolution_cpu.sbatch STEP DATASET}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
source workflow/hpc/deconvolution_common.sh
deconv_dataset "$DATASET"
# UCASpatial's NMF/NNLS results depend on the BLAS thread count on some slides, so it is fixed at 8
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR" logs "$OUT"
source workflow/hpc/env.sh
SLIDE="${SLIDES[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "job ${SLURM_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname), $(date): $STEP $DATASET $SLIDE"
COMMON=(--ref "$REF_H5" --data "$DATA" --out "$OUT" --panel "$PANEL" --celltypes "$CELLTYPES" --seed "$SEED")
case "$STEP" in
  ucaspatial) PY="$(hhca_env deconv_cpu)/bin/python"
              "$PY" scripts/deconvolution/run_ucaspatial.py "${COMMON[@]}" --random-seed 12345 "$SLIDE" ;;
  RCTD|SPOTlight)
              RP="$(hhca_env deconv_r)"
              R_MAKEVARS_USER=/dev/null "$RP/bin/Rscript" scripts/deconvolution/run_r_method.R "$STEP" "$SLIDE" \
                  "$REFDIR" "$DATA" "$OUT" "$CELLTYPES" "$SEED" ;;
  gt_nnls)    PY="$(hhca_env deconv_cpu)/bin/python"
              OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                "$PY" scripts/deconvolution/gt_nnls.py "${COMMON[@]}" "$SLIDE" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "done $(date)"
