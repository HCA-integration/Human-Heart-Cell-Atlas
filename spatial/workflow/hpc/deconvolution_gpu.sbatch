#!/bin/bash
# Deconvolution, GPU methods: one step on one dataset (array index = slide; the training steps take no array).
#   STEP    c2l_train | c2l | stereo_train | stereo | spotiphy
#   DATASET benchmark (11 slides) | gtsim (4 simulated HD sections)
# Usage (from the repository root): sbatch "${SBATCH_GPU[@]}" workflow/hpc/deconvolution_gpu.sbatch c2l_train benchmark
#                                   sbatch "${SBATCH_GPU[@]}" --array=0-10%2 workflow/hpc/deconvolution_gpu.sbatch c2l benchmark
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_deconv_gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
STEP="${1:?usage: deconvolution_gpu.sbatch STEP DATASET}"; DATASET="${2:?usage: deconvolution_gpu.sbatch STEP DATASET}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
source workflow/hpc/deconvolution_common.sh
deconv_dataset "$DATASET"
export PYTHONUNBUFFERED=1 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR" logs "$OUT" "$MODELS"
source workflow/hpc/env.sh
SLIDE="${SLIDES[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "job ${SLURM_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname), $(date): $STEP $DATASET $SLIDE"
nvidia-smi -L 2>/dev/null | head -2 || true
COMMON=(--ref "$REF_H5" --data "$DATA" --out "$OUT" --panel "$PANEL" --celltypes "$CELLTYPES" --seed "$SEED")
C2L_EPOCHS=15000; [ "$DATASET" = gtsim ] && C2L_EPOCHS=8000
case "$STEP" in
  c2l_train)    PY="$(hhca_env deconv_cell2location)/bin/python"
                "$PY" scripts/deconvolution/c2l_train_ref.py "${COMMON[@]}" --models "$MODELS" --epochs 250 ;;
  c2l)          PY="$(hhca_env deconv_cell2location)/bin/python"
                "$PY" scripts/deconvolution/run_cell2location.py "${COMMON[@]}" --models "$MODELS" --epochs "$C2L_EPOCHS" --ncells 8 "$SLIDE" ;;
  stereo_train) PY="$(hhca_env deconv_cell2location)/bin/python"
                "$PY" scripts/deconvolution/stereo_train_ref.py "${COMMON[@]}" --models "$MODELS" --epochs 100 ;;
  stereo)       PY="$(hhca_env deconv_cell2location)/bin/python"
                "$PY" scripts/deconvolution/run_stereoscope.py "${COMMON[@]}" --models "$MODELS" --epochs 2000 "$SLIDE" ;;
  spotiphy)     PY="$(hhca_env deconv_spotiphy)/bin/python"
                "$PY" scripts/deconvolution/run_spotiphy.py "${COMMON[@]}" --epochs 8000 "$SLIDE" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "done $(date)"
