#!/bin/bash
# FICTURE decode of one Visium HD slide at 2 um with the fixed Level-3 model of substrate_model.sbatch:
# input files, seeded minibatch, transform, slda_decode. One array task per slide, in the order of `slides`
# in config/substrate.yaml. MODEL=<tsv> selects another model (default results/substrate/model/atlas_L3_pseudobulk.tsv).
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" --array=0-10 workflow/hpc/substrate_decode.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_substrate_decode
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$TMPDIR"
source workflow/hpc/env.sh
PY="$(hhca_env substrate)/bin/python"
FICTURE="$(hhca_env ficture)/bin/ficture"
CFG=config/substrate.yaml
OUT="$REPO/$($PY scripts/substrate/config_get.py $CFG key roots.results)"
eval "$($PY scripts/substrate/config_get.py $CFG slides)"
read -r -a ARR <<< "$SLIDES"
SLIDE_NAME="${SLIDE:-${ARR[${SLURM_ARRAY_TASK_ID:-0}]}}"
eval "$($PY scripts/substrate/config_get.py $CFG slide "$SLIDE_NAME")"
MODEL="${MODEL:-$OUT/model/atlas_L3_pseudobulk.tsv}"
RUN="$OUT/decode/$SLIDE_NAME"
echo "job ${SLURM_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname), $(date)"
echo "slide=$SLIDE_NAME builder=$BUILDER thread=$THREAD seed=$SEED sq2=$SQ2 model=$MODEL"
case "$BUILDER" in
  stock)  bash scripts/substrate/run_ficture_decode.sh "$SQ2" "$RUN" "$MODEL" "$THREAD" "$SEED" "$PY" "$FICTURE" ;;
  lowmem) bash scripts/substrate/run_ficture_decode_lowmem.sh "$SQ2" "$RUN" "$MODEL" "$THREAD" "$SEED" "$PY" "$FICTURE" ;;
  *) echo "unknown builder $BUILDER"; exit 2 ;;
esac
echo "done $(date)"
