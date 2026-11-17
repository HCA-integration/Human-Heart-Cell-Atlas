#!/bin/bash
# Xenium, step 3: train CellPin, transfer Level-3 labels by distance-weighted kNN (K = 25) in the CellPin
# latent, add Level 4 (healthy: within Level 3; rejection: lymphoid lineage x proliferation) and assemble the
# label table read by the downstream analyses (ED6 i-j, Fig 6, ED9).
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/xenium_cellpin.sbatch healthy|rejection
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_xenium_cellpin
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail

COHORT="${1:?usage: sbatch workflow/hpc/xenium_cellpin.sbatch healthy|rejection}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
OUT="$REPO/results/xenium/$COHORT"
ANN="$OUT/annotation"
NT="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$ANN" "$TMPDIR"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date), cohort $COHORT"

source workflow/hpc/env.sh
PY="$(hhca_env xenium_cellpin)/bin/python"
# training and label transfer are skipped when their outputs exist (FORCE=1 reruns everything)
have() { [ "${FORCE:-0}" != 1 ] && [ -s "$1" ] && echo "skip: $1 exists"; }

if [ "$COHORT" = healthy ]; then
  have "$OUT/cellpin_imputed_full.h5ad" || \
  "$PY" scripts/xenium/train_cellpin.py --ref "$OUT/reference_subset.h5ad" --xenium "$OUT/xenium_genes.h5ad" \
        --outdir "$OUT" --tag full --accelerator cpu
  have "$ANN/annotation_summary.json" || \
  "$PY" scripts/xenium/transfer_labels.py --ref "$OUT/reference_subset.h5ad" --xenium "$OUT/xenium_genes.h5ad" \
        --model "$OUT/cellpin_model_full.pt" --imputed "$OUT/cellpin_imputed_full.h5ad" --outdir "$ANN" \
        --l4-col cell_type_L4
  "$PY" scripts/xenium/assemble_labels.py healthy --annotation "$ANN" --config config/xenium.yaml \
        --xenium "$OUT/xenium_genes.h5ad" --out "$OUT/xenium_cellpin_labels.csv"
else
  have "$OUT/cellpin_model_full.pt" || \
  "$PY" scripts/xenium/train_cellpin.py --ref "$OUT/reference_subset.h5ad" --xenium "$OUT/xenium_genes.h5ad" \
        --outdir "$OUT" --tag full --accelerator cpu --skip-impute
  have "$ANN/annotation_summary.json" || \
  "$PY" scripts/xenium/transfer_labels.py --ref "$OUT/reference_subset.h5ad" --xenium "$OUT/xenium_genes.h5ad" \
        --model "$OUT/cellpin_model_full.pt" --outdir "$ANN"
  "$PY" scripts/xenium/lymphoid_level4.py --ref "$OUT/reference_subset.h5ad" --extra "$OUT/reference_extra_lymphoid.h5ad" \
        --xenium "$OUT/xenium_genes.h5ad" --model "$OUT/cellpin_model_full.pt" --latent "$ANN/latent.npz" \
        --pred "$ANN/per_cell_predictions.csv" --config config/xenium.yaml --outdir "$OUT/lymphoid_l4"
  "$PY" scripts/xenium/assemble_labels.py rejection --annotation "$ANN" --config config/xenium.yaml \
        --obs "$OUT/extracted/obs.csv" --lymphoid "$OUT/lymphoid_l4/xenium_lymphoid8_cellpin.csv" \
        --out "$OUT/xenium_cellpin_labels.csv"
fi
echo "done $(date)"
