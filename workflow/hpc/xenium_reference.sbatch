#!/bin/bash
# Xenium, step 2: CellPin reference from the HHCA atlas (class-balanced subsample, at most 2000 nuclei per
# Level-3 type, seed 0; genes = panel ∩ atlas ∪ 3000 HVGs ∪ markers). The rejection run also writes every
# atlas T / NK / proliferating-lymphoid / ILC cell for the Level-4 lymphoid kNN.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/xenium_reference.sbatch healthy|rejection
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_xenium_reference
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail

COHORT="${1:?usage: sbatch workflow/hpc/xenium_reference.sbatch healthy|rejection}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
ATLAS="${HHCA_ATLAS:-/path/to/atlas/final_cleaned_hhca_annotated.h5ad}"
OUT="$REPO/results/xenium/$COHORT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$TMPDIR"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date), cohort $COHORT"

source workflow/hpc/env.sh
PY="$(hhca_env xenium_cellpin)/bin/python"

EXTRA=()
if [ "$COHORT" = rejection ]; then
  EXTRA=(--extra-l3 "T cells,NK cells,Lymphoid Cells (Proliferating),ILC" --extra-out "$OUT/reference_extra_lymphoid.h5ad")
fi
"$PY" scripts/xenium/build_cellpin_reference.py --atlas "$ATLAS" --xenium "$OUT/xenium_genes.h5ad" \
      --out "$OUT/reference_subset.h5ad" "${EXTRA[@]}"
echo "done $(date)"
