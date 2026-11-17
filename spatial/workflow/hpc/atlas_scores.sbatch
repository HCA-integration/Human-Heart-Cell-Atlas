#!/bin/bash
# Per-donor gene-set and cytokine-programme scores streamed from the HHCA atlas, pooled to 280 donors, and the
# age effects of the cytokine programmes (Fig 2 j-l, Fig 5 m-n).
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/atlas_scores.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_atlas_scores
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
ATLAS="${HHCA_ATLAS:-/path/to/atlas/final_cleaned_hhca_annotated.h5ad}"
OUT="$REPO/results/atlas"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$TMPDIR"

source workflow/hpc/env.sh
PY="$(hhca_env atlas)/bin/python"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
"$PY" -c "import numpy, scipy, pandas, h5py; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'h5py', h5py.__version__)"

"$PY" scripts/atlas/donor_geneset_scores.py --atlas "$ATLAS" --genesets resources/atlas/ALL_genesets.json \
      --out "$OUT/per_donor_celltype_scores.npz"
"$PY" scripts/atlas/donor_cytokine_scores.py --atlas "$ATLAS" --programs resources/atlas/cytokine_programs.json \
      --outdir "$OUT/cytokine"
"$PY" scripts/atlas/pool_duplicate_donors.py --npz "$OUT/per_donor_celltype_scores.npz" \
      --map resources/atlas/donor_map.csv --out "$OUT/per_donor_celltype_scores_280donors.npz"
"$PY" scripts/atlas/cytokine_age_effects.py --cytokine-npz "$OUT/cytokine/per_donor_cytokine_scores.npz" \
      --celltype-npz "$OUT/per_donor_celltype_scores.npz" --stream-table "$OUT/cytokine/age_cytokine_effects.csv" \
      --map resources/atlas/donor_map.csv --out-table "$OUT/age_cytokine_effects_280.csv" \
      --out-npz "$OUT/per_donor_cytokine_scores_280.npz"
echo "done $(date)"
