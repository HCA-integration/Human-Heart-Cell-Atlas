#!/bin/bash
# Builders of the curated atlas resources in resources/atlas/: the cytokine programmes from the public huCIRA
# dictionary, and the duplicate donors behind the donor map (291 atlas donor ids -> 280 donors), found from the
# data. ALL_genesets.json is built at the end of atlas_hormones.sbatch.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/atlas_resources.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_atlas_resources
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
ATLAS="${HHCA_ATLAS:-/path/to/atlas/final_cleaned_hhca_annotated.h5ad}"
# public huCIRA dictionary as saved by hucira 0.1.3 load_human_cytokine_dict()
# (https://cdn.parsebiosciences.com/gigalab/10m/DEGs.csv, index reset)
HUCIRA="${HHCA_HUCIRA_DICT:-/path/to/scratch/hucira_data/human_cytokine_dict.csv}"
OUT="$REPO/results/atlas/resources"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$TMPDIR"

source workflow/hpc/env.sh
PY="$(hhca_env atlas_h2c)/bin/python"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"

echo "== cytokine programmes"
"$PY" -u scripts/atlas/build_cytokine_programs.py --dict "$HUCIRA" --match resources/atlas/hucira_celltype_match.yaml \
      --outdir "$OUT/cytokine"
echo "== duplicate donors"
"$PY" -u scripts/atlas/find_duplicate_donors.py --atlas "$ATLAS" --map resources/atlas/donor_map.csv \
      --outdir "$OUT/duplicate_donors"
echo "done $(date)"
