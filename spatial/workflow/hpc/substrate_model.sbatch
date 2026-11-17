#!/bin/bash
# FICTURE model of pipeline `substrate`: the gene set (genes detected in one 800 um field of BO_HDZ) and the
# atlas gene x cell-type profiles at Level 4 and Level 3. Settings: config/substrate.yaml.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/substrate_model.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_substrate_model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
source workflow/hpc/env.sh
PY="$(hhca_env substrate)/bin/python"
CFG=config/substrate.yaml
OUT="$REPO/$($PY scripts/substrate/config_get.py $CFG key roots.results)"
ATLAS="$($PY scripts/substrate/config_get.py $CFG key roots.atlas_h5ad)"
GS_SLIDE="$($PY scripts/substrate/config_get.py $CFG key geneset.slide)"
CROP="$($PY scripts/substrate/config_get.py $CFG key geneset.crop_um)"
read -r -a CROP_ARGS <<< "$CROP"
eval "$($PY scripts/substrate/config_get.py $CFG slide "$GS_SLIDE")"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date); crop of $GS_SLIDE: $CROP"

# gene set: non-mitochondrial genes with >= 1 UMI in the geneset.crop_um crop (X0 X1 Y0 Y1, um) of geneset.slide
mkdir -p "$OUT/geneset"
"$PY" -u scripts/substrate/build_ficture_input.py --slide-dir "$SQ2" --out-dir "$OUT/geneset" --crop "${CROP_ARGS[@]}" \
  2>&1 | tee "$OUT/geneset/01_build_input.log"

# atlas profiles restricted to the gene set
mkdir -p "$OUT/model/tables"
"$PY" -u scripts/substrate/build_atlas_profiles.py --atlas "$ATLAS" --features "$OUT/geneset/feature.clean.tsv.gz" \
  --outdir "$OUT/model" --tabdir "$OUT/model/tables" 2>&1 | tee "$OUT/model/02_build_atlas_profiles.log"
echo "done $(date)"
