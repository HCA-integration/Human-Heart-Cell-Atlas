#!/bin/bash
# Nerves and perineural macrophages on the 11 Visium HD slides: data of Fig 5 d, e, g, h, k, l and the H&E field
# extracts of Fig 5 b, c, f, j. Settings: config/nerve_macrophage.yaml; outputs: results/nerve_macrophage/.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/nerve_macrophage.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
# Only some steps (nerves d e gh kl fields): add --export=ALL,STEPS="gh fields"
#SBATCH --job-name=hhca_nm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" logs
source workflow/hpc/env.sh
PY="$(hhca_env nerve_macrophage)/bin/python"
CFG="$PY scripts/nerve_macrophage/read_config.py"
OUT="results/nerve_macrophage"; mkdir -p "$OUT"
FO="$($CFG substrate.factor_order)"
DEC="$($CFG substrate.decode)"
H5="$($CFG substrate.h5ad_2um)"
SR="$($CFG substrate.spaceranger)"; SRAH="$($CFG substrate.spaceranger_ah)"
CAL="$($CFG substrate.caliber)"
read -r -a BERLIN <<< "$($CFG slides.berlin)"; read -r -a AH <<< "$($CFG slides.ah)"
SLIDES=("${BERLIN[@]}" "${AH[@]}")
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
run() { echo; echo "## $*"; "$@" || echo "FAILED ($?): $*"; }
STEPS="${STEPS:-nerves d e gh kl fields}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
echo "steps: $STEPS"

# ---- nerves: bins, per-nerve pseudobulks ----
PB=()
for s in "${SLIDES[@]}"; do
  want nerves && run "$PY" -u scripts/nerve_macrophage/nerves.py --slide "$s" --h5ad "${H5//\{s\}/$s}" --outdir "$OUT/nerves"
  PB+=("$OUT/nerves/${s}_nerve_pseudobulk.h5ad")
done
# ---- Fig 5 d: per nerve, against the same nerve shapes at random tissue positions ----
if want d; then
  run "$PY" -u scripts/nerve_macrophage/nerve_vessel_distance_pernerve.py --h5ad "$H5" --nerves-dir "$OUT/nerves" \
      --slides "${SLIDES[@]}" --placements "$($CFG proximity.placements_per_nerve)" --seed "$($CFG proximity.seed)" \
      --out "$OUT/d/per_nerve.npz"
fi
# ---- Fig 5 e ----
want e && run "$PY" -u scripts/nerve_macrophage/myelination_size.py --pseudobulk "${PB[@]}" --slides "${SLIDES[@]}" \
    --out "$OUT/e/per_nerve.csv"
# ---- Fig 5 g, h: drug2cell scores, myelin genes removed from the target sets; 18,085-gene universe
#      (inner join of full-panel pseudobulks) ----
if want gh; then
  run "$PY" -u scripts/nerve_macrophage/cipn_drug2cell.py --pseudobulk "${PB[@]}" --outdir "$OUT/gh"
fi

# ---- Fig 5 k, l ----
if want kl; then
if [ -f "$CAL" ]; then CALARG="$CAL"; else CALARG=none; echo "no calibre table ($CAL): vessel-structure columns NaN"; fi
run "$PY" -u scripts/nerve_macrophage/macrophage_pseudocells.py --decode "$DEC" --spaceranger "$SR" --spaceranger-ah "$SRAH" \
    --factor-order "$FO" --caliber "$CALARG" --outdir "$OUT/macrophage"
run "$PY" -u scripts/nerve_macrophage/macrophage_covariates.py --cache "$OUT/macrophage/cache/Macrophage_pseudocells_allslides.h5ad" \
    --h5ad "$H5" --outdir "$OUT/macrophage"
run "$PY" -u scripts/nerve_macrophage/nerve_clusters.py --cov "$OUT/macrophage/tables/macrophage_percell_covariates.csv" \
    --h5ad "$H5" --outdir "$OUT/macrophage"
run "$PY" -u scripts/nerve_macrophage/perineural_de.py --cache "$OUT/macrophage/cache/Macrophage_pseudocells_allslides.h5ad" \
    --distances "$OUT/macrophage/tables/percell_nerve_distances_by_k.csv" --outdir "$OUT/kl"
fi

# ---- Fig 5 b, c, f, j field extracts ----
want fields && run "$PY" -u scripts/nerve_macrophage/fields.py --run-dir "$OUT"
rm -rf "$TMPDIR"
echo "done $(date)"
