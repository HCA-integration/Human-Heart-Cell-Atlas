#!/bin/bash
# hormone2cell analyses on the HHCA atlas: producer and receiver strength per donor subgroup, the relevance gate
# and the donor-level sex differences (Fig 2 h-i), the producer table of the vascular relay (Fig 4 l), and
# ALL_genesets.json from public sources. Settings: config/atlas_hormones.yaml.
# Needs results/atlas/per_donor_celltype_scores.npz (atlas_scores.sbatch) for donor sex and age.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/atlas_hormones.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_atlas_hormones
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
ATLAS="${HHCA_ATLAS:-/path/to/atlas/final_cleaned_hhca_annotated.h5ad}"
# Level-4 atlas profile (mean CP10k per Final Level 4 type on the 14,016-gene FICTURE panel) and nuclei per
# Level-4 type
PROFILE="${HHCA_L4_PROFILE:-/path/to/hhca_precomputed/scanpy_analysis/ficture/newatlas/output/atlas_v2_L4_pseudobulk.tsv}"
NCELLS="${HHCA_L4_NCELLS:-/path/to/hhca_precomputed/scanpy_analysis/ficture/newatlas/tables/atlas_v2_L4_ncells.csv}"
OUT="$REPO/results/atlas/hormones"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$TMPDIR" "$REPO/results/atlas/resources"

source workflow/hpc/env.sh
PY="$(hhca_env atlas_h2c)/bin/python"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"

# The hormone and drug-target dictionaries ship inside the hormone2cell and drug2cell packages; stop if the
# installed files are not the releases this analysis was written for.
"$PY" - <<'PYEOF'
import hashlib, importlib.util, os
SHA256 = {
    "hormone2cell": {
        "hormone_strength.py": "3a52437135745950eba034b25b85989181384313cdb64313b0999bbad0cb21a8",
        "utils.py": "5e29b8b143a2399841761ae5fa29df75fd8ad6dd5a1cd95ad3506d23defe9421",
        "hormone_producing_definition_v2.0.0.pkl": "e50d3575ebd804a5bf6590ca2891e642e3ebc8e9b51d55f38337e5e1877cd2a7",
        "hormone_receptor_definition_v2.0.0.pkl": "9a37a99076bc05652adae691f2e45840a02023344e2e5439e42113cde9df930f"},
    "drug2cell": {"drug-target_dicts.pkl": "4a53e3e1dec08b339fafd5b055f2b51f8e4b201d153efa94153f97cc47bc9158"},
}
for pkg, files in SHA256.items():
    folder = importlib.util.find_spec(pkg).submodule_search_locations[0]
    for name, expected in files.items():
        found = hashlib.sha256(open(os.path.join(folder, name), "rb").read()).hexdigest()
        assert found == expected, f"{pkg}/{name}: sha256 {found}, expected {expected}"
print("hormone2cell and drug2cell package files as expected")
PYEOF

echo "== stratified hormone2cell"
"$PY" -u scripts/atlas/hormone2cell_stratified.py --atlas "$ATLAS" --profile "$PROFILE" \
      --outdir "$OUT/data_stratified" --ckptdir "$OUT/_ckpt_stratified"
echo "== producer table (Fig 4 l)"
"$PY" -u scripts/atlas/hormone2cell_producer_L4.py --profile "$PROFILE" --ncells "$NCELLS" --outdir "$OUT/collab_hormone"
echo "== hormone gene-set tables"
"$PY" -u scripts/atlas/hormone_genesets_export.py --outdir "$OUT/genesets"
echo "== relevance gate"
"$PY" -u scripts/atlas/hormone_relevance_gate.py --data "$OUT/data_stratified" --outdir "$OUT/g7_sexdiff"
echo "== LV donor tensor (291 donor ids)"
"$PY" -u scripts/atlas/lv_hormone_tensor.py --atlas "$ATLAS" --gated "$OUT/g7_sexdiff" --genesets "$OUT/genesets" \
      --outdir "$OUT/sexdiff_291"
echo "== donor metadata"
"$PY" -u scripts/atlas/donor_meta.py --npz results/atlas/per_donor_celltype_scores.npz --out results/atlas/donor_meta.csv
echo "== pooling to 280 donors"
"$PY" -u scripts/atlas/pool_lv_tensor.py --npz "$OUT/sexdiff_291/lv_donor_L2_gene_meanNorm.npz" \
      --map resources/atlas/donor_map.csv --out "$OUT/sexdiff_280/lv_donor_L2_gene_meanNorm.npz"
echo "== sex-difference significance (280 donors)"
"$PY" -u scripts/atlas/hormone_sexdiff_significance.py --tensor "$OUT/sexdiff_280/lv_donor_L2_gene_meanNorm.npz" \
      --genesets-json "$OUT/sexdiff_291/hormone_genesets.json" --gated "$OUT/g7_sexdiff" \
      --donor-meta results/atlas/donor_meta.csv --out "$OUT/sexdiff_280/hormone_sexdiff_significance_ALLDOTS.csv"
echo "== ALL_genesets.json"
"$PY" -u scripts/atlas/build_genesets.py --atlas "$ATLAS" --panel "$PROFILE" \
      --producer-table "$OUT/collab_hormone/producer_strength_L4_long.csv" --axes resources/atlas/cardiac_axes.yaml \
      --out results/atlas/resources/ALL_genesets.json
echo "done $(date)"
