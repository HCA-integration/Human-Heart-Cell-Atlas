#!/bin/bash
# Deconvolution, method summaries (ED6 h and supporting tables), after the method jobs:
#   1. compare_methods.py: cross-method concordance, composition, runtime and marker-support tables, once on the
#      benchmark per-spot outputs behind ED6 g-h (tables_precomputed/) and once on the outputs of this run (tables/)
#   2. areafraction_L3.py: HD area fraction of the two LV-endo HD sections from the FICTURE decode
#   3. build_endo_LV_composition.py: endo-LV composition across modalities and method agreement with the HD area fraction
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/deconvolution_summary.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_deconv_summary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" || exit 1
source workflow/hpc/deconvolution_common.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR" logs "$D/tables" "$D/tables_precomputed" "$D/endo_LV" "$D/hd_areafraction"
source workflow/hpc/env.sh
PYN="$(hhca_env deconv_nnls)/bin/python"
PY="$(hhca_env deconv_cpu)/bin/python"
# settings for the scripts of the NNLS environment, which has no YAML reader
CFG="$TMPDIR/deconvolution.json"; "$PY" scripts/deconvolution/deconv_config.py > "$CFG"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
SVTAB="$BENCHTAB/scanpy_analysis/standard_visium"
CATAB="$BENCHTAB/scanpy_analysis/composition_atlas/tables"

echo "## 1. method summary tables (compare_methods.py)"
"$PYN" scripts/deconvolution/compare_methods.py --outputs "$PRECOMPUTED/outputs" --nnls-dir "$SVTAB/atlas_deconv/tables" \
      --tables "$D/tables_precomputed" --figures "$D/tables_precomputed/figures" --config "$CFG"
"$PYN" scripts/deconvolution/compare_methods.py --outputs "$D/outputs" --nnls-dir "$D/nnls/tables" \
      --tables "$D/tables" --figures "$D/tables/figures" --marker-scores "$D/marker_scores" --config "$CFG"

echo "## 2. HD area fraction, LV-endo sections"
"$PYN" scripts/deconvolution/areafraction_L3.py --decode-template "$DECODE_TEMPLATE" --factor-order "$FACTOR_ORDER" \
      --out "$D/hd_areafraction" --slides VS_CA_H0054LV_henricke_endo,VS_CA_H0063LV_henricke_endo

echo "## 3. endo-LV composition"
# area fraction of the two endo sections; other slides' columns are not read by the script
"$PYN" scripts/deconvolution/build_endo_LV_composition.py --outputs "$D/outputs" --nnls-dir "$D/nnls/tables" \
      --areafraction "$D/hd_areafraction/lv_areafraction_L3_perslide.csv" --units "$CATAB/lv_composition_units_pooled.csv" \
      --sn "$CATAB/hca_ref_composition_ventricular.csv" --leaderboard "$D/gtsim/tables/leaderboard_hhca.csv" \
      --region-effect "$D/gtsim/tables/region_effect_hhca.csv" --factor-order "$FACTOR_ORDER" --out "$D/endo_LV"
echo "methods ranked by RMSE vs HD area fraction:"; cut -d, -f1,2 "$D/endo_LV/tables/method_agreement.csv"
echo "done $(date)"
