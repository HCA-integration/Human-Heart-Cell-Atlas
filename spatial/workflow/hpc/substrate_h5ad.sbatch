#!/bin/bash
# h5ads of one slide after substrate_certainty.sbatch: decode rows, 2 um bins with square_002um counts, 8 um
# bins. One array task per slide, in the order of `slides` in config/substrate.yaml.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" --array=0-10 workflow/hpc/substrate_h5ad.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_substrate_h5ad
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$TMPDIR"
source workflow/hpc/env.sh
PY="$(hhca_env substrate)/bin/python"
CFG=config/substrate.yaml
OUT="$REPO/$($PY scripts/substrate/config_get.py $CFG key roots.results)"
eval "$($PY scripts/substrate/config_get.py $CFG slides)"
read -r -a ARR <<< "$SLIDES"
S="${SLIDE:-${ARR[${SLURM_ARRAY_TASK_ID:-0}]}}"
eval "$($PY scripts/substrate/config_get.py $CFG slide "$S")"
DEC="$OUT/decode/$S/output/atlas_L3.decode.prj_12.r_4.0_5.0.pixel.tsv.gz"
FO="$OUT/model/atlas_L3_factor_order.json"
CERT="$OUT/certainty/tables/${S}_certainty_columns.tsv.gz"
CERT_ARGS=(); [ -s "$CERT" ] && CERT_ARGS=(--certainty "$CERT")
SQ8_ARGS=()
# shellcheck disable=SC2153  # SITE, REGION, SQ2 and SQ8 are assigned by the eval of the slide settings above
[ "$SITE" = berlin ] && SQ8_ARGS=(--square-008um "$SQ8")
echo "job ${SLURM_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname), $(date): $S ($SITE) certainty: ${CERT_ARGS[*]:-none}"
mkdir -p "$OUT/h5ad_pixels/$S" "$OUT/h5ad_2um" "$OUT/h5ad_8um"
cd scripts/substrate
"$PY" -u build_pixel_h5ad.py --slide "$S" --decode "$DEC" --factor-order "$FO" \
  --out "$OUT/h5ad_pixels/$S/atlas_L3_2um_pixels.h5ad" --force
"$PY" -u build_2um_h5ad.py --slide "$S" --decode "$DEC" --square-002um "$SQ2" --factor-order "$FO" \
  --region "$REGION" "${CERT_ARGS[@]}" --out "$OUT/h5ad_2um/$S.h5ad"
"$PY" -u build_8um_h5ad.py --slide "$S" --decode "$DEC" --square-002um "$SQ2" "${SQ8_ARGS[@]}" \
  --factor-order "$FO" "${CERT_ARGS[@]}" --out "$OUT/h5ad_8um/$S.h5ad"
echo "done $(date)"
