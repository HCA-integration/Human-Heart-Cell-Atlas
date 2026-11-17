#!/bin/bash
# Pipeline `cardiomyocyte` (Fig 1 j, Fig 3 d-l, ED6 d-f): runs one step per call.
# Usage (from the repository root): scripts/cardiomyocyte/run_cardiomyocyte.sh STEP [SLIDE]
# Settings: config/cardiomyocyte.yaml; outputs: results/cardiomyocyte/ (<out> below); PY = python of
# envs/cardiomyocyte.yml. On the cluster each step is one job of workflow/hpc/cardiomyocyte_step.sbatch.
# Steps:
#   env             check that the environment imports
#   atlas_pb        atlas Level-2 / Level-3 pseudobulk (input of ed6e and fig3i_select)     -> <out>/atlas/
#   msearch         Fig 3 i: 45-bin transmural profiles of all genes, module selection      -> <out>/fig3i/
#   fig3i_select    Fig 3 i: snRNA cardiomyocyte specificity (after atlas_pb), filter rules -> <out>/fig3i/fig3i_cm_filter_*
#   niche SLIDE     Fig 3 f-h: 10 um niche bins of one slide                                -> <out>/fig3fgh/bins/
#   solidify        Fig 3 f-h: NPPA / NPPB high-vs-low tests over the 11 slides             -> <out>/fig3fgh/
#   stress          Fig 3 d: stress score on the fields of config fig3d.fields              -> <out>/fig3d/
#   pb2um           Fig 3 l: transmural pseudobulk, 2 donors and 3 donors                   -> <out>/fig3l/pb2um, pb2um_3donor/
#   fig3l           Fig 3 l: transcript slopes and DVP protein concordance (after pb2um)    -> <out>/fig3l/fig3l_concordance_*
#   label_pb SLIDE  ED6 d, e: counts per Level-3 label of one slide                         -> <out>/ed6e/labels/
#   ed6e            ED6 e: factor-marker concordance (after atlas_pb and label_pb)          -> <out>/ed6e/
#   fields          Fig 3 e window, ED6 f raster, Fig 1 j fields                            -> <out>/fields/
#   fig1j_locate    Fig 1 j: locate the gallery crops on the H&E (needs OpenCV)             -> <out>/fig1j/
set -euo pipefail
STEP="${1:?step}"; SLIDE="${2:-}"
CFG=config/cardiomyocyte.yaml
PY="${PY:-python}"
cfg() { "$PY" scripts/cardiomyocyte/cm_config.py "$CFG" "$@"; }
OUT=results/cardiomyocyte
mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] step=$STEP slide=$SLIDE"

case "$STEP" in
  msearch)
    "$PY" -u scripts/cardiomyocyte/compute_marker_search.py --out-dir "$OUT/fig3i"
    "$PY" -u scripts/cardiomyocyte/select_transmural_genes.py --tab "$OUT/fig3i"
    ;;
  niche)
    "$PY" -u scripts/cardiomyocyte/run_niche_slide.py "$SLIDE" --decode "$(cfg path decode "$SLIDE")" \
        --transcripts "$(cfg path transcripts "$SLIDE")" --factor-order "$(cfg path factor_order)" \
        --out-dir "$OUT/fig3fgh/bins" --flags "$(cfg path niche_flags "$SLIDE")"
    ;;
  solidify)
    "$PY" -u scripts/cardiomyocyte/solidify.py --bins-dir "$OUT/fig3fgh/bins" --out-dir "$OUT/fig3fgh"
    ;;
  pb2um)
    "$PY" -u scripts/cardiomyocyte/transmural_pseudobulk.py --out-dir "$OUT/fig3l/pb2um" \
        --donors H0054 H0063 --layers endo myo epi
    "$PY" -u scripts/cardiomyocyte/transmural_pseudobulk.py --out-dir "$OUT/fig3l/pb2um_3donor"
    ;;
  stress)
    "$PY" -c "import yaml,json; c=yaml.safe_load(open('$CFG'))['fig3d']; json.dump(c['genesets'], open('$OUT/fig3d_genesets.json','w')); json.dump(c['fields'], open('$OUT/fig3d_fields.json','w'))"
    "$PY" -u scripts/cardiomyocyte/stress_score.py --genesets "$OUT/fig3d_genesets.json" \
        --out "$OUT/fig3d/stress.npz" --fields "$OUT/fig3d_fields.json"
    ;;
  atlas_pb)
    "$PY" -u scripts/cardiomyocyte/atlas_pseudobulk.py --atlas "$(cfg get atlas.h5ad)" --out-dir results/cardiomyocyte/atlas \
        --layer "$(cfg get atlas.layer)" --chunk "$(cfg get atlas.chunk)"
    ;;
  label_pb)
    "$PY" -u scripts/cardiomyocyte/label_pseudobulk.py --h5ad "$(cfg path h5ad_2um "$SLIDE")" \
        --out "$OUT/ed6e/labels/${SLIDE}_label_pseudobulk.csv.gz"
    ;;
  fields)
    "$PY" -u scripts/cardiomyocyte/extract_fields.py --config "$CFG" --out-dir "$OUT/fields"
    ;;
  fig1j_locate)
    "$PY" -u scripts/cardiomyocyte/fig1j_locate.py --config "$CFG" --gallery-dir "$(cfg get roots.inputs)/cardiomyocyte/fig1j_gallery" \
        --out-dir "$OUT/fig1j"
    ;;
  ed6e)
    "$PY" -u scripts/cardiomyocyte/ed6e_concordance.py --config "$CFG" --atlas-mean results/cardiomyocyte/atlas/mean_normcounts_L3.parquet \
        --labels-dir "$OUT/ed6e/labels" --out-dir "$OUT/ed6e"
    ;;
  fig3i_select)
    "$PY" -u scripts/cardiomyocyte/cm_specificity.py --pseudobulk results/cardiomyocyte/atlas/mean_normcounts_L3.parquet \
        --out results/cardiomyocyte/atlas/snrna_cm_specificity_allgenes.csv
    "$PY" -u scripts/cardiomyocyte/fig3i_cm_filter.py --tab "$OUT/fig3i" \
        --specificity results/cardiomyocyte/atlas/snrna_cm_specificity_allgenes.csv --out-prefix "$OUT/fig3i/fig3i_cm_filter"
    ;;
  fig3l)
    DVPA="$(cfg get dvp.all)"; DVPC="$(cfg get dvp.candidates)"
    "$PY" -u scripts/cardiomyocyte/transmural_slopes.py --tab "$OUT/fig3l/pb2um" --dvp "$DVPA" --out-dir "$OUT/fig3l/slopes2um"
    "$PY" -u scripts/cardiomyocyte/transmural_slopes.py --tab "$OUT/fig3l/pb2um_3donor" --dvp "$DVPA" \
        --donors H0054,H0063,AH10 --out-dir "$OUT/fig3l/slopes2um_3donor"
    # LABEL=... names the substrate in the output tables (config panels.fig3l_substrate picks one)
    CT=concordance_CM_protein_vs_transcript.csv
    "$PY" -u scripts/cardiomyocyte/fig3l_concordance.py --dvp-all "$DVPA" --dvp-cand "$DVPC" \
        --specificity results/cardiomyocyte/atlas/snrna_cm_specificity_allgenes.csv --out-prefix "$OUT/fig3l/fig3l_concordance" \
        --substrate "cmflag_2um_2donor=$OUT/fig3l/slopes2um/$CT:transcript_CM_slope:protein_CM_consensus" \
        --substrate "cmflag_2um_3donor=$OUT/fig3l/slopes2um_3donor/$CT:transcript_CM_slope:protein_CM_consensus"
    ;;
  env)
    "$PY" -c "import numpy, scipy, pandas, anndata, h5py, statsmodels, openpyxl; print('env ok')"
    ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "[$(date +%H:%M:%S)] done $STEP"
