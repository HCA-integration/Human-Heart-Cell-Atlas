# Paths and slide lists shared by the deconvolution jobs (same values as config/deconvolution.yaml).
# Sourced from the repository root by workflow/hpc/deconvolution_*.sbatch; HHCA_* variables override the roots.
# shellcheck disable=SC2034  # the variables are read by the jobs that source this file

INPUTS="${HHCA_VISIUM_INPUTS:-/path/to/hhca_inputs/visium}"
NOSEDA="$INPUTS/noseda"
KANEMARU="$INPUTS/kanemaru"
# benchmark tables: NNLS per-spot tables (ED6 c, g, h), HD nucleus-count and snRNA composition tables
# (endo-LV comparison)
BENCHTAB="$INPUTS/reference_deconvolution"
# benchmark per-spot outputs behind ED6 c, g, h (<PRECOMPUTED>/outputs/<method>/<slide>.csv.gz)
PRECOMPUTED="${HHCA_DECONV_PRECOMPUTED:-/path/to/hhca_precomputed/deconv_benchmark}"
# HHCA atlas
ATLAS="${HHCA_ATLAS:-/path/to/atlas/final_cleaned_hhca_annotated.h5ad}"
PSEUDOBULK="results/substrate/model/atlas_L3_pseudobulk.tsv"
FACTOR_ORDER="results/substrate/model/atlas_L3_factor_order.json"
DECODE_TEMPLATE="results/substrate/decode/{S}/output/atlas_L3.decode.prj_12.r_4.0_5.0.pixel.tsv.gz"

D="results/deconvolution"
REFDIR="$D/reference"
REF_H5="$REFDIR/hhca_L3_ref_1000.h5ad"
CELLTYPES="$REFDIR/hhca_L3_celltypes.txt"
PANEL="$REFDIR/gene_panel_hvg.txt"
SEED=0

BENCH_SLIDES=(HHH1-LA HHH2-LV-endo-1 HHH4-RA2 HHH4-RV2 A17-AH1-C1 A18-AH2-D1 HHH2-D1 HHH2-LV-endo-2 HHH4-A1 HHH5-B1 HHH5-C1)
GT_SLIDES=(VS_BO_HDZ150319_34_myo VS_CA_H0054LV_henricke_myo VS_CA_H0054LV_henricke_epi VS_CA_H0054LV_henricke_endo)

# dataset -> DATA (method inputs), OUT (proportions root), MODELS (trained c2l / stereoscope), SLIDES
deconv_dataset() {
  case "$1" in
    benchmark)   DATA="$D/inputs";          OUT="$D/outputs";             MODELS="$D/models";             SLIDES=("${BENCH_SLIDES[@]}") ;;
    gtsim)       DATA="$D/gtsim/data";      OUT="$D/gtsim/outputs/hhca";  MODELS="$D/gtsim/models/hhca";  SLIDES=("${GT_SLIDES[@]}") ;;
    *) echo "unknown dataset $1" >&2; return 1 ;;
  esac
}
