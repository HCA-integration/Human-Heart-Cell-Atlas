#!/bin/bash
# Deconvolution, downstream panels:
#   ED6 c, g   figure inputs: per-spot proportions of HHH2-LV-endo-1 (six methods and NNLS) and its H&E image
#   Fig 3 c    cardiomyocyte-dominant spots per chamber (Noseda NNLS; Kanemaru's own cell2location)
#   ED8 b-c    atrial-pericyte signature in pericyte-high spots (top within-region tertile of pericyte abundance)
# Usage (from the repository root, after deconvolution_inputs.sbatch and deconvolution_reference.sbatch):
#   sbatch "${SBATCH_CPU[@]}" workflow/hpc/deconvolution_downstream.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_deconv_downstream
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" || exit 1
source workflow/hpc/deconvolution_common.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
DS="$D/downstream"; mkdir -p "$TMPDIR" logs "$DS"
source workflow/hpc/env.sh
PYN="$(hhca_env deconv_nnls)/bin/python"
PY="$(hhca_env deconv_cpu)/bin/python"
# settings for the scripts of the NNLS environment, which has no YAML reader
CFG="$TMPDIR/deconvolution.json"; "$PY" scripts/deconvolution/deconv_config.py > "$CFG"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
CONC=resources/deconvolution/concordance_per_gene.csv
SIG=resources/deconvolution/atrial_pericyte_signature.json

echo "## figure inputs for ED6 c/g (renderers read results/ only)"
FI="$D/figure_inputs"; mkdir -p "$FI/precomputed/NNLS" "$FI/HHH2-LV-endo-1/spatial"
for m in spotiphy RCTD cell2location ucaspatial stereoscope SPOTlight; do
  mkdir -p "$FI/precomputed/$m"; cp "$PRECOMPUTED/outputs/$m/HHH2-LV-endo-1.csv.gz" "$FI/precomputed/$m/"
done
cp "$BENCHTAB/scanpy_analysis/standard_visium/atlas_deconv/tables/HHH2-LV-endo-1_L3_deconv.csv.gz" "$FI/precomputed/NNLS/"
cp "$NOSEDA/HHH2-LV-endo-1/spatial/tissue_hires_image.png" "$NOSEDA/HHH2-LV-endo-1/spatial/scalefactors_json.json" "$FI/HHH2-LV-endo-1/spatial/"

echo "## Fig 3 c"
"$PYN" scripts/deconvolution/build_cmdominant_panel.py --conc "$CONC" --noseda-h5ad "$D/nnls/h5ad" --kanemaru "$KANEMARU" \
      --out "$DS/fig3c" --config "$CFG"

echo "## ED8 b/c"
"$PYN" scripts/deconvolution/build_ED8c.py --kanemaru "$KANEMARU" --signature "$SIG" --out "$DS/ed8" --config "$CFG"
echo "done $(date)"
