#!/bin/bash
# Deconvolution, reference and method inputs: 1,000 cells per Level-3 type from the HHCA atlas, inputs of the
# 11 benchmark slides, the shared HVG panel, marker scores, and NNLS on all 16 Noseda slides (signatures = the
# Level-3 atlas profile of the substrate pipeline).
# Usage (from the repository root, after deconvolution_envs.sbatch):
#   sbatch "${SBATCH_CPU[@]}" workflow/hpc/deconvolution_reference.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_deconv_reference
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
source workflow/hpc/deconvolution_common.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR" logs "$REFDIR" "$D/inputs" "$D/nnls" "$D/marker_scores"
source workflow/hpc/env.sh
PY="$(hhca_env deconv_cpu)/bin/python"
PYN="$(hhca_env deconv_nnls)/bin/python"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
B=$(IFS=,; echo "${BENCH_SLIDES[*]}")
# settings for the scripts of the NNLS environment, which has no YAML reader
CFG="$TMPDIR/deconvolution.json"; "$PY" scripts/deconvolution/deconv_config.py > "$CFG"

# reference (streamed from the atlas h5ad)
"$PY" scripts/deconvolution/build_reference.py --atlas "$ATLAS" --outdir "$REFDIR" --n-per-type 1000 --seed 0
# method inputs (+ reference mtx exports)
"$PY" scripts/deconvolution/prep_inputs.py --raw "$NOSEDA" --outdir "$D/inputs" --ref "$REF_H5" \
      --ref-outdir "$REFDIR" --summary "$D/inputs/prep_summary.csv" --slides "$B"
# HVG panel (11 benchmark slides) and marker scores
"$PY" scripts/deconvolution/make_gene_panel.py --ref "$REF_H5" --data "$D/inputs" --panel "$PANEL" \
      --celltypes "$CELLTYPES" --n-hvg 3000 --slides "$B"
"$PY" scripts/deconvolution/marker_scores.py --ref "$REF_H5" --data "$D/inputs" --celltypes "$CELLTYPES" \
      --outdir "$D/marker_scores" --markers-json "$REFDIR/markers_top15.json" --slides "$B"
# NNLS on all 16 slides
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYN" scripts/deconvolution/run_atlas_deconv.py --data "$NOSEDA" --pseudobulk "$PSEUDOBULK" --out "$D/nnls" \
        --config "$CFG"
echo "done $(date)"
