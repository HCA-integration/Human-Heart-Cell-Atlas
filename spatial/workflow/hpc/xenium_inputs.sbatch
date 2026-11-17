#!/bin/bash
# Xenium, step 1: build the environments, download the public inputs (10x healthy section, GEO GSE290577),
# extract the Seurat objects to Matrix Market and build the CellPin query AnnData of both cohorts
# (ED6 i-j, Fig 6, ED9).
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/xenium_inputs.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_xenium_inputs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
INPUTS="${HHCA_XENIUM_INPUTS:-/path/to/hhca_inputs/xenium}"
OUT="$REPO/results/xenium"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT/healthy" "$OUT/rejection/extracted" "$TMPDIR"
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"

source workflow/hpc/env.sh
# micromamba installs the pip section of an env file through a helper that calls the binary by name, so the
# binary must be called `micromamba`; use a copy with that name (hhca_env honours HHCA_MAMBA).
if [ -z "${HHCA_MAMBA:-}" ]; then
  mkdir -p .tools/xenium_bin
  [ -x .tools/xenium_bin/micromamba ] || cp "$(hhca_mamba)" .tools/xenium_bin/micromamba
  export HHCA_MAMBA="$REPO/.tools/xenium_bin/micromamba"
fi
# rebuild the env if an interrupted build left it without cellpin
if [ -x .envs/xenium_cellpin/bin/python ] && ! .envs/xenium_cellpin/bin/python -c "import cellpin, torch" 2>/dev/null; then
  echo "incomplete .envs/xenium_cellpin, rebuilding"; rm -rf .envs/xenium_cellpin
fi
RENV="$(hhca_env xenium_r)"
PYENV="$(hhca_env xenium_cellpin)"
PY="$PYENV/bin/python"
"$PY" -c "import cellpin, torch" || { echo "xenium_cellpin env build failed"; exit 1; }
"$PY" -m pip freeze > "$OUT/env_xenium_cellpin.freeze.txt"
"$RENV/bin/Rscript" -e 'cat(R.version.string, "\n"); for (p in c("Seurat","SeuratObject","Matrix")) cat(p, as.character(packageVersion(p)), "\n")' \
  | tee "$OUT/env_xenium_r.versions.txt"
"$PY" -c "import cellpin, torch, scanpy; print('cellpin', cellpin.__version__, 'torch', torch.__version__, 'scanpy', scanpy.__version__)"

# ---- public inputs (the download scripts check the checksums of the published files) ----
bash scripts/xenium/download_10x_healthy.sh "$INPUTS/healthy_10x"
# GEO holds two versions of the object with the same counts: the label transfer reads
# GSE290577_heart_spatial_obj.rds (Mar 2025; query AnnData and author metadata of the label table), the
# downstream analyses read GSE290577_heart_spatial.rds (Aug 2025)
bash scripts/xenium/download_gse290577.sh "$INPUTS/GSE290577"

# ---- Seurat -> mtx (+ cell geometry) ----
"$RENV/bin/Rscript" scripts/xenium/extract_mtx.R "$INPUTS/GSE290577/GSE290577_heart_spatial_obj.rds" "$OUT/rejection/extracted"
# downstream input (config/xenium_downstream.yaml inputs.extracted): the Aug 2025 object
AUG="$OUT/inputs/gse290577_aug2025/extracted"
"$RENV/bin/Rscript" scripts/xenium/extract_mtx.R "$INPUTS/GSE290577/GSE290577_heart_spatial.rds" "$AUG"

# ---- query AnnData ----
"$PY" scripts/xenium/build_rejection_h5ad.py --extracted "$OUT/rejection/extracted" --out "$OUT/rejection/xenium_genes.h5ad"
"$PY" scripts/xenium/build_healthy_h5ad.py --matrix "$INPUTS/healthy_10x/cell_feature_matrix.h5" \
      --cells "$INPUTS/healthy_10x/cells.parquet" --out "$OUT/healthy/xenium_genes.h5ad"
echo "done $(date)"
