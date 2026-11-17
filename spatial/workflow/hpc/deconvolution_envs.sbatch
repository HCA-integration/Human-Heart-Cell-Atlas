#!/bin/bash
# Build the five deconvolution environments before any method job uses them: deconv_cpu (reference, inputs,
# UCASpatial), deconv_nnls (NNLS and downstream tables), deconv_spotiphy, deconv_cell2location (cell2location and
# stereoscope) and deconv_r (RCTD and SPOTlight). Package versions are written to results/deconvolution/envs/.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/deconvolution_envs.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_deconv_envs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR" logs results/deconvolution/envs
export PIP_CACHE_DIR="$REPO/.tmp/pipcache_deconv"; mkdir -p "$PIP_CACHE_DIR"
source workflow/hpc/env.sh
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
SPACEXR_SHA=9f5dc33c8060f946c6072a138b70e189636e1435
ENVOUT=results/deconvolution/envs

for e in deconv_cpu deconv_nnls deconv_spotiphy deconv_cell2location deconv_r; do
  P="$(hhca_env $e)"; echo "$e -> $P"
done

# spacexr 2.2.1 is not on conda for R 4.5: install it into deconv_r from a pinned commit
R_PREFIX="$REPO/.envs/deconv_r"
if ! "$R_PREFIX/bin/Rscript" -e 'quit(status = !requireNamespace("spacexr", quietly = TRUE))'; then
  curl -sSL --fail -o "$TMPDIR/spacexr.tar.gz" "https://github.com/dmcable/spacexr/archive/$SPACEXR_SHA.tar.gz"
  sha256sum "$TMPDIR/spacexr.tar.gz" | tee "$ENVOUT/spacexr_tarball.sha256"
  tar -xzf "$TMPDIR/spacexr.tar.gz" -C "$TMPDIR"
  R_MAKEVARS_USER=/dev/null "$R_PREFIX/bin/R" CMD INSTALL --no-test-load "$TMPDIR/spacexr-$SPACEXR_SHA"
fi

# package versions of each environment
for e in deconv_cpu deconv_spotiphy deconv_cell2location; do
  "$REPO/.envs/$e/bin/python" -m pip freeze --all > "$ENVOUT/$e.freeze.txt"
done
"$REPO/.envs/deconv_nnls/bin/python" -c "import sys, numpy, scipy, pandas, anndata, scanpy, h5py; print(sys.version.split()[0], numpy.__version__, scipy.__version__, pandas.__version__, anndata.__version__, scanpy.__version__, h5py.__version__)" | tee "$ENVOUT/deconv_nnls.versions.txt"
MAMBA_ROOT_PREFIX="$REPO/.tools/mamba_root" "$(hhca_mamba)" list -p "$REPO/.envs/deconv_nnls" > "$ENVOUT/deconv_nnls.list.txt"
"$REPO/.envs/deconv_cpu/bin/python" -c "import numpy, scipy, sklearn, scanpy, anndata, h5py; print('deconv_cpu', numpy.__version__, scipy.__version__, sklearn.__version__, scanpy.__version__, anndata.__version__, h5py.__version__)"
"$REPO/.envs/deconv_spotiphy/bin/python" -c "import torch, pyro, spotiphy, numpy, scanpy; print('deconv_spotiphy torch', torch.__version__, 'pyro', pyro.__version__, 'numpy', numpy.__version__, 'scanpy', scanpy.__version__)"
"$REPO/.envs/deconv_cell2location/bin/python" -c "import torch, scvi, cell2location, numpy, lightning; print('deconv_cell2location torch', torch.__version__, 'scvi', scvi.__version__, 'c2l', cell2location.__version__, 'numpy', numpy.__version__, 'lightning', lightning.__version__)"
"$R_PREFIX/bin/Rscript" -e 'for (p in c("SPOTlight","spacexr","Matrix","SingleCellExperiment")) cat(p, as.character(packageVersion(p)), "\n"); cat(R.version.string, "\n")' | tee "$ENVOUT/deconv_r.versions.txt"
MAMBA_ROOT_PREFIX="$REPO/.tools/mamba_root" "$(hhca_mamba)" list -p "$R_PREFIX" > "$ENVOUT/deconv_r.list.txt"
echo "done $(date)"
