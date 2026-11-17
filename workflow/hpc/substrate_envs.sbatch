#!/bin/bash
# Build the three environments of the FICTURE part of pipeline `substrate` (substrate, ficture, certainty)
# before the array jobs use them.
# Usage (from the repository root): sbatch "${SBATCH_CPU[@]}" workflow/hpc/substrate_envs.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_substrate_envs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR" logs
source workflow/hpc/env.sh
for e in substrate ficture certainty; do
  P="$(hhca_env $e)"; echo "$e -> $P"
done
# import test of each environment
"$REPO/.envs/substrate/bin/python" -c "import numpy, pandas, h5py, pyarrow, scipy, anndata, yaml"
"$REPO/.envs/ficture/bin/python" -c "import numpy, scipy, sklearn, pandas, ficture"
"$REPO/.envs/certainty/bin/python" -c "import numpy, pandas, scipy, sklearn, pyarrow"
echo "done $(date)"
