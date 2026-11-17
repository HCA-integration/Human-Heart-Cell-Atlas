#!/bin/bash
# square_002um layout of the 3 AH slides from Space Ranger feature_slice.h5 (their Space Ranger runs have no
# binned_outputs). Not part of run_all.sh: the decodes read the result from roots.spaceranger_ah.
# Output: results/substrate/ah_regen/<slide>_GRCh38-2020-A/outs/binned_outputs/square_002um/
# Usage (from the repository root):
#   FEATURE_SLICE_ROOT=<folder with <slide>_GRCh38-2020-A/outs/feature_slice.h5> \
#     sbatch "${SBATCH_CPU[@]}" workflow/hpc/substrate_regen_ah.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_substrate_regen_ah
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
: "${FEATURE_SLICE_ROOT:?set FEATURE_SLICE_ROOT to the directory holding <slide>_GRCh38-2020-A/outs/feature_slice.h5}"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"
source workflow/hpc/env.sh
PY="$(hhca_env substrate)/bin/python"
CFG=config/substrate.yaml
OUT="$REPO/$($PY scripts/substrate/config_get.py $CFG key roots.results)/ah_regen"
for S in VS_AH9_LV_FW_epi VS_AH10_LV_mid_endo VS_AH10_LV_mid_epi; do
  D="$OUT/${S}_GRCh38-2020-A/outs/binned_outputs/square_002um"
  "$PY" -u scripts/substrate/regen_002um_from_feature_slice.py \
    --feature-slice "$FEATURE_SLICE_ROOT/${S}_GRCh38-2020-A/outs/feature_slice.h5" --out-dir "$D"
done
echo "done $(date)"
