#!/usr/bin/env bash
# Xenium downstream analyses (Fig 6 c-l, ED9 a-c, ED6 i-j): label tables, per-core features, statistics, panels.
# Settings: config/xenium_downstream.yaml. Inputs: results/xenium/{healthy,rejection}/xenium_cellpin_labels.csv
# and the extraction of GSE290577_heart_spatial.rds (results/xenium/inputs/gse290577_aug2025/extracted).
# The inputs are small (477 genes x 162,638 cells); the steps run one after the other.
# Usage (from the repository root): bash scripts/xenium/downstream/run_downstream.sh
# PY selects the interpreter (packages of envs/xenium_downstream.yml; default python3).
set -euo pipefail
PY=${PY:-python3}
py() { "$PY" -E -s "$@"; }
D=scripts/xenium/downstream
LOGS=results/xenium/downstream/logs
mkdir -p "$LOGS"

# gene sets and label tables
py $D/program_genes.py
py $D/label_tables.py
py $D/healthy_display_classes.py

# per-core features and scores
py $D/core_features.py > "$LOGS/core_features.log" 2>&1
py $D/core_geometry.py > "$LOGS/core_geometry.log" 2>&1
py $D/celltype_cytokine_scores.py > "$LOGS/celltype_cytokine_scores.log" 2>&1
py $D/drug_targets.py > "$LOGS/drug_targets.log" 2>&1

# statistics and panel tables
py $D/fig6_stats.py > "$LOGS/fig6_stats.log" 2>&1
py $D/cytokine_by_rejection.py > "$LOGS/cytokine_by_rejection.log" 2>&1
py $D/fig6e_cores.py > "$LOGS/fig6e_cores.log" 2>&1
for timing in pre post; do
  py $D/fig6d_overview.py --timing $timing >> "$LOGS/fig6d_overview.log" 2>&1
done
py $D/fig6j_all_tests.py > "$LOGS/fig6j_all_tests.log" 2>&1
py $D/spatial_cells.py > "$LOGS/spatial_cells.log" 2>&1

# panels
for f in figures/Fig6/panel_{c,d,e,fg,h,i,k,l}.py figures/ED9/panel_a.py figures/ED9/panel_bc.py figures/ED6/panel_j.py; do
  py "$f"
done
py figures/ED6/panel_i.py
