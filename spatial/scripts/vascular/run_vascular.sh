#!/bin/bash
# Pipeline `vascular` (tables behind Fig 4 g-k): runs every step, one slide and one Python process at a time.
#
# Usage (from the repository root): scripts/vascular/run_vascular.sh [--flags DIR] [--steps LIST]
#   --flags   directory with the CMFLAG flags <slide>_clean_bins.parquet of the 11 slides
#             (default: `flags` in config/vascular.yaml)
#   --steps   comma-separated subset of structures,caliber,g,h,ij,k (default: all); finished per-slide outputs
#             are skipped
# Inputs: config/vascular.yaml. Outputs: results/vascular/. Interpreter: $PY (default python3), extra interpreter
# flags in $PYFLAGS.
set -euo pipefail
FLAGS=""; STEPS="structures,caliber,g,h,ij,k"
while [ $# -gt 0 ]; do
  case "$1" in
    --flags) FLAGS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
PY="${PY:-python3}"
read -r -a PYFLAGS_ARGS <<< "${PYFLAGS:-}"
PYTHON=("$PY" ${PYFLAGS_ARGS[@]+"${PYFLAGS_ARGS[@]}"})
CFG=config/vascular.yaml
D=scripts/vascular
OUT=results/vascular
mkdir -p "$OUT/logs"
cfg() { "${PYTHON[@]}" "$D/paths.py" "$CFG" "$@"; }
has() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
# run LOGNAME SCRIPT ARGS...: network storage can drop for a few seconds, so a failed step is tried three times
# (steps skip outputs that are already complete)
run() {
  local log="$1"; shift
  local i
  for i in 1 2 3; do
    if "${PYTHON[@]}" -u "$@" >>"$OUT/logs/$log.log" 2>&1; then return 0; fi
    echo "attempt $i failed: $log" | tee -a "$OUT/logs/$log.log" >&2
    sleep 20
  done
  echo "FAILED: $log (see $OUT/logs/$log.log)" >&2
  return 1
}

berlin=$(cfg slides berlin); ah=$(cfg slides ah)
read -r -a BERLIN <<< "$berlin"; read -r -a AH <<< "$ah"; ALL=("${BERLIN[@]}" "${AH[@]}")
FORDER=$(cfg path factor_order)
[ -n "$FLAGS" ] || FLAGS=$(cfg get flags)
FLAGS=$(cd "$FLAGS" && pwd)
for s in "${ALL[@]}"; do [ -f "$FLAGS/${s}_clean_bins.parquet" ] || { echo "missing $FLAGS/${s}_clean_bins.parquet" >&2; exit 1; }; done
{
  echo "date=$(date) host=$(hostname)"
  echo "factor_order=$FORDER"
  echo "flags=$FLAGS"
} > "$OUT/run_info.txt"

# ---- vessel structures from the full decode, before CMFLAG (all 11 slides) ----
if has structures; then
  for s in "${ALL[@]}"; do
    run "structures_$s" "$D/vessel_structures.py" --slide "$s" --decode "$(cfg path decode "$s")" \
        --factor-order "$FORDER" --out-dir "$OUT/structures"
  done
fi
# ---- calibre; cut points from the Berlin slides ----
if has caliber; then
  run caliber "$D/caliber.py" --structures-dir "$OUT/structures" --factor-order "$FORDER" \
      --berlin "${BERLIN[@]}" --ah "${AH[@]}" --out-dir "$OUT/caliber"
fi
# ---- Fig 4 g: CMFLAG-clean vote recount, composition along calibre ----
if has g; then
  DECS=(); for s in "${ALL[@]}"; do DECS+=("$(cfg path decode "$s")"); done
  if [ ! -f "$OUT/composition/vessel_structs_cmflag_clean.csv" ]; then
    run g_votes "$D/struct_votes_clean.py" --slides "${ALL[@]}" --decodes "${DECS[@]}" \
        --structures-dir "$OUT/structures" --flags-dir "$FLAGS" --factor-order "$FORDER" --out "$OUT/composition/vessel_structs_cmflag_clean.csv"
  fi
  run g_curves "$D/composition_curves.py" --caliber "$OUT/caliber/caliber_per_structure.csv" \
      --votes "$OUT/composition/vessel_structs_cmflag_clean.csv" --out-dir "$OUT/composition"
fi
# ---- Fig 4 h: marker counts per vessel-wall compartment ----
if has h; then
  for s in "${ALL[@]}"; do
    run "h_$s" "$D/marker_aggregate.py" --slide "$s" --decode "$(cfg path decode "$s")" --factor-order "$FORDER" \
        --spaceranger "$(cfg path spaceranger "$s")" --structures-dir "$OUT/structures" \
        --out-dir "$OUT/markers/checkpoints" --flags "$FLAGS/${s}_clean_bins.parquet"
  done
  run h_pool "$D/marker_pool.py" --ckpt-dir "$OUT/markers/checkpoints" --out-dir "$OUT/markers"
fi
# ---- Fig 4 i, j: RAAS substrate and model ----
if has ij; then
  for s in "${ALL[@]}"; do
    run "ij_$s" "$D/raas_substrate.py" build --slide "$s" --spaceranger "$(cfg path spaceranger "$s")" \
        --common-panel "$(cfg get resources.common_panel)" --structures-dir "$OUT/structures" \
        --factor-order "$FORDER" --out-dir "$OUT/raas/substrate" --flags "$FLAGS/${s}_clean_bins.parquet"
  done
  read -r -a RAAS_POOL <<< "$(cfg slides raas_pool_order)"
  run ij_pool "$D/raas_substrate.py" pool --slides "${RAAS_POOL[@]}" --dir "$OUT/raas/substrate" \
      --out "$OUT/raas/pooled_struct_uniform.parquet"
  run ij_model "$D/raas_model.py" --parquet "$OUT/raas/pooled_struct_uniform.parquet" --out-dir "$OUT/raas/model"
fi
# ---- Fig 4 k: CMFLAG-clean per-structure pseudobulk from the 2 um h5ads, drug-target calibre ----
if has k; then
  CF=(); for s in $(cfg slides common_from); do CF+=("$(cfg path h5ad_2um "$s")"); done
  for s in "${BERLIN[@]}" "${AH[@]}"; do
    run "k_$s" "$D/struct_pb_h5ad.py" --slide "$s" --h5ad "$(cfg path h5ad_2um "$s")" \
        --flags "$FLAGS/${s}_clean_bins.parquet" --cells "$OUT/structures/${s}_cells.parquet" \
        --factor-order "$FORDER" --common-from "${CF[@]}" --out "$OUT/drug/struct_pb/struct_pb_cmflag_clean_$s.parquet"
  done
  run k_pool "$D/struct_pb_h5ad.py" --pool "$OUT/drug/struct_pb" --slides "${BERLIN[@]}" "${AH[@]}" \
      --out "$OUT/drug/struct_pb_cmflag_clean.parquet"
  run k_fit "$D/drug_caliber.py" --struct-pb "$OUT/drug/struct_pb_cmflag_clean.parquet" \
      --caliber "$OUT/caliber/caliber_per_structure.csv" --variant clean --out-dir "$OUT/drug/clean"
fi
echo "done: $OUT ($(date))" | tee -a "$OUT/run_info.txt"
