#!/bin/bash
# Atlas-guided FICTURE decode of one Visium HD slide at 2 um with the fixed Level-3 model (8 Berlin slides):
# build input -> minibatch -> transform -> slda_decode. Steps whose output exists are skipped.
# The minibatch sort runs under LC_ALL=$SORT_LOCALE (default C).
# Usage:
#   run_ficture_decode.sh <square_002um dir> <run dir> <model tsv> <threads> <minibatch seed> <python> <ficture>
set -uo pipefail

SLIDE="${1:?square_002um dir}"
RUN="${2:?run dir}"
MODEL="${3:?model tsv}"
THREAD="${4:?thread}"
SEED="${5:?minibatch seed}"
PYTHON="${6:?python of the substrate env}"
FICTURE="${7:?ficture executable}"
SORT_LOCALE="${SORT_LOCALE:-C}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IN=$RUN/inputs
O=$RUN/output
LOG=$RUN/logs
mkdir -p "$IN" "$O" "$LOG"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] ==== FICTURE 2 um L3 decode: $RUN (thread=$THREAD seed=$SEED) ===="
ls -la "$SLIDE" "$SLIDE/spatial"

# ---- [1/4] FICTURE input (transcripts sorted by Y) --------------------------
if [ ! -s "$IN/transcripts.tsv.gz" ]; then
  echo "[$(ts)] [1/4] build FICTURE input"
  "$PYTHON" -u \
    "$HERE/build_input_2um.py" \
    --slide-dir "$SLIDE" --out-dir "$IN" --precision 2 \
    2>&1 | tee "$LOG/01_build_input.log"
else
  echo "[$(ts)] [1/4] transcripts.tsv.gz exists, skip"
fi
INPUT=$IN/transcripts.tsv.gz
FEAT=$IN/feature.clean.tsv.gz

# ---- [2/4] make_spatial_minibatch ------------------------------------------
if [ ! -s "$O/batched.matrix.tsv.gz" ]; then
  echo "[$(ts)] [2/4] make_spatial_minibatch"
  "$FICTURE" make_spatial_minibatch --input "$INPUT" --output "$O/batched.matrix.tsv" \
    --mu_scale 1.0 --batch_size 500 --batch_buff 30 --major_axis Y --seed "$SEED" \
    2>&1 | tee "$LOG/02_make_minibatch.log"
  LC_ALL=$SORT_LOCALE sort -T "$O" -k 2,2n -k 1,1g "$O/batched.matrix.tsv" | gzip -c > "$O/batched.matrix.tsv.gz"
  rm -f "$O/batched.matrix.tsv"
else
  echo "[$(ts)] [2/4] batched.matrix.tsv.gz exists, skip"
fi

# ---- [3/4] transform: project 12 um hexagons onto the fixed L3 model --------
PRJ=$O/atlas_L3.prj_12.r_4.0
if [ ! -s "${PRJ}.fit_result.tsv.gz" ]; then
  echo "[$(ts)] [3/4] ficture transform"
  "$FICTURE" transform --input "$INPUT" --output_pref "$PRJ" \
    --model "$MODEL" --feature "$FEAT" --key Count --major_axis Y \
    --hex_width 12 --n_move 3 --min_ct_per_unit 20 --mu_scale 1.0 \
    --thread "$THREAD" --precision 2 \
    2>&1 | tee "$LOG/03_transform_L3.log"
else
  echo "[$(ts)] [3/4] transform fit_result exists, skip"
fi

# ---- [4/4] slda_decode: pixel-level decode at 2 um --------------------------
DEC=$O/atlas_L3.decode.prj_12.r_4.0_5.0
if [ ! -s "${DEC}.pixel.tsv.gz" ]; then
  echo "[$(ts)] [4/4] ficture slda_decode"
  "$FICTURE" slda_decode --input "$O/batched.matrix.tsv.gz" \
    --output "$DEC" --model "$MODEL" --feature "$FEAT" \
    --anchor "${PRJ}.fit_result.tsv.gz" --anchor_in_um \
    --neighbor_radius 5.0 --mu_scale 1.0 --key Count --precision 0.1 \
    --lite_topk_output_pixel 5 --lite_topk_output_anchor 5 --thread "$THREAD" \
    2>&1 | tee "$LOG/04_slda_decode.log"
else
  echo "[$(ts)] [4/4] decode pixel output exists, skip"
fi
ls -la "${DEC}.pixel.tsv.gz" || exit 5
echo "[$(ts)] ==== done: $RUN ===="
