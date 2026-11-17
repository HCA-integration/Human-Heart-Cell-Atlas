#!/bin/bash
# Draw the figure panels from results/ into results/figures/<Fig>/, each in the environment of its pipeline.
# A panel that fails is logged in results/figures/render_status.tsv and the others still run. Fig 4 g-k and
# Fig 6, ED9, ED6 i-j are drawn by their own pipeline jobs (vascular, xenium_downstream).
# Usage (from the repository root, after the pipelines): sbatch "${SBATCH_CPU[@]}" workflow/hpc/figures_render.sbatch
# (SBATCH_CPU / SBATCH_GPU: workflow/hpc/cluster.sh)
#SBATCH --job-name=hhca_figures
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$TMPDIR" logs results/figures/logs
source workflow/hpc/env.sh
echo "job ${SLURM_JOB_ID:-NA} on $(hostname), $(date)"
STATUS=results/figures/render_status.tsv
printf "env\tpanel\targs\tstatus\n" > "$STATUS"
# render ENV SCRIPT [ARGS...]: run one panel script and record its status
render() {
  local env="$1" f="$2"; shift 2
  local py log tag
  py="$(hhca_env "$env")/bin/python" || { printf "%s\t%s\t%s\t%s\n" "$env" "$f" "$*" "env failed" >> "$STATUS"; return; }
  tag="$(echo "$f $*" | tr ' /' '__')"; log="results/figures/logs/$tag.log"
  if "$py" "$f" "$@" > "$log" 2>&1; then st=ok; else st="FAILED ($(tail -1 "$log" | cut -c1-120))"; fi
  printf "%s\t%s\t%s\t%s\n" "$env" "$f" "$*" "$st" | tee -a "$STATUS"
}

# atlas (Fig 2 h-l, Fig 4 l, Fig 5 m-n)
for f in figures/Fig2/panel_hi.py figures/Fig2/panel_jkl.py figures/Fig4/panel_l.py figures/Fig5/panel_m.py figures/Fig5/panel_n.py; do
  render atlas_h2c "$f"
done
# deconvolution (Fig 3 c, ED6 c, g, h, ED8 b-c)
render deconv_nnls figures/Fig3/panel_c.py
# ED6 c and g read config/deconvolution.yaml and need PyYAML, which deconv_nnls does not have
for p in c g; do render nerve_macrophage figures/ED6/panel_$p.py; done
render deconv_nnls figures/ED6/panel_h.py
render deconv_nnls figures/ED8/panel_b.py
render deconv_nnls figures/ED8/panel_c.py
# cardiomyocyte (Fig 1 j, Fig 3 d-l, ED6 d-f)
for f in figures/Fig1/panel_j.py figures/Fig3/panel_{d,e,f,g,h,i,l}.py figures/ED6/panel_{d,e,f}.py; do
  render cardiomyocyte "$f"
done
# nerve and macrophage (Fig 5 b-l)
for f in figures/Fig5/panel_{b,c,d,e,f,g,h,j,k,l}.py; do render nerve_macrophage "$f"; done

rm -rf "$TMPDIR"
echo "$(grep -c $'\tok$' "$STATUS") ok, $(grep -c 'FAILED\|env failed' "$STATUS") failed; $STATUS"
echo "done $(date)"
