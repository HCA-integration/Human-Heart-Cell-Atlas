#!/bin/bash
# Submit the method, ground-truth, summary and downstream jobs of the deconvolution pipeline with their
# dependencies. Arrays are throttled so that at most about 12 of these jobs run at once.
# Usage (from the repository root, on the login node):
#   bash workflow/hpc/deconvolution_submit.sh INPUTS_JOBID REFERENCE_JOBID [DECODE_JOBID]
# INPUTS_JOBID, REFERENCE_JOBID: jobs of deconvolution_inputs.sbatch and deconvolution_reference.sbatch
# DECODE_JOBID: the substrate_decode array, if still running (the ground-truth build and the HD area fraction
# read results/substrate/decode/). Pass "none" for a job that has already finished.
set -euo pipefail
I="${1:?usage: deconvolution_submit.sh INPUTS_JOBID REFERENCE_JOBID [DECODE_JOBID]}"; R="${2:?}"; DEC="${3:-none}"
source workflow/hpc/cluster.sh
# deps "ok" ID... ["any" ID...] -> one --dependency=afterok:a:b,afterany:c:d flag ("none" ids skipped)
deps() {
  local kind="" ok="" an="" j
  for j in "$@"; do
    case "$j" in ok|any) kind=$j ;; none|"") ;; *) [ "$kind" = ok ] && ok="${ok:+$ok:}$j" || an="${an:+$an:}$j" ;; esac
  done
  local d=""; [ -n "$ok" ] && d="afterok:$ok"; [ -n "$an" ] && d="${d:+$d,}afterany:$an"
  [ -n "$d" ] && echo "--dependency=$d" || echo ""
}
# submit_cpu / submit_gpu DEPENDENCY SBATCH_ARGS...: DEPENDENCY is the output of deps, left out when empty
submit_cpu() { local dep="$1"; shift; sbatch --parsable "${SBATCH_CPU[@]}" ${dep:+"$dep"} "$@"; }
submit_gpu() { local dep="$1"; shift; sbatch --parsable "${SBATCH_GPU[@]}" ${dep:+"$dep"} "$@"; }
GPU=workflow/hpc/deconvolution_gpu.sbatch; CPU=workflow/hpc/deconvolution_cpu.sbatch; GT=workflow/hpc/deconvolution_gtsim.sbatch

# benchmark, 11 slides
c2lt=$(submit_gpu "$(deps ok "$R")" --job-name=hhca_deconv_c2ltrain $GPU c2l_train benchmark)
c2l=$(submit_gpu "$(deps ok "$c2lt")" --array=0-10%2 --job-name=hhca_deconv_c2l $GPU c2l benchmark)
stt=$(submit_gpu "$(deps ok "$R")" --job-name=hhca_deconv_stereotrain $GPU stereo_train benchmark)
ste=$(submit_gpu "$(deps ok "$stt")" --array=0-10%2 --job-name=hhca_deconv_stereo $GPU stereo benchmark)
spo=$(submit_gpu "$(deps ok "$R")" --array=0-10%2 --job-name=hhca_deconv_spotiphy $GPU spotiphy benchmark)
uca=$(submit_cpu "$(deps ok "$R")" --array=0-10%2 --job-name=hhca_deconv_ucas $CPU ucaspatial benchmark)
rctd=$(submit_cpu "$(deps ok "$R")" --array=0-10%3 --job-name=hhca_deconv_rctd $CPU RCTD benchmark)
spl=$(submit_cpu "$(deps ok "$R")" --array=0-10%2 --job-name=hhca_deconv_spotlight $CPU SPOTlight benchmark)

# ground-truth simulation (4 HD sections); methods start once the benchmark arrays are through
gb=$(submit_cpu "$(deps ok "$R" "$DEC")" --array=0-3%2 --job-name=hhca_deconv_gtbuild $GT build)
gp=$(submit_cpu "$(deps ok "$gb")" --job-name=hhca_deconv_gtprep $GT prep)
AFTER=(any "$c2l" "$ste" "$spo" "$uca" "$spl")
g1=$(submit_gpu "$(deps ok "$gp")" --job-name=hhca_deconv_gt_c2ltrain $GPU c2l_train gtsim)
g2=$(submit_gpu "$(deps ok "$g1")" --array=0-3%2 --job-name=hhca_deconv_gt_c2l $GPU c2l gtsim)
g3=$(submit_gpu "$(deps ok "$gp")" --job-name=hhca_deconv_gt_stereotrain $GPU stereo_train gtsim)
g4=$(submit_gpu "$(deps ok "$g3")" --array=0-3%2 --job-name=hhca_deconv_gt_stereo $GPU stereo gtsim)
g5=$(submit_gpu "$(deps ok "$gp" "${AFTER[@]}")" --array=0-3%2 --job-name=hhca_deconv_gt_spotiphy $GPU spotiphy gtsim)
g6=$(submit_cpu "$(deps ok "$gp" "${AFTER[@]}")" --array=0-3%2 --job-name=hhca_deconv_gt_ucas $CPU ucaspatial gtsim)
g7=$(submit_cpu "$(deps ok "$gp" "${AFTER[@]}")" --array=0-3%2 --job-name=hhca_deconv_gt_rctd $CPU RCTD gtsim)
g8=$(submit_cpu "$(deps ok "$gp" "${AFTER[@]}")" --array=0-3%2 --job-name=hhca_deconv_gt_spotlight $CPU SPOTlight gtsim)
g9=$(submit_cpu "$(deps ok "$gp")" --array=0-3%2 --job-name=hhca_deconv_gt_nnls $CPU gt_nnls gtsim)
gs=$(submit_cpu "$(deps any "$g2" "$g4" "$g5" "$g6" "$g7" "$g8" "$g9")" --job-name=hhca_deconv_gtscore $GT score)

# method summaries and downstream panels
summ=$(submit_cpu "$(deps ok "$DEC" any "$c2l" "$ste" "$spo" "$uca" "$rctd" "$spl" "$gs")" --job-name=hhca_deconv_summary workflow/hpc/deconvolution_summary.sbatch)
dwn=$(submit_cpu "$(deps ok "$R" "$I")" --job-name=hhca_deconv_downstream workflow/hpc/deconvolution_downstream.sbatch)

cat <<EOF
benchmark: c2l_train=$c2lt c2l=$c2l stereo_train=$stt stereo=$ste spotiphy=$spo ucaspatial=$uca RCTD=$rctd SPOTlight=$spl
gtsim: build=$gb prep=$gp c2l_train=$g1 c2l=$g2 stereo_train=$g3 stereo=$g4 spotiphy=$g5 ucaspatial=$g6 RCTD=$g7 SPOTlight=$g8 NNLS=$g9 score=$gs
summary=$summ downstream=$dwn
ids: c2l_train=$c2lt c2l=$c2l stereo_train=$stt stereo=$ste spotiphy=$spo ucaspatial=$uca rctd=$rctd spotlight=$spl gt_build=$gb gt_prep=$gp gt_c2l_train=$g1 gt_c2l=$g2 gt_stereo_train=$g3 gt_stereo=$g4 gt_spotiphy=$g5 gt_ucaspatial=$g6 gt_rctd=$g7 gt_spotlight=$g8 gt_nnls=$g9 gt_score=$gs summary=$summ downstream=$dwn
EOF
