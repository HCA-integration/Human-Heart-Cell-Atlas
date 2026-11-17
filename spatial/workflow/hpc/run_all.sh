#!/bin/bash
# Submit every pipeline of this repository to SLURM, with job dependencies.
#
# Usage (from the repository root, on the login node):
#   bash workflow/hpc/run_all.sh                         # every block
#   bash workflow/hpc/run_all.sh cardiomyocyte figures   # only these blocks
#   RESUME=logs/run_all_<stamp>.tsv bash workflow/hpc/run_all.sh vascular
#       resubmit blocks; dependencies on jobs of an earlier submission are read from its TSV, and jobs that
#       have finished are dropped from the dependency list
#
# Blocks, in submission order:
#   atlas              atlas_scores -> atlas_hormones -> atlas_resources
#   substrate          substrate_envs -> substrate_model -> substrate_decode[0-10] -> substrate_certainty
#                      -> substrate_h5ad[0-10]
#   cmflag             substrate_cmflag_env; per slide substrate_cmflag_flags -> substrate_cmflag_clean
#   vascular           vascular (Fig 4 g-k tables and panels)
#   cardiomyocyte      cardiomyocyte_step: env, then one job per step (per slide for niche and label_pb)
#   nerve_macrophage   nerve_macrophage_env -> nerve_macrophage
#   deconvolution      deconvolution_envs -> deconvolution_inputs, deconvolution_reference -> deconvolution_submit.sh
#   xenium             xenium_inputs -> xenium_reference {healthy,rejection} -> xenium_cellpin {healthy,rejection}
#   xenium_downstream  xenium_downstream (Fig 6, ED9, ED6 i-j)
#   figures            figures_render (all other panels, after the pipelines)
#
# Three steps need extra inputs and are run by hand (see the README): the Fig 4 f fields
# (scripts/vascular/fig4f_fields.py), the AH 2 um matrices from Space Ranger feature_slice.h5
# (substrate_regen_ah.sbatch) and Fig 3 k.
#
# Partitions and QOS come from workflow/hpc/cluster.sh; HHCA_SBATCH_EXTRA adds sbatch options to every job.
# Job ids are written to logs/run_all_<stamp>.tsv (block, step, job id, dependency, command). About 45 jobs of
# this graph run at the same time at most.
set -euo pipefail

[ -f workflow/hpc/env.sh ] || { echo "run from the repository root" >&2; exit 2; }
mkdir -p logs .tmp
STAMP="$(date +%Y%m%d_%H%M%S)"
TSV="logs/run_all_${STAMP}.tsv"
printf "block\tstep\tjobid\tdependency\tcommand\n" > "$TSV"
source workflow/hpc/cluster.sh

ALL_BLOCKS=(atlas substrate cmflag vascular cardiomyocyte nerve_macrophage deconvolution xenium xenium_downstream figures)
BLOCKS=("$@"); [ ${#BLOCKS[@]} -eq 0 ] && BLOCKS=("${ALL_BLOCKS[@]}")
want() { local b; for b in "${BLOCKS[@]}"; do [ "$b" = "$1" ] && return 0; done; return 1; }

# the 11 Visium HD sections (config/substrate.yaml `slides`; array index order of substrate_decode / substrate_h5ad)
SLIDES=(VS_BO_HDZ150319_34_myo VS_BO_2024BH-008_myo VS_CA_H0054LV_henricke_endo VS_CA_H0054LV_henricke_epi
        VS_CA_H0054LV_henricke_myo VS_CA_H0063LV_henricke_endo VS_CA_H0063LV_henricke_epi VS_CA_H0063LV_henricke_myo
        VS_AH9_LV_FW_epi VS_AH10_LV_mid_endo VS_AH10_LV_mid_epi)

declare -A J     # step name -> job id
if [ -n "${RESUME:-}" ]; then
  while IFS=$'\t' read -r _b step id _rest; do
    [ "$step" = step ] && continue
    [ -n "$id" ] && J[$step]="$id"
  done < "$RESUME"
  echo "resume: ${#J[@]} job ids from $RESUME"
fi

# alive ID -> true if the job (any array task) is not finished; finished jobs are dropped from dependencies
alive() {
  local st
  st=$(sacct -n -X -P -j "$1" -o State 2>/dev/null | sort -u | tr '\n' ' ')
  [ -z "$st" ] && return 0                           # not in accounting yet: just submitted, keep it
  [[ "$st" == *PENDING* || "$st" == *RUNNING* || "$st" == *REQUEUED* || "$st" == *SUSPENDED* ]]
}
# dep ok STEP... [any STEP...] -> "--dependency=afterok:..,afterany:.." (steps never submitted are skipped)
dep() {
  local kind="" ok="" an="" s id
  for s in "$@"; do
    case "$s" in
      ok|any) kind=$s; continue ;;
    esac
    id="${J[$s]:-}"; [ -z "$id" ] && continue
    if [ -n "${RESUME:-}" ] && ! alive "$id"; then continue; fi
    if [ "$kind" = ok ]; then ok="${ok:+$ok:}$id"; else an="${an:+$an:}$id"; fi
  done
  local d=""; [ -n "$ok" ] && d="afterok:$ok"; [ -n "$an" ] && d="${d:+$d,}afterany:$an"
  [ -n "$d" ] && echo "--dependency=$d"
  return 0
}
# jid STEP -> job id for scripts that take ids as arguments ("none" if never submitted or already finished)
jid() {
  local id="${J[$1]:-}"
  if [ -z "$id" ] || { [ -n "${RESUME:-}" ] && ! alive "$id"; }; then echo none; else echo "$id"; fi
}
# sub BLOCK STEP "DEPFLAG" [VAR=value ...] -- sbatch arguments...
sub() {
  local block="$1" step="$2" depf="$3"; shift 3
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  local cmd=(sbatch --parsable "${SBATCH_CPU[@]}" ${depf:+"$depf"} "$@")
  local id
  id=$(env "${envs[@]}" "${cmd[@]}")
  id="${id%%;*}"
  J[$step]="$id"
  printf "%-18s %-40s %s %s\n" "$block" "$step" "$id" "$depf"
  printf "%s\t%s\t%s\t%s\t%s\n" "$block" "$step" "$id" "$depf" "${envs[*]:+${envs[*]} }${cmd[*]}" >> "$TSV"
}

echo "repository: $PWD ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "blocks: ${BLOCKS[*]}"

# Environments build inside the first job of each pipeline and share one micromamba package cache (its lock
# serialises concurrent builds). The substrate envs are on the critical path (decode ~2.5 h), so they build first;
# the first jobs of atlas, cmflag, cardiomyocyte, nerve_macrophage and xenium start after them (afterany: a failed
# build there does not hold the others). The deconvolution envs (longest pipeline) are not held.
if want substrate; then
  sub substrate substrate_envs "" -- workflow/hpc/substrate_envs.sbatch
fi
GATE="$(dep any substrate_envs)"

# ---------------------------------------------------------------- atlas
if want atlas; then
  sub atlas atlas_scores    "$GATE" -- workflow/hpc/atlas_scores.sbatch
  sub atlas atlas_hormones  "$(dep ok atlas_scores)" -- workflow/hpc/atlas_hormones.sbatch
  sub atlas atlas_resources "$(dep ok atlas_hormones)" -- workflow/hpc/atlas_resources.sbatch
fi

# ---------------------------------------------------------------- substrate, FICTURE
if want substrate; then
  sub substrate substrate_model     "$(dep ok substrate_envs)" -- workflow/hpc/substrate_model.sbatch
  sub substrate substrate_decode    "$(dep ok substrate_model)" -- --array=0-10 workflow/hpc/substrate_decode.sbatch
  sub substrate substrate_certainty "$(dep ok substrate_decode)" -- workflow/hpc/substrate_certainty.sbatch
  sub substrate substrate_h5ad      "$(dep ok substrate_certainty)" -- --array=0-10 workflow/hpc/substrate_h5ad.sbatch
fi

# ---------------------------------------------------------------- substrate, CMFLAG
if want cmflag; then
  sub cmflag cmflag_env "$GATE" -- workflow/hpc/substrate_cmflag_env.sbatch
  for s in "${SLIDES[@]}"; do
    sub cmflag "cmflag_$s" "$(dep ok cmflag_env substrate_h5ad)" \
        -- --job-name="cmflag_${s#VS_}" workflow/hpc/substrate_cmflag_flags.sbatch "$s"
    sub cmflag "cmclean_$s" "$(dep ok "cmflag_$s")" \
        -- --job-name="cmclean_${s#VS_}" workflow/hpc/substrate_cmflag_clean.sbatch "$s"
  done
fi
FLAGS_ALL=(); CLEAN_ALL=()
for s in "${SLIDES[@]}"; do FLAGS_ALL+=("cmflag_$s"); CLEAN_ALL+=("cmclean_$s"); done

# ---------------------------------------------------------------- vascular
if want vascular; then
  sub vascular vascular "$(dep ok substrate_h5ad "${FLAGS_ALL[@]}")" -- workflow/hpc/vascular.sbatch
fi

# ---------------------------------------------------------------- cardiomyocyte
if want cardiomyocyte; then
  CM=workflow/hpc/cardiomyocyte_step.sbatch
  sub cardiomyocyte cm_env "$GATE" -- --mem=8G --time=04:00:00 $CM env
  sub cardiomyocyte cm_atlas_pb "$(dep ok cm_env)" -- --mem=32G $CM atlas_pb
  sub cardiomyocyte cm_msearch  "$(dep ok cm_env "${CLEAN_ALL[@]}")" -- $CM msearch
  sub cardiomyocyte cm_pb2um    "$(dep ok cm_env "${CLEAN_ALL[@]}")" -- --mem=24G $CM pb2um
  sub cardiomyocyte cm_stress   "$(dep ok cm_env "${CLEAN_ALL[@]}")" -- --mem=32G $CM stress
  sub cardiomyocyte cm_fields   "$(dep ok cm_env "${CLEAN_ALL[@]}")" -- --mem=24G $CM fields
  sub cardiomyocyte cm_fig1j_locate "$(dep ok cmflag_env "${FLAGS_ALL[@]}")" CM_ENV=substrate_cmflag \
      -- --mem=32G $CM fig1j_locate
  CN=(); LP=()
  for s in "${SLIDES[@]}"; do
    sub cardiomyocyte "cm_niche_$s" "$(dep ok cm_env "${FLAGS_ALL[@]}")" -- --mem=24G $CM niche "$s"
    CN+=("cm_niche_$s")
  done
  sub cardiomyocyte cm_solidify "$(dep ok "${CN[@]}")" -- --mem=8G $CM solidify
  for s in "${SLIDES[@]}"; do
    sub cardiomyocyte "cm_label_pb_$s" "$(dep ok cm_env "cmclean_$s")" -- --mem=16G $CM label_pb "$s"
    LP+=("cm_label_pb_$s")
  done
  sub cardiomyocyte cm_ed6e         "$(dep ok cm_atlas_pb "${LP[@]}")" -- --mem=16G $CM ed6e
  sub cardiomyocyte cm_fig3i_select "$(dep ok cm_atlas_pb cm_msearch)" -- --mem=8G --time=01:00:00 $CM fig3i_select
  sub cardiomyocyte cm_fig3l        "$(dep ok cm_fig3i_select cm_pb2um)" -- --mem=8G --time=01:00:00 $CM fig3l
fi

# ---------------------------------------------------------------- nerve_macrophage
if want nerve_macrophage; then
  sub nerve_macrophage nm_env "$GATE" -- workflow/hpc/nerve_macrophage_env.sbatch
  # all steps (nerves d e gh kl fields); kl reads the vascular calibre table
  sub nerve_macrophage nm "$(dep ok nm_env "${CLEAN_ALL[@]}" vascular)" -- workflow/hpc/nerve_macrophage.sbatch
fi

# ---------------------------------------------------------------- deconvolution
if want deconvolution; then
  sub deconvolution deconv_envs      "" -- workflow/hpc/deconvolution_envs.sbatch
  sub deconvolution deconv_inputs    "$(dep ok deconv_envs)" -- workflow/hpc/deconvolution_inputs.sbatch
  # the NNLS step reads the substrate model pseudobulk
  sub deconvolution deconv_reference "$(dep ok deconv_envs substrate_model)" -- workflow/hpc/deconvolution_reference.sbatch
  # method, ground-truth, summary and downstream jobs; the ground-truth build and the HD area fraction read the
  # substrate decodes (third argument)
  out=$(bash workflow/hpc/deconvolution_submit.sh "$(jid deconv_inputs)" "$(jid deconv_reference)" "$(jid substrate_decode)")
  echo "$out"
  for kv in $(echo "$out" | grep '^ids:' | grep -o '[a-zA-Z0-9_]*=[0-9]*'); do
    k="deconv_${kv%%=*}"; v="${kv#*=}"; J[$k]="$v"
    printf "%s\t%s\t%s\t%s\t%s\n" deconvolution "$k" "$v" "(deconvolution_submit.sh)" "bash workflow/hpc/deconvolution_submit.sh" >> "$TSV"
  done
fi

# ---------------------------------------------------------------- xenium
if want xenium; then
  # public inputs (10x healthy section, GEO GSE290577) are downloaded into the checkout, not into the shared inputs
  XDL="$PWD/results/xenium/inputs/downloads"
  sub xenium xenium_inputs "$GATE" HHCA_XENIUM_INPUTS="$XDL" -- workflow/hpc/xenium_inputs.sbatch
  for c in healthy rejection; do
    sub xenium "xenium_reference_$c" "$(dep ok xenium_inputs)" -- workflow/hpc/xenium_reference.sbatch "$c"
    sub xenium "xenium_cellpin_$c" "$(dep ok "xenium_reference_$c")" -- workflow/hpc/xenium_cellpin.sbatch "$c"
  done
fi

# ---------------------------------------------------------------- xenium downstream
if want xenium_downstream; then
  sub xenium_downstream xenium_downstream "$(dep ok xenium_inputs xenium_cellpin_healthy xenium_cellpin_rejection)" \
      -- workflow/hpc/xenium_downstream.sbatch
fi

# ---------------------------------------------------------------- figures
if want figures; then
  FIG_AFTER=(atlas_hormones vascular cm_fig3l cm_ed6e cm_solidify cm_stress cm_fields cm_fig1j_locate
             nm deconv_summary deconv_downstream xenium_downstream)
  sub figures figures_render "$(dep any "${FIG_AFTER[@]}")" -- workflow/hpc/figures_render.sbatch
fi

echo "job ids: $TSV"
