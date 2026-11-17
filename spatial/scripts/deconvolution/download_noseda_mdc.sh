#!/bin/bash
# Fetch the Space Ranger outputs of the Noseda standard-Visium cohort from the data providers' SFTP server
# (unpublished cohort without an accession; access to the server is restricted).
# Per slide: filtered_feature_bc_matrix.h5, metrics_summary.csv and spatial/ (five slides have no spatial/).
# Server and login come from the environment: HHCA_MDC_HOST (sftp://host:port), HHCA_MDC_USER and
# HHCA_MDC_PASSWORD (leave the password unset to be prompted by lftp).
# Usage: bash scripts/deconvolution/download_noseda_mdc.sh OUTDIR [slide ...]
set -euo pipefail
OUT="${1:?usage: download_noseda_mdc.sh OUTDIR [slide ...]}"; shift || true
HOST="${HHCA_MDC_HOST:?set HHCA_MDC_HOST (sftp://host:port)}"
USER_="${HHCA_MDC_USER:?set HHCA_MDC_USER}"
REMOTE_ROOT="${HHCA_MDC_ROOT:-/Noseda_Visium_standard}"
SPATIAL=(HHH1-LA HHH2-LV-endo-1 HHH4-RA2 HHH4-RV2 A17-AH1-C1 A18-AH2-D1 HHH2-D1 HHH2-LV-endo-2 HHH4-A1 HHH5-B1 HHH5-C1)
MATONLY=(HHH2-LV-epi-1 HHH2-LV-epi-2 HHH2-RA HHH4-LA HHH4-LV-epi)
SLIDES=("$@"); [ ${#SLIDES[@]} -gt 0 ] || SLIDES=("${SPATIAL[@]}" "${MATONLY[@]}")
command -v lftp >/dev/null || { echo "lftp not found" >&2; exit 1; }
AUTH="$USER_"; [ -n "${HHCA_MDC_PASSWORD:-}" ] && AUTH="$USER_,$HHCA_MDC_PASSWORD"
for S in "${SLIDES[@]}"; do
  mkdir -p "$OUT/$S/spatial"
  sp=""; for x in "${SPATIAL[@]}"; do [ "$x" = "$S" ] && sp=1; done
  {
    echo "set sftp:auto-confirm yes; set net:timeout 30; set net:max-retries 3; set xfer:clobber yes"
    echo "lcd $OUT/$S"
    echo "get $REMOTE_ROOT/$S/outs/filtered_feature_bc_matrix.h5 -o filtered_feature_bc_matrix.h5"
    echo "get $REMOTE_ROOT/$S/outs/metrics_summary.csv -o metrics_summary.csv"
    [ -n "$sp" ] && echo "mirror --verbose $REMOTE_ROOT/$S/outs/spatial spatial"
    echo "bye"
  } | lftp -u "$AUTH" "$HOST"
done
( cd "$OUT" && find . -type f | sort | xargs md5sum > MD5SUMS )
echo "wrote $OUT/MD5SUMS"
