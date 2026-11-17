#!/bin/bash
# Download the adult-heart Visium objects of Kanemaru et al. 2023 (Nature 619:801; OCT cryosections of
# LA, RA, LV and RV) from heartcellatlas.org (ArrayExpress E-MTAB-12975). Public, no credentials needed.
# The *_lognormalised.h5ad objects hold log-normalised expression and the authors' cell2location result
# (proportions in obsm, pericyte abundances PC1_vent ... PC4_CMC-like in obs); Fig 3 c and ED8 b-c read them.
# Writes the four objects, their checksums (MD5SUMS) and their URL, size and ETag (SOURCE.tsv) to OUTDIR.
# Usage: bash scripts/deconvolution/download_kanemaru.sh OUTDIR
set -euo pipefail
OUT="${1:?usage: download_kanemaru.sh OUTDIR}"
BASE="https://cellgeni.cog.sanger.ac.uk/heartcellatlas"
mkdir -p "$OUT"
files=()
for r in LA RA LV RV; do
  files+=("visium/visium-OCT_${r}_lognormalised.h5ad")
done
: > "$OUT/SOURCE.tsv.part"
for f in "${files[@]}"; do
  name=$(basename "$f")
  hdr=$(curl -sSI --fail "$BASE/$f" | tr -d '\r')
  size=$(echo "$hdr" | awk 'tolower($1)=="content-length:"{print $2}')
  etag=$(echo "$hdr" | awk 'tolower($1)=="etag:"{print $2}')
  printf "%s\t%s\t%s\t%s\n" "$name" "$BASE/$f" "$size" "$etag" >> "$OUT/SOURCE.tsv.part"
  if [ -f "$OUT/$name" ] && [ "$(stat -c %s "$OUT/$name")" = "$size" ]; then
    echo "have $name ($size B)"; continue
  fi
  echo "get $name ($size B)"
  curl -sSL --fail --retry 5 --retry-delay 20 -C - -o "$OUT/$name.part" "$BASE/$f"
  [ "$(stat -c %s "$OUT/$name.part")" = "$size" ] || { echo "size mismatch $name" >&2; exit 1; }
  mv "$OUT/$name.part" "$OUT/$name"
done
mv "$OUT/SOURCE.tsv.part" "$OUT/SOURCE.tsv"
( cd "$OUT" && md5sum visium-OCT_*_lognormalised.h5ad > MD5SUMS )
cat "$OUT/MD5SUMS"
