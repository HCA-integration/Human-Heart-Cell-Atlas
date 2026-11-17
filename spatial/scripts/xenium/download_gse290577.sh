#!/bin/bash
# Download the transplant-rejection Xenium cohort GSE290577 (Amancherla et al.; slides GSM8817228 / GSM8817229):
# the authors' processed Seurat objects (162,638 cells x 477 genes, author QC and per-core metadata) and the
# clinical metadata. GEO holds two versions of the object with the same counts, cells and genes:
# GSE290577_heart_spatial_obj.rds (Mar 2025; input of the CellPin label transfer) and GSE290577_heart_spatial.rds
# (Aug 2025; input of the downstream analyses), in which `lineage` reads "Stromal" instead of "Mesenchymal".
# Usage: bash download_gse290577.sh OUTDIR
set -euo pipefail
OUT="${1:?usage: download_gse290577.sh OUTDIR}"
BASE=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE290nnn/GSE290577/suppl
mkdir -p "$OUT"
FILES=(GSE290577_heart_spatial.rds GSE290577_heart_spatial_metadata.txt.gz GSE290577_README.txt GSE290577_heart_spatial_obj.rds)
for f in "${FILES[@]}"; do
  if [ ! -s "$OUT/$f" ]; then
    curl -sSL --fail --retry 3 -o "$OUT/$f.part" "$BASE/$f"
    mv "$OUT/$f.part" "$OUT/$f"
  fi
done
# checksums of the published files
cat > "$OUT/MD5SUMS.expected" <<'EOF'
68deb4e543f11ad6f1d94321821d3111  GSE290577_heart_spatial.rds
31bd9aa01cfa61aa8aedb34a29535a72  GSE290577_heart_spatial_metadata.txt.gz
d0a6e0a4c6cf9831c61d3118ee583a8c  GSE290577_heart_spatial_obj.rds
EOF
(cd "$OUT" && md5sum "${FILES[@]}" > MD5SUMS && md5sum -c MD5SUMS.expected)
