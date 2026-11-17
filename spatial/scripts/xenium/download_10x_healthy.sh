#!/bin/bash
# Download the public 10x Xenium dataset "Human Non-diseased Heart (FFPE)" (hMulti_v1, 377 genes, 26,366 cells,
# onboard segmentation) used for ED6 i. Only the three files the pipeline reads are fetched; the whole output
# bundle is ${BASE}/${DS}_outs.zip. Dataset page: https://www.10xgenomics.com/datasets
# Usage: bash download_10x_healthy.sh OUTDIR
set -euo pipefail
OUT="${1:?usage: download_10x_healthy.sh OUTDIR}"
DS=Xenium_V1_hHeart_nondiseased_section_FFPE
BASE=https://cf.10xgenomics.com/samples/xenium/1.9.0/$DS
mkdir -p "$OUT"
for f in cell_feature_matrix.h5 cells.parquet gene_panel.json; do
  if [ ! -s "$OUT/$f" ]; then
    curl -sSL --fail --retry 3 -o "$OUT/$f.part" "$BASE/${DS}_$f"
    mv "$OUT/$f.part" "$OUT/$f"
  fi
done
# checksums of the published files
cat > "$OUT/MD5SUMS.expected" <<'EOF'
0961ea9d952cab401adf3575c7c8cb27  cell_feature_matrix.h5
ca6ee4010b771ae77c652b831c44230d  cells.parquet
4ab0270260af7a1b8660d9cc4e721cb4  gene_panel.json
EOF
(cd "$OUT" && md5sum cell_feature_matrix.h5 cells.parquet gene_panel.json > MD5SUMS && md5sum -c MD5SUMS.expected)
