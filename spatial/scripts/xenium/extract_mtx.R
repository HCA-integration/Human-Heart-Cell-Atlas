#!/usr/bin/env Rscript
# Export the GSE290577 Seurat object to plain files: counts.mtx (genes x cells), genes.txt, barcodes.txt, obs.csv
# (author metadata) and cells_geom.csv (the obs columns plus centroids and areas, read by the geometry steps).
# Usage: Rscript extract_mtx.R OBJECT.rds OUTDIR
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
suppressWarnings(suppressMessages({library(Seurat); library(Matrix)}))
obj <- readRDS(args[1])
outdir <- args[2]
dir.create(outdir, showWarnings=FALSE, recursive=TRUE)
cnts <- GetAssayData(obj, assay="RNA", layer="counts")
cat("counts dim:", paste(dim(cnts),collapse=" x "), "class:", class(cnts)[1], "\n")
cat("range:", range(cnts), " integer-like:", all(cnts==round(cnts)), "\n")
cnts <- as(as.matrix(cnts), "CsparseMatrix")
cat("coerced to:", class(cnts)[1], " nnz:", length(cnts@x), "\n")
genes <- rownames(cnts)
writeMM(cnts, file.path(outdir,"counts.mtx"))
writeLines(genes, file.path(outdir,"genes.txt"))
writeLines(colnames(cnts), file.path(outdir,"barcodes.txt"))
md <- obj@meta.data
keep <- c("cell_id","Sample","patient_id","patient_biopsy","ct_second_pass","lineage",
          "biopsy_rejection_type","patient_rejection_type","biopsy_cellular_grading",
          "biopsy_antibody_grading","biopsy_timing","slide","age_at_biopsy","sex","total_counts",
          "nCount_Xenium","nFeature_Xenium")
keep <- keep[keep %in% colnames(md)]
obs <- md[, keep, drop=FALSE]; obs$barcode <- rownames(md)
write.csv(obs, file.path(outdir,"obs.csv"), row.names=FALSE)

# ---- geometry table ----
geo_cols <- c("cell_id","Sample","patient_id","patient_biopsy","ct_second_pass","lineage",
              "biopsy_rejection_type","patient_rejection_type","biopsy_cellular_grading",
              "biopsy_antibody_grading","biopsy_timing","slide","age_at_biopsy","sex",
              "x_centroid","y_centroid","adj_x_centroid","adj_y_centroid","cell_area","nucleus_area",
              "total_counts","nCount_Xenium","nFeature_Xenium")
miss <- setdiff(geo_cols, colnames(md))
if (length(miss)) cat("cells_geom: missing columns:", paste(miss, collapse=", "), "\n")
geo <- data.frame(barcode = rownames(md), md[, intersect(geo_cols, colnames(md)), drop=FALSE],
                  check.names = FALSE)
write.csv(geo, file.path(outdir,"cells_geom.csv"), row.names=FALSE)
cat("R:", R.version.string, " Seurat:", as.character(packageVersion("Seurat")),
    " SeuratObject:", as.character(packageVersion("SeuratObject")),
    " Matrix:", as.character(packageVersion("Matrix")), "\n")
