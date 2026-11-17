#!/usr/bin/env Rscript
# Run one R deconvolution method (RCTD or SPOTlight) on one slide with the shared HHCA reference.
#
# Input:  REF_DIR/ref_matrix.mtx.gz, ref_genes.txt, ref_meta.csv (reference; prep_inputs.py)
#         DATA_DIR/<SLIDE>/matrix.mtx.gz, genes.txt, barcodes.txt, coords.csv (slide)
# Output: OUT_ROOT/<METHOD>/<SLIDE>.csv.gz (spots x cell types, rows sum to 1) and <SLIDE>.json
#
# Usage: Rscript run_r_method.R METHOD SLIDE REF_DIR DATA_DIR OUT_ROOT CELLTYPES [SEED]
suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
  METHOD <- args[1]; SLIDE <- args[2]
  REF <- args[3]; DATA_ROOT <- args[4]; OUT_ROOT <- args[5]; CTS_FILE <- args[6]
  SEED <- if (length(args) >= 7) as.integer(args[7]) else 0L
  library(Matrix)
}))
t0 <- proc.time()["elapsed"]
elapsed <- function() as.numeric(proc.time()["elapsed"] - t0)
logm <- function(...) cat(sprintf("[%6.1fs] ", elapsed()), ..., "\n")

DATA <- file.path(DATA_ROOT, SLIDE)
OUTD <- file.path(OUT_ROOT, METHOD); dir.create(OUTD, recursive = TRUE, showWarnings = FALSE)

# cell types, in output column order
CTS <- readLines(CTS_FILE)
CTS <- CTS[nchar(CTS) > 0]

# reference
logm("loading reference mtx")
ref_counts <- readMM(gzfile(file.path(REF, "ref_matrix.mtx.gz")))         # genes x cells
ref_genes <- readLines(file.path(REF, "ref_genes.txt"))
ref_meta <- read.csv(file.path(REF, "ref_meta.csv"), stringsAsFactors = FALSE)
rownames(ref_counts) <- ref_genes; colnames(ref_counts) <- ref_meta$barcode
ct <- as.character(ref_meta$cell_type_L3)

# slide
logm("loading slide mtx")
st_counts <- readMM(gzfile(file.path(DATA, "matrix.mtx.gz")))             # genes x spots
st_genes <- readLines(file.path(DATA, "genes.txt"))
st_bc <- readLines(file.path(DATA, "barcodes.txt"))
rownames(st_counts) <- st_genes; colnames(st_counts) <- st_bc
coords <- read.csv(file.path(DATA, "coords.csv"), stringsAsFactors = FALSE)
rownames(coords) <- coords$barcode

common <- intersect(rownames(ref_counts), rownames(st_counts))
logm("common genes:", length(common))
ref_counts <- ref_counts[common, ]; st_counts <- st_counts[common, ]

finalize <- function(prop_mat, n_genes_used, extra = list()) {
  # prop_mat: spots x cell types (any subset); written for all barcodes and all cell types, rows sum to 1
  prop_mat <- as.matrix(prop_mat)
  full <- matrix(0.0, nrow = length(st_bc), ncol = length(CTS),
                 dimnames = list(st_bc, CTS))
  cc <- intersect(colnames(prop_mat), CTS)
  rr <- intersect(rownames(prop_mat), st_bc)
  full[rr, cc] <- prop_mat[rr, cc]
  rs <- rowSums(full); rs[rs == 0] <- 1
  full <- full / rs
  df <- data.frame(barcode = rownames(full), full, check.names = FALSE)
  gz <- gzfile(file.path(OUTD, paste0(SLIDE, ".csv.gz")), "w")
  write.csv(df, gz, row.names = FALSE); close(gz)
  meta <- sprintf('{"method":"%s","slide":"%s","runtime_sec":%.1f,"n_genes_used":%d,"n_spots":%d,"n_celltypes":%d,"seed":%d}',
                  METHOD, SLIDE, elapsed(), as.integer(n_genes_used), nrow(full), ncol(full), SEED)
  writeLines(meta, file.path(OUTD, paste0(SLIDE, ".json")))
  logm("wrote", file.path(OUTD, paste0(SLIDE, ".csv.gz")), "runtime", round(elapsed(),1), "s")
}

compute_mgs <- function(ref_counts, ct, n_top = 100) {
  # markers per type: top n genes by log fold change (type vs rest) on log1p CP10k
  libs <- Matrix::colSums(ref_counts); libs[libs == 0] <- 1
  cpm <- ref_counts %*% Diagonal(x = 1e4 / libs)
  logn <- log1p(cpm)
  types <- sort(unique(ct))
  mean_in <- sapply(types, function(tt) Matrix::rowMeans(logn[, ct == tt, drop = FALSE]))
  grand <- Matrix::rowMeans(logn)
  mgs <- do.call(rbind, lapply(types, function(tt) {
    fc <- mean_in[, tt] - (grand * length(ct) - mean_in[, tt] * sum(ct == tt)) /
                          max(1, (length(ct) - sum(ct == tt)))
    fc[is.na(fc)] <- 0
    ord <- order(fc, decreasing = TRUE)
    keep <- ord[seq_len(min(n_top, sum(fc > 0)))]
    if (length(keep) == 0) keep <- ord[seq_len(min(n_top, length(ord)))]
    data.frame(gene = rownames(logn)[keep], cluster = tt, weight = pmax(fc[keep], 0),
               stringsAsFactors = FALSE)
  }))
  mgs
}

ok <- TRUE
res_err <- tryCatch({
set.seed(SEED)
if (METHOD == "RCTD") {
  library(spacexr)
  ctf <- factor(ct); names(ctf) <- colnames(ref_counts)
  reference <- Reference(ref_counts, ctf, colSums(ref_counts))
  has_xy <- all(c("x", "y") %in% colnames(coords))
  cdf <- if (has_xy) data.frame(x = coords[st_bc, "x"], y = coords[st_bc, "y"]) else
                     data.frame(x = seq_along(st_bc), y = rep(1, length(st_bc)))
  rownames(cdf) <- st_bc
  puck <- SpatialRNA(cdf, st_counts, colSums(st_counts))
  # one core: the parallel backend of RCTD is not reliable under SLURM, and full mode on a few thousand
  # spots is fast enough
  myRCTD <- create.RCTD(puck, reference, max_cores = 1, CELL_MIN_INSTANCE = 20)
  myRCTD <- run.RCTD(myRCTD, doublet_mode = "full")
  w <- as.matrix(myRCTD@results$weights)
  w <- sweep(w, 1, pmax(rowSums(w), 1e-9), "/")
  finalize(w, length(common))

} else if (METHOD == "SPOTlight") {
  library(SPOTlight)
  logm("computing marker genes")
  mgs <- compute_mgs(ref_counts, ct, n_top = 100)
  logm("running SPOTlight: mgs", nrow(mgs))
  res <- SPOTlight(x = as.matrix(ref_counts), y = as.matrix(st_counts),
                   groups = as.character(ct), mgs = mgs,
                   gene_id = "gene", group_id = "cluster", weight_id = "weight")
  mat <- if (!is.null(res$mat)) res$mat else res[[1]]
  rownames(mat) <- colnames(st_counts)
  finalize(mat, length(common))

} else {
  stop(paste("unknown method", METHOD))
}
NULL
}, error = function(e) { ok <<- FALSE; conditionMessage(e) })

if (!ok) {
  logm("failed:", res_err)
  writeLines(sprintf('{"method":"%s","slide":"%s","status":"FAILED","error":%s}',
             METHOD, SLIDE, shQuote(substr(res_err, 1, 400))),
             file.path(OUTD, paste0(SLIDE, ".FAILED.json")))
  quit(status = 3)
}
logm("done")
