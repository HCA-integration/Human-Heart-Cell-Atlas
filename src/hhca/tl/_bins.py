"""The 2 um bins of a section: reading the count matrix in blocks and the expression of one cell type.

A section is an AnnData of 2 um bins with raw counts in ``X`` and ``obs`` columns ``L3_label``, ``x_um`` and
``y_um`` (the CMFLAG-clean h5ads of pipeline `substrate`). It can be opened with ``backed="r"``: the functions
here read ``X`` in row blocks.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

VENTRICULAR_CM = "Ventricular Cardiomyocytes"
CARDIOMYOCYTES = [VENTRICULAR_CM, "Atrial Cardiomyocytes"]


def _coordinates(adata):
    """Bin centres (x, y) in um as float64."""
    return adata.obs["x_um"].to_numpy().astype(np.float64), adata.obs["y_um"].to_numpy().astype(np.float64)


def _label_codes(adata, key="L3_label"):
    """Categories (str) and integer codes of a categorical obs column."""
    labels = adata.obs[key]
    return labels.cat.categories.to_numpy().astype(str), labels.cat.codes.to_numpy().astype(np.int64)


def _row_blocks(adata, chunk):
    """Yield (start, stop, CSR block of ``X``) over the rows of a section."""
    for start in range(0, adata.n_obs, chunk):
        stop = min(adata.n_obs, start + chunk)
        block = adata.X[start:stop]
        yield start, stop, block.tocsr() if sp.issparse(block) else sp.csr_matrix(block)


def _as_sections(sections):
    """A single section as a one-entry mapping; a mapping of sections unchanged."""
    return sections if isinstance(sections, dict) else {"section": sections}


def log_cp10k(sections, genes, cell_type=VENTRICULAR_CM, chunk=250_000):
    """Log-normalised expression of a few genes in the bins of one cell type.

    Each bin is scaled to 10,000 counts over all genes of the section and log1p-transformed; only `genes` are
    kept, so the result is small and dense.

    Parameters
    ----------
    sections
        One section, or a mapping of name -> section; the bins of all sections are concatenated.
    genes
        Genes to keep; genes missing from the sections are left out.
    cell_type
        Level-3 label of the bins to keep.
    chunk
        Number of bins read at a time.

    Returns
    -------
    AnnData of the kept bins x genes with log1p CP10k in ``X`` (float32), ``obs["section"]``, ``obs["x_um"]``,
    ``obs["y_um"]`` and ``obsm["spatial"]``.
    """
    sections = _as_sections(sections)
    expression, obs, shared = [], [], None
    for name, adata in sections.items():
        var_names = adata.var_names.to_numpy().astype(str)
        present = [gene for gene in genes if gene in set(var_names)]
        if shared is not None and present != shared:
            raise ValueError("the sections do not share the requested genes")
        shared = present
        columns = np.array([int(np.where(var_names == gene)[0][0]) for gene in present])
        categories, codes = _label_codes(adata)
        keep = categories[codes] == cell_type
        x, y = _coordinates(adata)
        total = np.zeros(int(keep.sum()))
        counts = np.zeros((int(keep.sum()), len(columns)), np.float32)
        filled = 0
        for start, stop, block in _row_blocks(adata, chunk):
            block = block[keep[start:stop]]
            rows = slice(filled, filled + block.shape[0])
            total[rows] = np.asarray(block.sum(axis=1)).ravel()
            counts[rows] = block[:, columns].toarray()
            filled += block.shape[0]
        expression.append(np.log1p(counts / np.maximum(total, 1)[:, None] * 1e4).astype(np.float32))
        obs.append(pd.DataFrame({"section": name, "x_um": x[keep], "y_um": y[keep]}))
    obs = pd.concat(obs, ignore_index=True)
    obs["section"] = pd.Categorical(obs["section"], categories=list(sections))
    obs.index = obs.index.astype(str)
    return ad.AnnData(
        X=expression[0] if len(expression) == 1 else np.concatenate(expression),
        obs=obs,
        var=pd.DataFrame(index=shared),
        obsm={"spatial": obs[["x_um", "y_um"]].to_numpy()},
    )


def celltype_bins(adata, cell_type=VENTRICULAR_CM, chunk=1_000_000):
    """The bins of one cell type with all genes, log-normalised, in memory.

    Counts are scaled to 10,000 per bin and log1p-transformed in single precision; the matrix stays sparse.

    Parameters
    ----------
    adata
        One section (in memory or backed).
    cell_type
        Level-3 label of the bins to keep.
    chunk
        Number of bins read at a time.

    Returns
    -------
    AnnData of the kept bins x all genes with log1p CP10k in ``X`` (CSR, float32), ``obs["x_um"]``,
    ``obs["y_um"]``, ``obsm["spatial"]`` and ``var["n_bins"]``, the number of kept bins with a count of the gene.
    """
    categories, codes = _label_codes(adata)
    keep = categories[codes] == cell_type
    matrix = sp.vstack([block[keep[start:stop]] for start, stop, block in _row_blocks(adata, chunk)]).tocsr()
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.int64)
    total = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    total[total == 0] = 1.0
    scale = np.repeat((1e4 / total).astype(np.float32), np.diff(matrix.indptr))
    matrix.data = matrix.data.astype(np.float32)
    matrix.data *= scale
    np.log1p(matrix.data, out=matrix.data)
    x, y = _coordinates(adata)
    obs = pd.DataFrame({"x_um": x[keep], "y_um": y[keep]}, index=adata.obs_names[keep].astype(str))
    var = pd.DataFrame({"n_bins": detected}, index=adata.var_names.astype(str))
    return ad.AnnData(X=matrix, obs=obs, var=var, obsm={"spatial": np.c_[x[keep], y[keep]]})
